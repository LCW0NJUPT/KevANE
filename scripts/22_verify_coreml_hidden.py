#!/usr/bin/env python
"""22_verify_coreml_hidden.py — 阶段 3：Core ML / ANE backbone parity（实施手册 6.5-6.6）。

H1 hidden 参考对比：Core ML fp16 hidden vs merged_fp32_hidden_small.npz（cosine > 0.999）
H2 概率/argmax：hidden -> CPU PointerHead -> softmax，与官方 golden 概率比（argmax 100%）
H3 执行：CPU_AND_NE 加载跑通并记录延迟（placement 深度验证交给 23_profile_ane.sh）

用法：
    python scripts/22_verify_coreml_hidden.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kevane_common import ROOT, load_cases, require_kevane_runtime_env  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import coremltools as ct  # noqa: E402
from kev.api import SystemOneRequest, to_record  # noqa: E402
from kev.model import PointerHead, branch_mask_batch, encode as kev_encode  # noqa: E402

DEFAULT_MLPACKAGE = ROOT / "hf-model" / "kev-qwen3-0.6b-hidden.mlpackage"


def load_pointer_head():
    blob = torch.load(ROOT / "hf-model" / "kev-qwen3-0.6b-pointer-head.pt", map_location="cpu")
    head_dim = blob["meta"].get("head_dim", 256)
    head = PointerHead(1024, dp=head_dim)
    head.load_state_dict(blob["head"])
    head.temperature = float(blob["temperature"])
    head.eval()
    return head


def main():
    require_kevane_runtime_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlpackage", default=str(DEFAULT_MLPACKAGE))
    ap.add_argument("--hidden-ref", default=str(ROOT / "tests" / "fixtures" / "merged_fp32_hidden_small.npz"))
    ap.add_argument("--golden", default=str(ROOT / "tests" / "fixtures" / "official_fp32_probs.jsonl"))
    ap.add_argument("--compute-units", default="CPU_AND_NE",
                    choices=["CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"])
    ap.add_argument("--cosine-threshold", type=float, default=0.999)
    args = ap.parse_args()

    meta_path = Path(args.mlpackage).parent / (Path(args.mlpackage).stem + "-meta.json")
    seq = json.loads(meta_path.read_text())["seq_len"] if meta_path.exists() else 512
    print(f"[verify] mlpackage={Path(args.mlpackage).name} seq={seq} units={args.compute_units}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ROOT / "hf-model")
    head = load_pointer_head()
    refs = np.load(args.hidden_ref, allow_pickle=True)
    ref_meta = eval(str(refs["_meta"]))
    golden = {}
    for line in Path(args.golden).read_text(encoding="utf-8").splitlines()[1:]:
        row = json.loads(line)
        golden[row["id"]] = {o["qid"]: o for o in row["outputs"]}

    print("[verify] loading Core ML model ...")
    t0 = time.time()
    mlmodel = ct.models.MLModel(args.mlpackage, compute_units=getattr(ct.ComputeUnit, args.compute_units))
    print(f"[verify] loaded in {time.time() - t0:.1f}s")

    def run_coreml(ids, pos, seg, L):
        """一次 packed 前向；返回 h[:L] 为 torch fp32 [L, hidden]。"""
        p_ids = np.zeros((1, seq), dtype=np.int32); p_ids[0, :L] = ids
        p_pos = np.zeros((seq,), dtype=np.int32); p_pos[:L] = pos
        # 与基线完全相同的 branch mask 代码路径（dtype=fp16，pad 段被 branch_mask_batch 屏蔽）
        mask = branch_mask_batch([seg], "cpu", dtype=torch.float16, length=seq).numpy()
        state = mlmodel.make_state()
        out = mlmodel.predict({"input_ids": p_ids, "position_ids": p_pos, "causal_mask": mask}, state)
        return torch.from_numpy(out["hidden_states"][0, :L, :].astype(np.float32))

    cases = load_cases()

    # ---- H1: hidden 参考（4 个短 case + rand16/32/64，均为"state+单question"或纯 causal）----
    print("[verify] H1: hidden-state parity vs fp32 reference")
    h1 = []
    case_by_id = {c["id"]: c for c in cases}
    for m in ref_meta:
        name = m["name"]; L = m["L"]
        if m["kind"] == "case":
            req = SystemOneRequest(state=case_by_id[name]["state"], questions=case_by_id[name]["questions"])
            rec, _ = to_record(req)
            enc = kev_encode(tok, rec)
            h = run_coreml(enc["ids"], enc["pos"], enc["seg"], L)
        else:
            ids = refs[f"{name}_ids"].astype(np.int64).tolist()
            pos = list(range(L)); seg = [0] * L
            h = run_coreml(ids, pos, seg, L)
        ref_h = torch.from_numpy(refs[f"{name}_hidden"].astype(np.float32))
        cos = float(torch.nn.functional.cosine_similarity(h.flatten(), ref_h.flatten(), dim=0))
        maxabs = float((h - ref_h).abs().max())
        h1.append({"name": name, "L": L, "cosine": cos, "max_abs": maxabs})
        print(f"    {name:28s} L={L:4d} cos={cos:.6f} max|Δ|={maxabs:.4f}")

    # ---- H2: 全量 golden cases -> PointerHead 概率 vs 官方 ----
    print("[verify] H2: probabilities via CoreML hidden + CPU PointerHead")
    n_q, flips, boundary, worst = 0, [], [], 0.0
    # 手册 10.2：margin 低于 fp16 噪声包络（实测 |Δp| 包络 ~0.08）的翻转记为边界翻转
    BOUNDARY_MARGIN = 0.10
    t_warm = time.time()
    for case in cases:
        req = SystemOneRequest(state=case["state"], questions=case["questions"])
        rec, meta = to_record(req)
        enc = kev_encode(tok, rec)
        L = len(enc["ids"])
        t1 = time.time()
        h = run_coreml(enc["ids"], enc["pos"], enc["seg"], L)
        dt = (time.time() - t1) * 1000
        for d, oi, m in zip(enc["decide_idx"], enc["opt_idx"], meta):
            qid = m["id"]
            logits = head(h[d], h[torch.tensor(oi)])
            p = F.softmax(logits, -1).tolist()
            g = golden[case["id"]][qid]
            dp = max(abs(a - b) for a, b in zip(p, g["probabilities"]))
            am = max(range(len(p)), key=lambda i: p[i])
            n_q += 1
            worst = max(worst, dp)
            if am != g["argmax"]:
                s = sorted(g["probabilities"], reverse=True)
                margin = s[0] - s[1]
                rec_flip = (case["id"], qid, g["argmax"], am, margin)
                (boundary if margin < BOUNDARY_MARGIN else flips).append(rec_flip)
        print(f"    {case['id']:28s} L={L:4d} q={len(meta)} {dt:7.1f}ms")
    warm_total = time.time() - t_warm

    h1_ok = all(r["cosine"] > args.cosine_threshold for r in h1)
    h2_ok = not flips
    print(f"[verify] H1 {'PASS' if h1_ok else 'FAIL'}: min cosine={min(r['cosine'] for r in h1):.6f} "
          f"(threshold {args.cosine_threshold})")
    print(f"[verify] H2 {'PASS' if h2_ok else 'FAIL'}: argmax {n_q - len(flips) - len(boundary)}/{n_q}, "
          f"max|Δp|={worst:.4f}")
    for f in flips:
        print(f"    FLIP {f[0]}.{f[1]}: golden={f[2]} coreml={f[3]} (margin={f[4]:.3f})")
    for f in boundary:
        print(f"    BOUNDARY-FLIP {f[0]}.{f[1]}: golden={f[2]} coreml={f[3]} "
              f"(margin={f[4]:.3f} < {BOUNDARY_MARGIN}, 记录不判失败，手册 10.2)")
    print(f"[verify] total {len(cases)} cases in {warm_total:.1f}s "
          f"(avg {warm_total / len(cases) * 1000:.0f}ms/case)")
    sys.exit(0 if (h1_ok and h2_ok) else 1)


if __name__ == "__main__":
    main()
