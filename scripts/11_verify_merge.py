#!/usr/bin/env python
"""11_verify_merge.py — 阶段 2 parity：merged backbone vs 官方基线（实施手册 5.3 / 5.4）。

验证层级（对 24 条 golden cases，全部与 tests/fixtures/official_fp32_probs.jsonl 比）：
    L1 tokenizer   ids/pos/decide_idx/opt_idx 必须完全一致（同一 tokenizer，结构性保证）
    L2 概率        max |Δp| 阈值（默认 1e-5，fp32 merge 数学上应达到 ~1e-6 量级）
    L3 argmax      100% 一致（硬性 Stop/Go）

用法：
    python scripts/11_verify_merge.py                    # 默认与 official_fp32_probs.jsonl 比
    python scripts/11_verify_merge.py --threshold 1e-5
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kevane_common import ROOT, load_cases, require_kevane_env  # noqa: E402

import torch  # noqa: E402
from kev.api import SystemOneRequest, to_record  # noqa: E402
from kev.model import DecisionModel  # noqa: E402
from kev.device import sync  # noqa: E402

BACKBONE_DIR = ROOT / "build" / "merged" / "kev-qwen3-0.6b-backbone"
HEAD_PT = ROOT / "build" / "merged" / "kev-qwen3-0.6b-pointer-head.pt"


def load_merged(device="mps", dtype=torch.float32):
    """从 merge 产物重建 DecisionModel：backbone 从本地目录加载，PointerHead 从 head.pt 加载。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BACKBONE_DIR)
    m = DecisionModel(str(BACKBONE_DIR), tok, device, lora=None, dtype=dtype)
    blob = torch.load(HEAD_PT, map_location="cpu")
    m.head.load_state_dict(blob["head"])
    m.head.temperature = float(blob["temperature"])
    m.eval()
    return tok, m, blob["meta"]


def main():
    require_kevane_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--reference", default=str(ROOT / "tests" / "fixtures" / "official_fp32_probs.jsonl"))
    ap.add_argument("--cases", default=None)
    ap.add_argument("--threshold", type=float, default=1e-5, help="max |Δp| 阈值（手册 5.3: 1e-5~1e-4）")
    args = ap.parse_args()

    cases = {c["id"]: c for c in load_cases(args.cases)}
    ref_lines = Path(args.reference).read_text(encoding="utf-8").splitlines()
    ref_header = json.loads(ref_lines[0])
    ref = {}
    for line in ref_lines[1:]:
        row = json.loads(line)
        ref[row["id"]] = {o["qid"]: o for o in row["outputs"]}
    missing = set(cases) - set(ref)
    assert not missing, f"参考输出缺 cases: {sorted(missing)}"
    print(f"[verify] reference: {ref_header['backend']} {ref_header['model_requested']} "
          f"({ref_header['dtype']}, kev@{ref_header['kev_commit'][:8]}), {len(ref)} cases")

    tok, model, meta = load_merged(args.device)
    print(f"[verify] merged backbone loaded: hidden={model.lm.config.hidden_size}, "
          f"temperature={model.head.temperature}")

    n_q, argmax_mismatch, worst_dp = 0, [], 0.0
    per_case = []
    for cid, case in cases.items():
        req = SystemOneRequest(state=case["state"], questions=case["questions"])
        rec, m = to_record(req)
        enc = model.encode(tok, rec)
        ps = model.probs(enc)
        sync(model.device)
        for p, mm, qid in zip(ps, m, case["questions"]):
            r = ref[cid][qid]
            p = p.tolist()
            n_q += 1
            dp = max(abs(a - b) for a, b in zip(p, r["probabilities"]))
            worst_dp = max(worst_dp, dp)
            per_case.append((dp, cid, qid))
            am = max(range(len(p)), key=lambda i: p[i])
            if am != r["argmax"]:
                argmax_mismatch.append((cid, qid, r["argmax"], am))

    per_case.sort(reverse=True)
    print(f"[verify] questions={n_q}  argmax 一致={n_q - len(argmax_mismatch)}/{n_q}  max|Δp|={worst_dp:.3e}")
    print("[verify] 最差 5 个 question：")
    for dp, cid, qid in per_case[:5]:
        print(f"    {dp:.3e}  {cid}.{qid}")

    ok = not argmax_mismatch and worst_dp < args.threshold
    if argmax_mismatch:
        print(f"[verify] ARGMAX 翻转: {argmax_mismatch}")
    print(f"[verify] {'PASS' if ok else 'FAIL'}  (argmax 100% 一致 且 max|Δp| < {args.threshold:g})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
