#!/usr/bin/env python
"""20_export_hidden_reference.py — 导出 PyTorch final hidden state 参考（实施手册 6.5 / 10.3）。

对若干短序列（16/32/64 token 档）导出 packed 编码经 merged backbone（final RMSNorm 后）
的完整 [L, hidden] hidden states 到 tests/fixtures/merged_fp32_hidden_small.npz。
这是阶段 3 Core ML hidden-state parity 的对照参考。

输入故意用"state + 单 question"的短 case，并另加随机 id 序列作为最小可控样本。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kevane_common import ROOT, load_cases, require_kevane_env  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from kev.api import SystemOneRequest, to_record  # noqa: E402
from kev.device import sync  # noqa: E402

SHORT_CASES = ("tiny_state_noul_001", "short_state_intent_001", "jarvis_greeting_001", "disk_full_001")


def load_merged(device="cpu"):
    from kev.model import DecisionModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    backbone = ROOT / "build" / "merged" / "kev-qwen3-0.6b-backbone"
    head_pt = ROOT / "build" / "merged" / "kev-qwen3-0.6b-pointer-head.pt"
    tok = AutoTokenizer.from_pretrained(backbone)
    m = DecisionModel(str(backbone), tok, device, lora=None, dtype=torch.float32)
    blob = torch.load(head_pt, map_location="cpu")
    m.head.load_state_dict(blob["head"]); m.eval()
    return tok, m


def main():
    require_kevane_env()
    tok, model = load_merged("cpu")   # cpu fp32：与 ANE 侧比较时排除 MPS kernel 差异
    cases = {c["id"]: c for c in load_cases()}

    arrays, meta = {}, []
    # 1) 短 golden cases（真实分布的 packed 序列）
    for cid in SHORT_CASES:
        req = SystemOneRequest(state=cases[cid]["state"], questions=cases[cid]["questions"])
        rec, _ = to_record(req)
        enc = model.encode(tok, rec)
        with torch.no_grad():
            h = model.hidden(enc).numpy()  # [L, d] final RMSNorm 后
        arrays[f"{cid}_hidden"] = h
        arrays[f"{cid}_ids"] = np.array(enc["ids"], dtype=np.int32)
        arrays[f"{cid}_pos"] = np.array(enc["pos"], dtype=np.int32)
        arrays[f"{cid}_decide_idx"] = np.array(enc["decide_idx"], dtype=np.int32)
        meta.append({"name": cid, "L": len(enc["ids"]), "kind": "case"})

    # 2) 合成 id 序列（可控长度：16/32/64；无 special token，纯排错用）
    g = torch.Generator().manual_seed(42)
    for L in (16, 32, 64):
        ids = torch.randint(0, tok.vocab_size, (L,), generator=g).tolist()
        enc = {"ids": ids, "seg": [0] * L, "pos": list(range(L)), "opt": [-1] * L,
               "decide_idx": [], "opt_idx": []}
        ids_t = torch.tensor([enc["ids"]]); pos_t = torch.tensor([enc["pos"]])
        mask = torch.zeros(1, 1, L, L).masked_fill(
            torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1), torch.finfo(torch.float32).min)
        with torch.no_grad():
            h = model.lm(input_ids=ids_t, position_ids=pos_t, attention_mask=mask).last_hidden_state[0].float()
        arrays[f"rand{L}_hidden"] = h.numpy()
        arrays[f"rand{L}_ids"] = np.array(ids, dtype=np.int32)
        meta.append({"name": f"rand{L}", "L": L, "kind": "rand"})

    out = ROOT / "tests" / "fixtures" / "merged_fp32_hidden_small.npz"
    np.savez(out, **arrays, _meta=np.array(repr(meta)))
    sync(model.device)
    for m in meta:
        h = arrays[f"{m['name']}_hidden"]
        print(f"[hidden] {m['name']:28s} kind={m['kind']:4s} L={m['L']:4d} shape={h.shape}")
    print(f"[hidden] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
