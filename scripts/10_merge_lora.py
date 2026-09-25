#!/usr/bin/env python
"""10_merge_lora.py — 阶段 2：把 Kev LoRA 合并进 Qwen3-0.6B backbone（实施手册 5.2）。

产物：
    build/merged/kev-qwen3-0.6b-backbone/          merged backbone + tokenizer（无 lm_head）
    build/merged/kev-qwen3-0.6b-pointer-head.pt    PointerHead 权重 + temperature + meta

Core ML Runtime 第一版不动态理解 PEFT：fp32 加载 → merge_and_unload() → 落盘。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kevane_common import require_kevane_env  # noqa: E402

import torch  # noqa: E402
from kev.checkpoint import Checkpoint, LoadOptions  # noqa: E402

BACKBONE_DIR = "build/merged/kev-qwen3-0.6b-backbone"
HEAD_PT = "build/merged/kev-qwen3-0.6b-pointer-head.pt"


def main():
    require_kevane_env()
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jaredpalmer/kev-0.6b@qwen3")
    ap.add_argument("--out-dir", default="build/merged")
    args = ap.parse_args()

    ck = Checkpoint(args.model)
    t0 = time.time()
    # 手册 5.2：fp32 + merge=True；adapter 有 trainable_token_embeddings 时 Checkpoint 会自动不合并
    tok, model = ck.load("cpu", LoadOptions(dtype=torch.float32, merge=True, backend="torch"))
    assert not hasattr(model.lm, "peft_config"), "LoRA 未被合并，检查 adapter_config"
    print(f"[merge] loaded+merged on cpu in {time.time() - t0:.0f}s")

    backbone_dir = Path(args.out_dir) / "kev-qwen3-0.6b-backbone"
    backbone_dir.mkdir(parents=True, exist_ok=True)
    model.lm.save_pretrained(backbone_dir)
    tok.save_pretrained(backbone_dir)
    head_pt = Path(args.out_dir) / "kev-qwen3-0.6b-pointer-head.pt"
    torch.save({"head": model.head.state_dict(), "temperature": float(model.head.temperature),
                "meta": ck.meta.to_dict()}, head_pt)
    print(f"[merge] backbone -> {backbone_dir}")
    print(f"[merge] pointer head -> {head_pt}")
    print(f"[merge] merged dtype: {model.dtype}, hidden_size: {model.lm.config.hidden_size}, "
          f"layers: {model.lm.config.num_hidden_layers}")


if __name__ == "__main__":
    main()
