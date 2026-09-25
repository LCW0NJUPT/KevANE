#!/usr/bin/env python
"""21_convert_anemll_backbone.py — 阶段 3：merged backbone → Core ML（backbone hidden 输出，conversion pipeline）。

导出图（第一版：FP16、无 LUT、backbone-only、PointerHead 留在 CPU）：

    input_ids[1,SEQ] + position_ids[SEQ] + causal_mask[1,1,SEQ,SEQ]
        -> embed -> 28 层 prefill（KV state 以 0 偏移写入） -> final RMSNorm
    output: hidden_states[1,SEQ,hidden] fp16

关键点：ANEMLL prefill attention 使用任意 additive mask、RoPE 按显式 position_ids
逐 token 应用，因此 Kev 的 packed block-causal branch mask 与分支位置重启可直接复用。

⚠️ 转换环境：本脚本必须在 **kevane-conversion** 辅助环境运行（torch==2.5.0 +
numpy==1.26.4 + coremltools==9.0，即 ANEMLL requirements.txt 的验证组合）。
numpy>=2 会使 coremltools 9.0 的 torch 前端在 aten::Int 上抛
"only 0-dimensional arrays can be converted to Python scalars"（conversion note）。
运行用 kevane-runtime；合并 Kev LoRA 请使用独立的 kevane-build 环境。

用法：
    conda activate kevane-conversion
    python scripts/21_convert_anemll_backbone.py --seq-len 512
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "anemll"))

from kevane_common import KEV_SOURCE_COMMIT, ROOT, git_commit, require_kevane_coreml_env  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import coremltools as ct  # noqa: E402
from anemll.models import qwen_model as anemll_qwen  # noqa: E402
from anemll.models.qwen_model import QwenConfig, QwenForCausalLM, TEST_DEVICE  # noqa: E402

MERGED = ROOT / "build" / "merged" / "kev-qwen3-0.6b-backbone"
DEFAULT_OUT = ROOT / "hf-model" / "kev-qwen3-0.6b-hidden.mlpackage"


class BackboneHiddenWrapper(torch.nn.Module):
    """手册 6.3 的 QwenHiddenWrapper，按 ANEMLL 当前 commit 的实际签名实现：
    embed -> process_layers(IN_PREFILL=True) -> final RMSNorm，不经过 lm_head。"""

    def __init__(self, model: QwenForCausalLM):
        super().__init__()
        self.model = model
        self.states = [
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(2 * model.config.num_hidden_layers,
                           model.config.num_key_value_heads,
                           model.config.state_length,
                           model.config.head_dim),
                    dtype=np.float16,
                ),
                name="model.model.kv_cache_0",
            )
        ]

    def forward(self, input_ids, position_ids, causal_mask):
        m = self.model.model
        hidden = m.embed_tokens(input_ids)
        rotary = m.get_rotary_embedding_prefill(position_ids)
        out = m.process_layers(
            hidden, position_ids, causal_mask,
            torch.zeros(1, dtype=torch.int32),   # current_pos：v1 恒 0（KV 写 0 偏移）
            rotary, IN_PREFILL=True,
        )
        return m.norm(out)


def reset_kv_cache_buffers(module):
    """与 QwenConverter._reset_kv_cache_buffers 相同：trace 前后清零 KV buffer。"""
    with torch.no_grad():
        for name, buf in module.named_buffers():
            if "kv_cache_" in name:
                buf.zero_()


def main():
    require_kevane_coreml_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MERGED))
    ap.add_argument("--seq-len", type=int, default=512, help="静态序列/上下文长度（golden 最长 packed 371）")
    ap.add_argument("--dtype", default="fp16", choices=["fp16"], help="第一版只支持 fp16（手册 6.4：不上量化）")
    ap.add_argument("--lut", default="none", choices=["none"], help="第一版不量化")
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    model_dir = Path(args.model)
    seq = args.seq_len
    cfg_dict = json.loads((model_dir / "config.json").read_text())
    config = QwenConfig(**cfg_dict, context_length=seq, state_length=seq)
    print(f"[convert] {model_dir.name}: hidden={config.hidden_size} layers={config.num_hidden_layers} "
          f"heads={config.num_attention_heads} kv={config.num_key_value_heads} head_dim={config.head_dim} "
          f"ctx/state={seq}")

    t0 = time.time()
    model = QwenForCausalLM(config, enable_coreml=True).to(TEST_DEVICE).eval()
    if not model.model.load_pretrained_weights(str(model_dir)):
        raise SystemExit("权重加载失败：存在 missing keys（见上方日志）")
    print(f"[convert] weights loaded in {time.time() - t0:.0f}s (fp16)")

    wrapper = BackboneHiddenWrapper(model).eval()
    input_ids = torch.zeros((1, seq), dtype=torch.int32, device=TEST_DEVICE)
    position_ids = torch.zeros((seq,), dtype=torch.int32, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, seq, seq), dtype=torch.float16, device=TEST_DEVICE)

    reset_kv_cache_buffers(wrapper)
    t1 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (input_ids, position_ids, causal_mask))
    reset_kv_cache_buffers(wrapper)
    reset_kv_cache_buffers(traced)
    print(f"[convert] traced in {time.time() - t1:.0f}s")

    t2 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=input_ids.shape, dtype=np.int32),
            ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="hidden_states", dtype=np.float16)],
        states=wrapper.states,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    print(f"[convert] coreml conversion in {time.time() - t2:.0f}s")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out))
    sidecar = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_backbone": "Kev LoRA merged with Qwen/Qwen3-0.6B-Base",
        "seq_len": seq,
        "dtype": args.dtype, "lut": args.lut,
        "convert_env": "kevane-conversion (torch 2.5.0 + numpy 1.26.4 + coremltools 9.0)",
        "anemll_commit": git_commit(ROOT / "third_party" / "anemll"),
        "kev_commit": KEV_SOURCE_COMMIT,
        "inputs": {"input_ids": [1, seq], "position_ids": [seq], "causal_mask": [1, 1, seq, seq]},
        "output": "hidden_states [1, seq, hidden] fp16（final RMSNorm 后）",
        "notes": "KV state 以 0 偏移写入，调用方可忽略；mask 为 additive fp16，"
                 "branch mask 需 pad 到 [1,1,seq,seq]（finfo(fp16).min）",
    }
    (out.parent / (out.stem + "-meta.json")).write_text(json.dumps(sidecar, indent=2, ensure_ascii=False))
    print(f"[convert] saved -> {out}")
    print(f"[convert] meta   -> {out.parent / (out.stem + '-meta.json')}")


if __name__ == "__main__":
    main()
