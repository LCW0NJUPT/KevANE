#!/usr/bin/env python
"""01_baseline_official.py — 阶段 1：官方 Kev-0.6B(Qwen3) fp32 概率基线（实施手册 4.2 / 4.4）。

直接走 kev.checkpoint 库路径（不起 HTTP 服务），对 tests/fixtures/cases.jsonl 的每条
golden case 导出完整概率分布 + argmax 到 JSONL。这是后续 merged / Core ML 实现的
唯一参考输出（阶段 2/3/4 的 parity 都与这份文件比）。

用法：
    python scripts/01_baseline_official.py                    # 默认 0.6b@qwen3, mps, fp32
    python scripts/01_baseline_official.py --repeat 2         # 进程内跑两遍并断言逐位一致
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kevane_common import KEV_SOURCE_COMMIT, ROOT, load_cases, require_kevane_env  # noqa: E402

import torch  # noqa: E402
from kev.api import SystemOneRequest, to_record  # noqa: E402
from kev.checkpoint import Checkpoint, LoadOptions  # noqa: E402
from kev.device import sync  # noqa: E402


def run_pass(model, tok, cases):
    """跑一遍全部 cases；返回 (rows, 逐 case 延迟 ms)。probs() 是 exact 路径（无 prefix cache）。"""
    rows, latencies = [], []
    for case in cases:
        req = SystemOneRequest(state=case["state"], questions=case["questions"])
        rec, meta = to_record(req)
        enc = model.encode(tok, rec)          # 训练上下文（384/1024/2048），cases 已验证不截断
        sync(model.device); t = time.time()
        ps = model.probs(enc)                 # packed block-causal, fp32
        sync(model.device); latencies.append((time.time() - t) * 1000)
        outs = []
        for p, m in zip(ps, meta):
            p = p.tolist()
            outs.append({"qid": m["id"], "type": m["type"], "keys": m["keys"],
                         "probabilities": [round(x, 8) for x in p],
                         "argmax": max(range(len(p)), key=lambda i: p[i])})
        rows.append({"id": case["id"], "tokens": len(enc["ids"]), "state_tokens": enc["seg"].count(0),
                     "outputs": outs})
    return rows, latencies


def max_abs_diff(rows_a, rows_b):
    worst = 0.0
    for a, b in zip(rows_a, rows_b):
        for qa, qb in zip(a["outputs"], b["outputs"]):
            worst = max(worst, max(abs(x - y) for x, y in zip(qa["probabilities"], qb["probabilities"])))
    return worst


def main():
    require_kevane_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jaredpalmer/kev-0.6b@qwen3")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--cases", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--repeat", type=int, default=1, help="进程内重复次数（>=2 时断言逐位一致）")
    args = ap.parse_args()

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    cases = load_cases(args.cases)
    print(f"[baseline] model={args.model} device={args.device} dtype={args.dtype} cases={len(cases)}")

    t0 = time.time()
    ck = Checkpoint(args.model)
    opts = LoadOptions(dtype=dtype, merge=True, backend="torch")
    tok, model = ck.load(args.device, opts)
    cold_start_s = time.time() - t0
    print(f"[baseline] loaded in {cold_start_s:.1f}s via {model.backend} ({model.dtype}); "
          f"meta: base={ck.meta.base} lora={ck.meta.lora} head_dim={ck.meta.head_dim} "
          f"temperature={ck.meta.temperature} option_isolation={ck.meta.option_isolation}")

    all_passes, all_lat = [], []
    for i in range(max(args.repeat, 1)):
        rows, lat = run_pass(model, tok, cases)
        all_passes.append(rows); all_lat.append(lat)
        print(f"[baseline] pass {i + 1}: {len(rows)} cases, p50={sorted(lat)[len(lat) // 2]:.1f}ms")
    if args.repeat >= 2:
        for i in range(1, args.repeat):
            d = max_abs_diff(all_passes[0], all_passes[i])
            print(f"[baseline] determinism pass0 vs pass{i}: max|dp| = {d:.3e}")
            assert d == 0.0, "同一进程内两次运行不一致，检查确定性"

    header = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": "kev-official",
        "model_requested": args.model,
        "model_path": ck.path,
        "base": ck.meta.base,
        "base_revision": ck.meta.base_revision,
        "lora": ck.meta.lora,
        "head_dim": ck.meta.head_dim,
        "temperature": ck.meta.temperature,
        "option_isolation": ck.meta.option_isolation,
        "kev_backend": model.backend, "runtime_dtype": model.dtype, "device": model.device,
        "dtype": args.dtype,
        "kev_commit": KEV_SOURCE_COMMIT,
        "cold_start_s": round(cold_start_s, 2),
        "latency_ms_p50": round(sorted(all_lat[-1])[len(all_lat[-1]) // 2], 1),
    }

    out = Path(args.out) if args.out else ROOT / "tests" / "fixtures" / f"official_{args.dtype}_probs.jsonl"
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for row, lat in zip(all_passes[-1], all_lat[-1]):
            f.write(json.dumps({**row, "latency_ms": round(lat, 1)}, ensure_ascii=False) + "\n")
    print(f"[baseline] wrote {len(all_passes[-1]) + 1} lines -> {out}")


if __name__ == "__main__":
    main()
