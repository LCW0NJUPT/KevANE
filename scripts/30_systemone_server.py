#!/usr/bin/env python
"""30_systemone_server.py — 启动 KevANE System One 服务（阶段 4）。

用法（主环境）：
    conda activate kevane-runtime
    python scripts/30_systemone_server.py --port 8008
    KEVANE_API_KEY=local python scripts/30_systemone_server.py --port 8008   # 开启 Bearer 鉴权
"""
import argparse
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kevane_common import require_kevane_runtime_env  # noqa: E402


def main():
    require_kevane_runtime_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--model-dir", default=None, help="Hugging Face model directory (default: hf-model)")
    ap.add_argument("--mlpackage", default=None)
    ap.add_argument("--compute-units", default="CPU_AND_NE",
                    choices=("CPU_AND_NE", "CPU_ONLY", "CPU_AND_GPU", "ALL"),
                    help="Core ML allowed devices; this does not prove actual placement")
    args = ap.parse_args()

    import uvicorn
    from kevane.server import create_app, default_server

    # launchd may stop us during the expensive model load, before Uvicorn
    # installs its own signal handler. Convert SIGTERM into normal unwinding.
    def terminate(_signum, _frame):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate)
    server = None
    try:
        server = default_server(args.mlpackage, compute_units=args.compute_units, model_dir=args.model_dir)
        print(f"[kevane] Core ML backbone compiled in {server.backbone.compile_s:.1f}s, "
              f"loaded in {server.backbone.load_s:.1f}s "
              f"(seq={server.backbone.seq_len}, units={server.backbone.compute_units}); "
              f"serving {server.model_name} on {args.host}:{args.port}")
        uvicorn.run(create_app(server), host=args.host, port=args.port, log_level="info",
                    access_log=False)
    finally:
        if server is not None:
            server.backbone.close()


if __name__ == "__main__":
    main()
