"""Portable Conda environment guards and golden-case helpers."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEV_SOURCE_COMMIT = "c9c1f855505336ac32092a5f68305d397f7fcc3e"
def _require_conda(purpose: str, env_name: str | None = None):
    # launchd starts the Conda Python executable directly and does not inherit
    # the shell's CONDA_PREFIX variable.
    if sys.version_info[:2] != (3, 12) or not (Path(sys.prefix) / "conda-meta").is_dir():
        raise SystemExit(f"ENV ERROR: activate a Python 3.12 Conda environment for {purpose}")
    # 同为 py3.12 的其他环境（如 runtime 里跑转换脚本）会在 import 阶段才失败，
    # 报错难以定位；这里直接比对环境名。KEVANE_SKIP_ENV_NAME_CHECK=1 可跳过。
    if env_name and os.environ.get("KEVANE_SKIP_ENV_NAME_CHECK") != "1" and Path(sys.prefix).name != env_name:
        raise SystemExit(f"ENV ERROR: {purpose} needs the '{env_name}' Conda environment "
                         f"(current: '{Path(sys.prefix).name}'). "
                         f"Set KEVANE_SKIP_ENV_NAME_CHECK=1 to override.")


def require_kevane_runtime_env():
    _require_conda("runtime", "kevane-runtime")


def require_kevane_coreml_env():
    _require_conda("conversion", "kevane-conversion")


def require_kevane_env(allowed=(), py_min=(3, 12)):
    _require_conda("build", "kevane-build")


def load_cases(path=None):
    """读取 golden cases（TypeSafe System One 请求格式），返回原始 dict 列表。"""
    path = Path(path) if path else ROOT / "tests" / "fixtures" / "cases.jsonl"
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        raise SystemExit(f"no cases in {path}")
    return cases


def git_commit(path):
    import subprocess
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "not-a-git-checkout"
