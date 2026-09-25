"""Core ML / ANE backbone 运行时。

fp16 backbone 图（见 scripts/21_convert_anemll_backbone.py）：
每次 predict 使用全新 KV state（v1：整条 state+questions 一次打包送入，
手册 6.7-A；prefix cache 留待 v0.4）。线程模型：一把锁串行化 predict。
"""
import threading
import time
import multiprocessing as mp
import atexit
import traceback
import hashlib
import json
import os
import platform
from pathlib import Path

import coremltools as ct
import numpy as np
import torch


def _compiled_backbone_path(mlpackage: str) -> tuple[Path, float]:
    """Compile once into the ignored build directory, then reuse the artifact.

    Loading an .mlpackage through MLModel asks Core ML to compile it at every
    service start. The compiled directory is keyed by source file metadata and
    runtime/OS version, and is installed by rename only after a full compile.
    """
    package = Path(mlpackage).resolve()
    sources = [
        package / "Manifest.json",
        package / "Data/com.apple.CoreML/model.mlmodel",
        package / "Data/com.apple.CoreML/weights/weight.bin",
    ]
    signature = {
        "package": str(package),
        "coremltools": ct.__version__,
        "macos": platform.mac_ver()[0],
        "files": [(str(path.relative_to(package)), path.stat().st_size,
                   path.stat().st_mtime_ns) for path in sources],
    }
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
    cache_dir = Path(__file__).resolve().parents[2] / "build" / "compiled"
    cache_dir.mkdir(parents=True, exist_ok=True)
    compiled = cache_dir / f"kevane-{digest}.mlmodelc"
    if compiled.is_dir():
        return compiled, 0.0
    temporary = cache_dir / f".kevane-{digest}-{os.getpid()}.mlmodelc"
    t0 = time.monotonic()
    try:
        ct.utils.compile_model(str(package), destination_path=str(temporary))
        # Another process may have completed the same compile while we worked;
        # rename only if ours is still needed, and tolerate losing the race.
        if not compiled.exists():
            try:
                temporary.rename(compiled)
            except OSError:
                pass
    finally:
        if temporary.exists():
            import shutil
            shutil.rmtree(temporary)
    return compiled, time.monotonic() - t0


class CoreMLBackbone:
    def __init__(self, mlpackage: str, seq_len: int, compute_units: str = "CPU_AND_NE"):
        self.seq_len = seq_len
        self.compute_units = compute_units
        compiled, self.compile_s = _compiled_backbone_path(mlpackage)
        t0 = time.monotonic()
        self.model = ct.models.CompiledMLModel(str(compiled), compute_units=getattr(ct.ComputeUnit, compute_units))
        self.load_s = time.monotonic() - t0
        self.lock = threading.Lock()
        self.calls = 0
        self.worker_pid = None
        self.restarts = 0
        self.last_worker_error = None

    def hidden(self, inputs: dict, L: int) -> torch.Tensor:
        """跑一次前向，返回前 L 个真实 token 的 final hidden [L, d]（fp32）。"""
        with self.lock:
            state = self.model.make_state()
            out = self.model.predict(inputs, state)
            self.calls += 1
        return torch.from_numpy(out["hidden_states"][0, :L, :].astype(np.float32))


def _backbone_worker(conn, mlpackage: str, seq_len: int, compute_units: str):
    """Own the native Core ML objects in a process that the API can replace."""
    try:
        backbone = CoreMLBackbone(mlpackage, seq_len, compute_units)
        conn.send(("ready", (backbone.compile_s, backbone.load_s)))
        while True:
            command = conn.recv()
            if command is None:
                break
            inputs, length = command
            try:
                hidden = backbone.hidden(inputs, length).numpy().copy()
                conn.send(("ok", hidden))
            except Exception:
                conn.send(("error", traceback.format_exc()))
    except EOFError:
        pass
    except Exception:
        try:
            conn.send(("fatal", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        conn.close()


class IsolatedCoreMLBackbone:
    """Keep the HTTP process alive across a native Core ML worker crash."""

    def __init__(self, mlpackage: str, seq_len: int, compute_units: str = "CPU_AND_NE",
                 startup_timeout: float = 90, predict_timeout: float = 30):
        self.mlpackage = mlpackage
        self.seq_len = seq_len
        self.compute_units = compute_units
        self.startup_timeout = startup_timeout
        self.predict_timeout = predict_timeout
        self.lock = threading.Lock()
        self.calls = 0
        self.restarts = 0
        self.last_worker_error = None
        self._ctx = mp.get_context("spawn")
        self._conn = None
        self._process = None
        self._start()
        atexit.register(self.close)

    @property
    def worker_pid(self) -> int | None:
        return self._process.pid if self._process and self._process.is_alive() else None

    def _receive(self, timeout: float):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._conn.poll(min(0.2, max(0, deadline - time.monotonic()))):
                return self._conn.recv()
            if not self._process.is_alive():
                raise RuntimeError(f"Core ML worker exited ({self._process.exitcode})")
        raise TimeoutError(f"Core ML worker timed out after {timeout}s")

    def _stop(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(timeout=5)
            else:
                self._process.join(timeout=0)
            self._process = None

    def _start(self):
        self._stop()
        parent, child = self._ctx.Pipe()
        process = self._ctx.Process(target=_backbone_worker,
                                    args=(child, self.mlpackage, self.seq_len, self.compute_units),
                                    daemon=True, name="kevane-coreml")
        process.start()
        child.close()
        self._conn = parent
        self._process = process
        try:
            status, payload = self._receive(self.startup_timeout)
            if status != "ready":
                raise RuntimeError(f"Core ML worker failed to load: {payload}")
            self.compile_s, self.load_s = payload
        except BaseException:
            self._stop()
            raise

    def hidden(self, inputs: dict, L: int) -> torch.Tensor:
        with self.lock:
            for attempt in range(2):
                try:
                    if not self._process.is_alive():
                        raise RuntimeError(f"Core ML worker exited ({self._process.exitcode})")
                    self._conn.send((inputs, L))
                    status, payload = self._receive(self.predict_timeout)
                    if status == "ok":
                        self.calls += 1
                        return torch.from_numpy(payload)
                    raise ValueError(f"Core ML prediction failed:\n{payload}")
                except (EOFError, BrokenPipeError, OSError, RuntimeError, TimeoutError) as exc:
                    self.last_worker_error = str(exc)
                    self.restarts += 1
                    if attempt:
                        raise RuntimeError("Core ML worker failed twice") from exc
                    self._start()
        raise RuntimeError("Core ML worker did not return a prediction")

    def close(self):
        with self.lock:
            self._stop()
