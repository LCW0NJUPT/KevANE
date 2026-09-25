"""KevANE System One HTTP 服务（实施手册 §7）。

TypeSafe 兼容：POST /v1/systemone、GET /v1/models、x-typesafe-request-id
响应头，以及 KEVANE_API_KEY 设置时的 Bearer 鉴权（未设置 = 本地开放服务，
与 kev.serve 同策略）。响应体结构与官方 kev 语义一致（schema.to_answers）。
"""
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import threading
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .encoder import ContextOverflowANE, Encoder
from .pointer_head import PointerHeadRuntime
from .runtime import CoreMLBackbone, IsolatedCoreMLBackbone
from .schema import SystemOneRequest, output_tokens, to_answers, to_record

ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class KevANEServer:
    encoder: Encoder
    backbone: CoreMLBackbone | IsolatedCoreMLBackbone
    head: PointerHeadRuntime
    model_name: str = "kevane-0.6b"
    description: str = "Local KevANE Qwen3-0.6B on Apple Neural Engine"
    release_date: str = "2026-09-24"
    api_key: str | None = None
    lock: object = field(default_factory=threading.Lock)

    def answer(self, req: SystemOneRequest) -> dict:
        rec, meta = to_record(req)
        with self.lock:
            t0 = time.time()
            try:
                enc = self.encoder.encode(rec)
            except ContextOverflowANE as e:
                raise HTTPException(422, str(e))
            try:
                rows = self.encoder.inference_rows(enc)
            except ContextOverflowANE as e:
                raise HTTPException(422, str(e))
            ps = []
            for row in rows:
                inputs = self.encoder.coreml_inputs(row)
                h = self.backbone.hidden(inputs, len(row["ids"]))
                ps.extend(self.head.probs(h, row["decide_idx"], row["opt_idx"]))
            latency_ms = round((time.time() - t0) * 1000, 1)
        answers = to_answers(ps, meta)
        return {
            "model": req.model,
            "answers": answers,
            "usage": {"input_tokens": len(enc["ids"]), "output_tokens": output_tokens(self.encoder.tok, answers)},
            "latency_ms": latency_ms,
        }


# 本地服务不启用 CORS：客户端（Jarvis 等）是原生应用，无需跨域；放开 CORS
# 会让用户浏览器中任意网页直接 POST 并读取本接口（drive-by-localhost）。
MAX_BODY_BYTES = 4 * 1024 * 1024  # state/questions 本身远小于此；先挡异常大请求


def create_app(server: KevANEServer) -> FastAPI:
    app = FastAPI(title="kevane")

    @app.middleware("http")
    async def typesafe(request, call_next):
        if request.method == "POST" and int(request.headers.get("content-length") or 0) > MAX_BODY_BYTES:
            return JSONResponse({"detail": f"request body too large (limit {MAX_BODY_BYTES} bytes)"}, 413,
                                {"x-typesafe-request-id": request.headers.get("x-typesafe-request-id")
                                 or uuid.uuid4().hex})
        if server.api_key and request.url.path in {"/v1/systemone", "/systemone", "/v1/models", "/models"}:
            # header 按 latin-1 解码，可能含非 ASCII 字符；compare_digest 要求同类型，
            # 对 str 传非 ASCII 会抛 TypeError（变成 500），先编码成 bytes。
            given = request.headers.get("authorization", "").encode("latin-1", "ignore")
            expected = f"Bearer {server.api_key}".encode("latin-1", "ignore")
            if not hmac.compare_digest(given, expected):
                resp = JSONResponse({"detail": "missing or invalid API key; send Authorization: Bearer <KEVANE_API_KEY>"},
                                    401, {"www-authenticate": "Bearer"})
            else:
                resp = await call_next(request)
        else:
            resp = await call_next(request)
        resp.headers["x-typesafe-request-id"] = request.headers.get("x-typesafe-request-id") or uuid.uuid4().hex
        return resp

    @app.post("/v1/systemone")
    def systemone(req: SystemOneRequest):
        return server.answer(req)

    @app.post("/systemone")            # 别名：兼容 base URL 以 /v1 结尾的客户端拼接
    def systemone_alias(req: SystemOneRequest):
        return server.answer(req)

    @app.get("/v1/models")
    def models():
        return {"models": [{"name": server.model_name, "description": server.description,
                            "release_date": server.release_date}]}

    @app.get("/models")
    def models_alias():
        return models()

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "backend": "coreml", "compute_units": server.backbone.compute_units,
                "seq_len": server.backbone.seq_len,
                "pid": os.getpid(),
                "calls": server.backbone.calls,
                "worker_pid": server.backbone.worker_pid,
                "worker_restarts": server.backbone.restarts,
                "last_worker_error": server.backbone.last_worker_error}

    return app


@lru_cache(maxsize=4)
def default_server(mlpackage: str | None = None, seq_len: int = 512,
                   compute_units: str = "CPU_AND_NE", model_dir: str | None = None) -> KevANEServer:
    """Build the service from a self-contained Hugging Face model directory."""
    model_path = Path(model_dir or ROOT / "hf-model").expanduser().resolve()
    mlpackage = mlpackage or str(model_path / "kev-qwen3-0.6b-hidden.mlpackage")
    if not Path(mlpackage).exists():
        raise FileNotFoundError(f"Core ML package missing: {mlpackage}; download the model repository")
    meta_path = Path(mlpackage).parent / (Path(mlpackage).stem + "-meta.json")
    if meta_path.exists():
        seq_len = json.loads(meta_path.read_text()).get("seq_len", seq_len)
    return KevANEServer(
        encoder=Encoder(model_path, seq_len),
        backbone=IsolatedCoreMLBackbone(mlpackage, seq_len, compute_units),
        head=PointerHeadRuntime(model_path / "kev-qwen3-0.6b-pointer-head.pt", 1024),
        api_key=os.environ.get("KEVANE_API_KEY"),
    )
