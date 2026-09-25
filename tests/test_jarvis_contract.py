"""Jarvis 契约 smoke test（手册 §7.5）：KevANE 必须原样满足 jev-chat-jarvis-mac
的 TypeSafe Jev provider 调用方式，不修改 Jarvis 源码。

覆盖：
    1. GET /v1/models                    设置页"获取模型列表"
    2. 测试连接                          单选 Choice"问候" -> choice=问候
    3. Intent + Risk 双问题（手册 7.2 原始请求体）
    4. 候选排序                          候选文本直接作为 criteria key，best.probabilities 逐项可读
    5. Bearer 鉴权                       KEVANE_API_KEY 设置时 401/200 行为
"""
import sys
import os
import signal
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


from fastapi.testclient import TestClient  # noqa: E402

from kevane.server import KevANEServer, create_app, default_server  # noqa: E402


_SERVER = default_server()          # 进程内单例：Core ML 模型只加载一次
_CLIENT = TestClient(create_app(_SERVER))


def client():
    return _CLIENT


def test_1_models_list():
    r = client().get("/v1/models")
    assert r.status_code == 200
    m = r.json()["models"][0]
    assert m["name"] and m["description"] and m["release_date"]


def test_2_test_connection_greeting():
    r = client().post("/v1/systemone", json={
        "model": "kevane-0.6b",
        "state": "早上好！",
        "questions": {"intent": {"type": "choice", "instructions": "这句话的意图是什么？",
                                 "criteria": {"问候": None, "催进度": None, "求助": None}}},
    })
    assert r.status_code == 200
    a = r.json()["answers"]["intent"]
    assert a["type"] == "choice" and a["choice"] == "问候", a
    assert 0 <= a["confidence"] <= 1 and set(a["probabilities"]) == {"问候", "催进度", "求助"}


def test_3_intent_and_risk():
    """手册 7.2：Jarvis 判断路径真实发送的请求体。"""
    r = client().post("/v1/systemone", json={
        "model": "kevane-0.6b",
        "state": "微信消息上下文：模型是qwen3-0.6b，打算先在mac上把lora合了再转coreml。"
                 "昨天你说mps上fp32已经对齐了，今天又说要重跑，因为numpy版本动了。这个需求你今天跟一下。",
        "questions": {
            "intent": {"type": "choice", "instructions": "这句话的真实意图是什么？",
                       "criteria": {"催进度": "希望对方今天推进并给出结果", "问进度": "只是了解当前状态", "闲聊": None}},
            "risk": {"type": "score", "instructions": "如果直接回复『好的马上』，风险有多大？",
                     "criteria": ["1级：稳", "2级：基本稳", "3级：小风险", "4级：有对赌成分", "5级：一半一半", "6级：多半不行"]},
        },
    })
    assert r.status_code == 200
    answers = r.json()["answers"]
    intent = answers["intent"]
    assert intent["type"] == "choice" and intent["choice"] in {"催进度", "问进度", "闲聊"}
    assert intent["choice"] == "催进度", intent  # golden 语义（官方 p=0.906）
    risk = answers["risk"]
    assert risk["type"] == "score" and 1 <= risk["score"] <= 6
    assert risk["legend"] and all(f"{i}" in risk["probabilities"] for i in range(1, 6))


def test_4_candidate_ranking():
    """Jarvis 候选排序：候选文本本身作为 criteria key。"""
    r = client().post("/v1/systemone", json={
        "model": "kevane-0.6b",
        "state": "老板：那个0.6b的模型今天能跑通吗？背景：Core ML 转换预计还需两天。",
        "questions": {"best": {"type": "choice", "instructions": "哪条回复最合适？", "criteria": {
            "今天先给您PyTorch基线结果，Core ML还得两天，风险我会盯着": None,
            "跑不通": None,
            "没问题，今天一定搞定": None,
            "这不是我的活": None,
        }}},
    })
    assert r.status_code == 200
    best = r.json()["answers"]["best"]
    assert best["type"] == "choice"
    assert len(best["probabilities"]) == 4  # 逐项可读
    assert best["choice"] in best["probabilities"]


def test_5_bearer_auth():
    server = _SERVER
    server.api_key = "local-key"      # 复用单例，测完恢复
    try:
        c = _CLIENT
        req = {"model": "kevane-0.6b", "state": "在吗",
               "questions": {"q": {"type": "noul", "instructions": "需要回复吗", "criteria": {"yes": None, "no": None}}}}
        r = c.post("/v1/systemone", json=req)
        assert r.status_code == 401
        r = c.post("/v1/systemone", json=req, headers={"Authorization": "Bearer local-key"})
        assert r.status_code == 200
        r = c.get("/v1/models")
        assert r.status_code == 401  # 鉴权同样覆盖 /v1/models
        assert c.get("/models").status_code == 401
        assert c.post("/systemone", json=req).status_code == 401
        assert c.post("/systemone", json=req,
                      headers={"Authorization": "Bearer local-key"}).status_code == 200
    finally:
        server.api_key = None


def test_6_worker_crash_recovers_without_losing_api():
    """A native worker death must not take down the Jarvis HTTP endpoint."""
    old_pid = _SERVER.backbone.worker_pid
    assert old_pid
    os.kill(old_pid, signal.SIGKILL)
    for _ in range(50):
        if _SERVER.backbone.worker_pid is None:
            break
        time.sleep(0.02)
    req = {"model": "kevane-0.6b", "state": "你好", "questions": {
        "test": {"type": "choice", "instructions": "请选择问候", "criteria": {"问候": None}}}}
    r = client().post("/v1/systemone", json=req)
    assert r.status_code == 200, r.text
    assert r.json()["answers"]["test"]["choice"] == "问候"
    assert _SERVER.backbone.worker_pid != old_pid
    health = client().get("/healthz").json()
    assert health["worker_restarts"] >= 1 and health["worker_pid"]


if __name__ == "__main__":
    test_1_models_list()
    test_2_test_connection_greeting()
    test_3_intent_and_risk()
    test_4_candidate_ranking()
    test_5_bearer_auth()
    test_6_worker_crash_recovers_without_losing_api()
    print("[jarvis-contract] ALL PASS")
