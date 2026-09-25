"""L4 API parity：POST /v1/systemone 响应 vs 官方 kev golden 输出（阶段 4 Stop/Go）。

比较（fp16 噪声容差，与 22 号脚本同一套规则）：
    - choice/noul/score 的 argmax 语义字段（choice、noul、score 期望值容差 0.15）
    - probabilities max|Δp| < 0.12（fp16 包络 0.08 + 裕量）
    - argmax 翻转仅允许 margin < 0.10 的边界样本（手册 10.2，已知 1 例）
    - 结构字段：model/answers/usage/latency_ms 完整
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


from fastapi.testclient import TestClient  # noqa: E402

from kevane.server import create_app, default_server  # noqa: E402

BOUNDARY_MARGIN = 0.10
DP_TOL = 0.12
SCORE_TOL = 0.15


_SERVER = default_server()          # 进程内单例：Core ML 模型只加载一次
_CLIENT = TestClient(create_app(_SERVER))


def load_client():
    return _CLIENT, _SERVER


def golden_rows():
    rows = {}
    for line in (ROOT / "tests" / "fixtures" / "official_fp32_probs.jsonl").read_text(encoding="utf-8").splitlines()[1:]:
        row = json.loads(line)
        rows[row["id"]] = {o["qid"]: o for o in row["outputs"]}
    return rows


def test_systemone_parity():
    cases = [json.loads(l) for l in (ROOT / "tests" / "fixtures" / "cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    client, server = load_client()
    golden = golden_rows()
    n, flips_boundary = 0, []

    for case in cases:
        body = {"model": "kevane-0.6b", "state": case["state"], "questions": case["questions"]}
        r = client.post("/v1/systemone", json=body)
        assert r.status_code == 200, (case["id"], r.status_code, r.text[:200])
        resp = r.json()
        assert resp["model"] == "kevane-0.6b"
        assert "latency_ms" in resp and resp["usage"]["input_tokens"] > 0
        assert r.headers.get("x-typesafe-request-id")

        for qid, q in case["questions"].items():
            g = golden[case["id"]][qid]
            a = resp["answers"][qid]
            n += 1
            if q["type"] == "noul":
                # golden argmax 语义：[false, true]，answer = p(true)
                gp = g["probabilities"]
                assert abs(a["noul"] - gp[1]) < DP_TOL, (case["id"], qid, a["noul"], gp[1])
            elif q["type"] == "choice":
                keys = g["keys"]
                coreml_argmax = max(range(len(keys)), key=lambda i: a["probabilities"][keys[i]])
                if coreml_argmax != g["argmax"]:
                    s = sorted(g["probabilities"], reverse=True)
                    margin = s[0] - s[1]
                    assert margin < BOUNDARY_MARGIN, f"non-boundary argmax flip: {case['id']}.{qid}"
                    flips_boundary.append((case["id"], qid))
                for i, k in enumerate(keys):
                    assert abs(a["probabilities"][k] - g["probabilities"][i]) < DP_TOL, (case["id"], qid, k)
            else:  # score
                exp = sum(i * a["probabilities"][str(i)] for i in range(len(g["probabilities"])))
                gexp = sum(i * p for i, p in enumerate(g["probabilities"]))
                assert abs(exp - gexp) < SCORE_TOL, (case["id"], qid, exp, gexp)

    print(f"[api-parity] {n} questions across {len(cases)} cases; boundary flips: {flips_boundary}")


def test_models_endpoint():
    client, _ = load_client()
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["models"][0]["name"] == "kevane-0.6b"
    assert "description" in body["models"][0] and "release_date" in body["models"][0]


def test_context_overflow_422():
    client, _ = load_client()
    r = client.post("/v1/systemone", json={
        "model": "kevane-0.6b",
        "state": "x " * 2000,  # 超过 Core ML seq=512
        "questions": {"q": {"type": "choice", "instructions": "t", "criteria": {"a": None, "b": None}}},
    })
    assert r.status_code == 422, r.status_code


if __name__ == "__main__":
    test_systemone_parity()
    test_models_endpoint()
    test_context_overflow_422()
    print("[api-parity] ALL PASS")
