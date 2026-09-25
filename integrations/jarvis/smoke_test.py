#!/usr/bin/env python
"""Jarvis 接入 smoke test（真实 HTTP，阶段 5）。

对着运行中的 KevANE 服务执行手册 §7.5 的契约检查：
模型列表 / 测试连接（问候）/ intent+risk 双问题 / 候选 Choice 排序 /
base URL 两种写法（host-only 与带 /v1 前缀——后者是客户端拼接行为，此处验证
服务端路径规范）。

用法：
    python integrations/jarvis/smoke_test.py --base-url http://127.0.0.1:8008 --api-key local
"""
import argparse
import json
import sys
import urllib.error
import urllib.request


def call(base, path, api_key, body=None):
    url = base.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if body else "GET",
                                 headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read()), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), dict(e.headers)


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + str(detail)[:120]) if not cond else ''}")
    return bool(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8008")
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()
    ok = True

    print("[smoke] 1. GET /v1/models（Jarvis 设置页）")
    code, body, headers = call(args.base_url, "/v1/models", args.api_key)
    ok &= check("200 + models[].name/description/release_date",
                code == 200 and body.get("models") and all(
                    k in body["models"][0] for k in ("name", "description", "release_date")), body)
    ok &= check("x-typesafe-request-id 响应头", "x-typesafe-request-id" in {k.lower() for k in headers})

    print("[smoke] 2. 测试连接：单选 Choice『问候』")
    code, body, _ = call(args.base_url, "/v1/systemone", args.api_key, {
        "model": "kevane-0.6b", "state": "早上好！",
        "questions": {"intent": {"type": "choice", "instructions": "这句话的意图是什么？",
                                 "criteria": {"问候": None, "催进度": None, "求助": None}}}})
    ok &= check("choice == 问候", code == 200 and body["answers"]["intent"]["choice"] == "问候",
                body.get("answers", {}).get("intent"))

    print("[smoke] 3. intent + risk 双问题（判断路径）")
    code, body, _ = call(args.base_url, "/v1/systemone", args.api_key, {
        "model": "kevane-0.6b",
        "state": "微信消息上下文：模型是qwen3-0.6b，打算先在mac上把lora合了再转coreml。"
                 "昨天你说mps上fp32已经对齐了，今天又说要重跑，因为numpy版本动了。这个需求你今天跟一下。",
        "questions": {
            "intent": {"type": "choice", "instructions": "这句话的真实意图是什么？",
                       "criteria": {"催进度": "希望对方今天推进并给出结果", "问进度": "只是了解当前状态", "闲聊": None}},
            "risk": {"type": "score", "instructions": "如果直接回复『好的马上』，风险有多大？",
                     "criteria": ["1级：稳", "2级：基本稳", "3级：小风险", "4级：有对赌成分", "5级：一半一半", "6级：多半不行"]}}})
    a = body.get("answers", {})
    intent, risk = a.get("intent", {}), a.get("risk", {})
    ok &= check("intent.choice/confidence/probabilities 可读",
                code == 200 and intent.get("choice") and intent.get("confidence") is not None
                and len(intent.get("probabilities", {})) == 3, intent)
    ok &= check("risk.score/legend/probabilities 可读",
                risk.get("score") is not None and risk.get("legend")
                and len(risk.get("probabilities", {})) == 6, risk)

    print("[smoke] 4. 候选排序：候选文本作为 criteria key")
    candidates = ["今天先给您PyTorch基线结果，Core ML还得两天，风险我会盯着", "跑不通", "没问题，今天一定搞定", "这不是我的活"]
    code, body, _ = call(args.base_url, "/v1/systemone", args.api_key, {
        "model": "kevane-0.6b",
        "state": "老板：那个0.6b的模型今天能跑通吗？背景：Core ML 转换预计还需两天。",
        "questions": {"best": {"type": "choice", "instructions": "哪条回复最合适？",
                               "criteria": {c: None for c in candidates}}}})
    best = body.get("answers", {}).get("best", {})
    ok &= check("best.probabilities 逐项可读（4 项）",
                code == 200 and set(best.get("probabilities", {})) == set(candidates), best)

    print("[smoke] 5. base URL 两种写法")
    # host-only -> 拼 /v1/systemone；base 以 /v1 结尾 -> 拼 /systemone（服务端两种路径都路由）
    for base, path in ((args.base_url, "/v1/systemone"),
                       (args.base_url.rstrip("/") + "/v1", "/systemone")):
        code, _, _ = call(base, path, args.api_key, {
            "model": "kevane-0.6b", "state": "在吗",
            "questions": {"q": {"type": "noul", "instructions": "需要回复吗",
                                "criteria": {"yes": None, "no": None}}}})
        ok &= check(f"{base} + {path} -> 200", code == 200, code)

    print(f"\n[smoke] {'ALL PASS' if ok else 'FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
