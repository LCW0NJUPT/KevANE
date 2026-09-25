"""KevANE: Kev System One 决策模型在 Apple Neural Engine 上的本地运行时。

架构（实施手册 §7.1）：
    schema.py        TypeSafe 请求/响应兼容（直接复用 kev.api 语义）
    encoder.py       state/questions -> ids/pos/branch mask + Core ML 输入张量
    runtime.py       Core ML / ANE backbone（fp16）
    pointer_head.py  hidden -> logits/temperature/softmax
    server.py        FastAPI: POST /v1/systemone, GET /v1/models
"""
__version__ = "0.3.0"
