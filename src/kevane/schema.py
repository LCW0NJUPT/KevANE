"""TypeSafe System One 请求/响应契约。

手册 §7.4：最稳妥的实现是直接复用 Kev 的 System One response 语义，
不为 Jarvis 单独写返回格式。因此这里显式从 kev.api 再导出同一套
pydantic 模型与转换函数，KevANE 服务与官方 kev.serve 的 JSON 完全同构。
"""
from kev.api import (  # noqa: F401
    Noul,
    Choice,
    Score,
    Question,
    SystemOneRequest,
    to_record,
    to_answers,
    output_tokens,
    question_keys,
    render,
)

__all__ = [
    "Noul", "Choice", "Score", "Question", "SystemOneRequest",
    "to_record", "to_answers", "output_tokens", "question_keys", "render",
]
