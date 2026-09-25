"""编码器：System One record -> kev packed encoding -> Core ML 静态输入。

与官方/merged PyTorch 路径使用完全相同的 kev.model.encode（token、位置、
block-causal branch mask 语义一致），差异只在后端执行 Core ML 图：
    input_ids   [1, SEQ] int32   （右 padding 0，mask 屏蔽）
    position_ids[SEQ]    int32
    causal_mask [1,1,SEQ,SEQ] fp16（additive，finfo(fp16).min）
"""
from pathlib import Path

import numpy as np
import torch
from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE, ContextOverflow, branch_mask_batch, encode, rows_of
from transformers import AutoTokenizer

NEG16 = float(np.finfo(np.float16).min)


class ContextOverflowANE(ContextOverflow):
    """packed 序列超出 Core ML 静态上下文（SEQ）。"""


class Encoder:
    def __init__(self, tokenizer_dir: str | Path, seq_len: int):
        self.tok = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.seq_len = seq_len

    def encode(self, rec: dict) -> dict:
        """Encode without silently dropping old state tokens."""
        try:
            enc = encode(self.tok, rec, max_state=SERVE_MAX_STATE,
                         max_branch=SERVE_MAX_BRANCH, strict=True)
        except ContextOverflow as exc:
            raise ContextOverflowANE(str(exc)) from exc
        return enc

    def inference_rows(self, enc: dict) -> list[dict]:
        """Split independent questions only when the packed graph will not fit.

        Each question can see the shared state and its own branch, never another
        question. Running those causal rows separately preserves that attention
        pattern while keeping each Core ML call within the compiled shape.
        """
        if len(enc["ids"]) <= self.seq_len:
            return [enc]
        state_ids, state_pos, branches = rows_of(enc)
        if len(branches) > 16:
            raise ContextOverflowANE("request needs more than 16 Core ML passes; split the questions")
        rows = []
        for index, branch in enumerate(branches, start=1):
            length = len(state_ids) + len(branch["ids"])
            if length > self.seq_len:
                raise ContextOverflowANE(
                    f"question {index} needs {length} tokens including shared state; "
                    f"this Core ML model supports {self.seq_len} tokens per question")
            offset = len(state_ids)
            rows.append({
                "ids": state_ids + branch["ids"],
                "pos": state_pos + branch["pos"],
                "seg": [0] * offset + [1] * len(branch["ids"]),
                "decide_idx": [offset + branch["decide"]],
                "opt_idx": [[offset + opt for opt in branch["opts"]]],
            })
        return rows

    def coreml_inputs(self, enc: dict) -> dict:
        L = len(enc["ids"])
        ids = np.zeros((1, self.seq_len), dtype=np.int32); ids[0, :L] = enc["ids"]
        pos = np.zeros((self.seq_len,), dtype=np.int32); pos[:L] = enc["pos"]
        mask = branch_mask_batch([enc["seg"]], "cpu", dtype=torch.float16, length=self.seq_len).numpy()
        return {"input_ids": ids, "position_ids": pos, "causal_mask": mask}
