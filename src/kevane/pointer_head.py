"""PointerHead：hidden -> per-question 概率（留在 CPU，手册 6.1 刻意为之）。

权重来自 hf-model/kev-qwen3-0.6b-pointer-head.pt（head + temperature + meta），
与官方 kev 的 PointerHead 逐参数一致。
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from kev.model import PointerHead


class PointerHeadRuntime:
    def __init__(self, head_pt: str | Path, hidden_size: int, device: str = "cpu"):
        blob = torch.load(head_pt, map_location=device, weights_only=True)
        meta = blob["meta"]
        self.head = PointerHead(hidden_size, dp=meta.get("head_dim", 256)).to(device)
        self.head.load_state_dict(blob["head"])
        self.head.temperature = float(blob["temperature"])
        self.head.eval()
        self.device = device

    def probs(self, h: torch.Tensor, decide_idx: list, opt_idx: list) -> list[list[float]]:
        """h: [L, d]；返回每个 question 的概率列表（与 kev probs() 同序）。"""
        out = []
        with torch.no_grad():
            for d, oi in zip(decide_idx, opt_idx):
                z = self.head(h[d], h[torch.tensor(oi, device=self.device)])
                out.append(F.softmax(z, -1).tolist())
        return out
