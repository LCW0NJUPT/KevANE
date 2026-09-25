"""Check that splitting questions keeps Kev's packed attention semantics."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from kev.model import branch_mask_batch  # noqa: E402
from kevane.encoder import ContextOverflowANE, Encoder  # noqa: E402


def test_long_packed_request_splits_without_changing_question_attention():
    encoder = Encoder(ROOT / "hf-model", 512)
    record = {
        "state": "The user asks for a reply to a recent message.",
        "questions": [
            {"instr": "Assess the message tone and intent. " * 8,
             "options": ["friendly response", "brief response", "ask for details"], "label": 0}
            for _ in range(8)
        ],
    }
    packed = encoder.encode(record)
    assert len(packed["ids"]) > 512
    rows = encoder.inference_rows(packed)
    assert len(rows) == len(record["questions"])
    packed_mask = branch_mask_batch([packed["seg"]], "cpu", torch.float16)[0, 0]
    state_len = packed["seg"].count(0)
    for q, row in enumerate(rows, start=1):
        assert len(row["ids"]) <= 512
        indices = list(range(state_len)) + [i for i, segment in enumerate(packed["seg"]) if segment == q]
        assert row["ids"] == [packed["ids"][i] for i in indices]
        assert row["pos"] == [packed["pos"][i] for i in indices]
        assert row["decide_idx"] == [indices.index(packed["decide_idx"][q - 1])]
        assert row["opt_idx"][0] == [indices.index(i) for i in packed["opt_idx"][q - 1]]
        row_mask = branch_mask_batch([row["seg"]], "cpu", torch.float16)[0, 0]
        assert torch.equal(row_mask, packed_mask[indices][:, indices])


def test_single_question_still_rejects_overlong_context():
    encoder = Encoder(ROOT / "hf-model", 512)
    packed = encoder.encode({"state": "context " * 800,
                             "questions": [{"instr": "Choose", "options": ["a", "b"], "label": 0}]})
    with pytest.raises(ContextOverflowANE, match="question 1 needs"):
        encoder.inference_rows(packed)
