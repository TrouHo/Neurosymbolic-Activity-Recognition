"""Output heads shared by every backbone (reports/design/2026-11-17_model_architecture.md §3).

Each head is a 1x1 convolution (a linear layer applied at every 0.1-s step) on the shared
representation h(t) of 64 channels. L2 (20), L1 (7), S (6): logits for a softmax; W, T: one logit
for a sigmoid. Model B1 uses L2 + L1; B2 and NS* use all five.
"""
from __future__ import annotations

import torch
from torch import nn

from emob import config as C

B1_HEADS = ["l2", "l1"]
ALL_HEADS = list(C.HEADS)


class Heads(nn.Module):
    def __init__(self, ch: int, heads: list[str]):
        super().__init__()
        unknown = set(heads) - set(C.HEADS)
        if unknown or "l2" not in heads:
            raise ValueError(f"heads must include l2 and be among {list(C.HEADS)}: {heads}")
        self.heads = nn.ModuleDict({h: nn.Conv1d(ch, C.HEADS[h], 1) for h in heads})

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """h: (B, ch, T) -> {head: (B, units, T)}."""
        return {k: m(h) for k, m in self.heads.items()}
