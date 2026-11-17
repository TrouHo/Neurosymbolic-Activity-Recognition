"""TCN backbone (main) — see reports/design/2026-11-05_experiment_plan.md §2.3."""
from __future__ import annotations

import torch
from torch import nn

from emob import config as C


class DilatedResidual(nn.Module):
    """x -> Conv1d(k=3, dilation d) -> ReLU -> Conv1d 1x1 -> Dropout -> + x   (MS-TCN style)."""

    def __init__(self, ch: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.mix = nn.Conv1d(ch, ch, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.mix(torch.relu(self.conv(x))))


class Heads(nn.Module):
    def __init__(self, ch: int, use_s: bool):
        super().__init__()
        self.l1 = nn.Conv1d(ch, len(C.L1_CLASSES), 1)
        self.l2 = nn.Conv1d(ch, len(C.L2_CLASSES), 1)
        self.s = nn.Conv1d(ch, len(C.S_CLASSES), 1) if use_s else None

    def forward(self, h):
        out = {"l1": self.l1(h), "l2": self.l2(h)}
        if self.s is not None:
            out["s"] = self.s(h)
        return out


class TCN(nn.Module):
    """Input (B, L, 8) at 50 Hz -> logits (B, K, L/5) at 10 Hz for each head."""

    def __init__(self, in_ch: int = 8, ch: int = 64, layers: int = 8, dropout: float = 0.2,
                 use_s: bool = True):
        super().__init__()
        self.stem = nn.Conv1d(in_ch, ch, 5, padding=2)
        self.pool = nn.AvgPool1d(C.STEP, C.STEP)
        self.body = nn.Sequential(*[DilatedResidual(ch, 2 ** i, dropout) for i in range(layers)])
        self.heads = Heads(ch, use_s)

    @staticmethod
    def receptive_field_steps(layers: int) -> int:
        return 1 + 2 * (2 ** layers - 1)

    def forward(self, x):
        h = self.pool(torch.relu(self.stem(x.transpose(1, 2))))
        return self.heads(self.body(h))
