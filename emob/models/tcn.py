"""TCN backbone (main). Reference: reports/design/2026-11-17_model_architecture.md §2.1.

x (B, L, C) at 50 Hz
 -> Conv1d C->64, kernel 5, ReLU                      (stem)
 -> AvgPool x5                                        (10 Hz: one vector per 0.1 s)
 -> 8 dilated residual layers, d = 1, 2, ..., 128:
      x -> Conv1d(64, k=3, dilation d) -> ReLU -> Conv1d 1x1 -> Dropout -> + x   (MS-TCN style)
 -> h(t) (64) -> heads
Receptive field 1 + 2*(2^8 - 1) = 511 steps (about 51 s, +-25 s). Non-causal (offline task).
One stage and no smoothing loss on purpose, so that B1 is pure cross-entropy.
"""
from __future__ import annotations

import torch
from torch import nn

from emob import config as C
from emob.models.heads import ALL_HEADS, Heads


class DilatedResidual(nn.Module):
    def __init__(self, ch: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.mix = nn.Conv1d(ch, ch, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.mix(torch.relu(self.conv(x))))


class TCN(nn.Module):
    """Input (B, L, in_ch) at 50 Hz -> {head: (B, units, L/5)} at 10 Hz."""

    def __init__(self, in_ch: int = 17, ch: int = 64, layers: int = 8, dropout: float = 0.2,
                 heads: list[str] | None = None):
        super().__init__()
        self.stem = nn.Conv1d(in_ch, ch, 5, padding=2)
        self.pool = nn.AvgPool1d(C.STEP, C.STEP)
        self.body = nn.Sequential(*[DilatedResidual(ch, 2 ** i, dropout) for i in range(layers)])
        self.heads = Heads(ch, heads or ALL_HEADS)

    @staticmethod
    def receptive_field_steps(layers: int) -> int:
        return 1 + 2 * (2 ** layers - 1)

    def features(self, x):
        return self.body(self.pool(torch.relu(self.stem(x.transpose(1, 2)))))

    def forward(self, x):
        return self.heads(self.features(x))
