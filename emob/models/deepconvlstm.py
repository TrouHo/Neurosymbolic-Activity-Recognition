"""DeepConvLSTM backbone (control). Reference: reports/design/2026-11-17_model_architecture.md §2.2.

Original (Ordóñez & Roggen 2016, Sensors 16(1):115, §3 and Fig. 3, checked in the PDF):
C(64)-C(64)-C(64)-C(64)-R(128)-R(128)-Softmax; convolutions along time only, kernel 5, no pooling,
each sensor channel keeps its own feature maps until the recurrent layers.

Our adaptation (each change stated):
- "same" padding, to keep the length aligned with the 0.1-s grid;
- AvgPool x5 after the convolutions (10 Hz, same grid as the TCN);
- bidirectional LSTM (decided 2026-11-05; the TCN also sees the future);
- 60-s training crops and the full recording at test, instead of short sliding windows;
- a linear projection 2x128 -> 64 so that the heads are identical to the TCN's (added);
- dropout before the LSTM is a setting of ours, not taken from the paper.

x (B, L, C) -> view (B, 1, L, C) -> 4 x [Conv2d kernel (5, 1), 64 maps, ReLU]   (per-channel)
 -> (B, 64*C, L) -> AvgPool x5 -> (B, T, 64*C) -> 2-layer BiLSTM(128) -> Linear(256 -> 64) -> heads
"""
from __future__ import annotations

import torch
from torch import nn

from emob import config as C
from emob.models.heads import ALL_HEADS, Heads


class DeepConvLSTM(nn.Module):
    """Input (B, L, in_ch) at 50 Hz -> {head: (B, units, L/5)} at 10 Hz."""

    def __init__(self, in_ch: int = 17, conv_ch: int = 64, kernel: int = 5, conv_layers: int = 4,
                 lstm_units: int = 128, lstm_layers: int = 2, bidirectional: bool = True,
                 out_ch: int = 64, dropout: float = 0.2, heads: list[str] | None = None):
        super().__init__()
        convs, c = [], 1
        for _ in range(conv_layers):
            convs += [nn.Conv2d(c, conv_ch, (kernel, 1), padding=(kernel // 2, 0)), nn.ReLU()]
            c = conv_ch
        self.convs = nn.Sequential(*convs)
        self.pool = nn.AvgPool1d(C.STEP, C.STEP)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(conv_ch * in_ch, lstm_units, num_layers=lstm_layers, batch_first=True,
                            bidirectional=bidirectional)
        self.proj = nn.Linear(lstm_units * (2 if bidirectional else 1), out_ch)
        self.heads = Heads(out_ch, heads or ALL_HEADS)

    def features(self, x):
        B, L, Cin = x.shape
        f = self.convs(x.unsqueeze(1))                      # (B, 64, L, C)
        f = f.permute(0, 1, 3, 2).reshape(B, -1, L)         # (B, 64*C, L)
        f = self.pool(f).transpose(1, 2)                    # (B, T, 64*C)
        f, _ = self.lstm(self.drop(f))                      # (B, T, 2*128)
        return self.proj(f).transpose(1, 2)                 # (B, 64, T), linear projection

    def forward(self, x):
        return self.heads(self.features(x))
