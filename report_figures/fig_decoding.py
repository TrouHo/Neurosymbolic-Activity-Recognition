"""Figure: arg-max versus rule-constrained decoding on the real outputs of a trained TCN-B2 (all
devices, 100 % labels, seed 0) for one held-out participant (decoder: emob/decode.py).

Usage (from code/):  .venv\\Scripts\\python.exe -m report_figures.fig_decoding [--fold 6 --start 29.6 --end 32.8]
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from emob import config as C
from emob.decode import decode_logits
from emob.evaluate import decide
from report_figures.style import (ACCENT, ACT_EN, GRID, MAIN_MID, OUT, POSTURE_EN, RESULTS, TEXTWIDTH_IN, V2,
                                  drop_overflowing_labels, setup, track)


def _runs(a):
    out, i = [], 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[j + 1] == a[i]:
            j += 1
        out.append((int(a[i]), i, j + 1))
        i = j + 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=6)
    ap.add_argument("--start", type=float, default=29.6)
    ap.add_argument("--end", type=float, default=32.8)
    a = ap.parse_args(argv)
    setup()
    z = np.load(RESULTS / f"fold{a.fold}" / "test_logits.npz")
    lg = {h: z[f"ALL_{h}"].astype(np.float64) for h in ("l2", "l1", "s", "w", "t")}
    am, dec = decide(lg), decode_logits(lg)
    d = np.load(V2 / f"user{a.fold + 1}.npz")
    s, e = int(round(a.start * 10)), int(round(a.end * 10))
    act = [ACT_EN[c] for c in C.L2_CLASSES]
    rows = [("Annotation", d["l2"][s:e], act), ("Arg-max", am["l2"][s:e], act), ("Decoded", dec["l2"][s:e], act),
            ("Posture (arg-max)", am["s"][s:e], POSTURE_EN)]
    gap = 1.45                                           # vertical distance between rows
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.4))
    ys = []
    for i, (name, vals, names) in enumerate(rows):
        y = (len(rows) - 1 - i) * gap
        ys.append(y)
        ax.axhline(y, color=GRID, linewidth=0.4, zorder=0)
        track(ax, y, vals, names, 0.0, minlabel=0.3)
        # name the short transition segments with a call-out above the box (adjacent ones spread apart)
        short = [(v, a0, b0) for v, a0, b0 in _runs(vals)
                 if names is act and v >= 0 and C.L2_CLASSES[v] in C.POSTURE_TRANSITIONS and (b0 - a0) <= 3]
        for j, (v, a0, b0) in enumerate(short):
            x = (a0 + b0) / 20
            near = [k for k, (_, c0, c1) in enumerate(short) if abs((c0 + c1) / 20 - x) < 0.5]
            dx = 0.0 if len(near) == 1 else (-0.3 if near.index(j) == 0 else 0.3)
            ax.annotate(names[v], xy=(x, y + 0.31), xytext=(x + dx, y + 0.66), ha="center", va="center", fontsize=6,
                        arrowprops=dict(arrowstyle="-", color="black", linewidth=0.6))
    changed = np.flatnonzero(am["l2"][s:e] != dec["l2"][s:e])
    y_am = ys[1]
    for k in changed:                                    # steps changed by the decoder: strip under the arg-max row
        ax.add_patch(plt.Rectangle((k / 10, y_am - 0.46), 0.1, 0.1, color=ACCENT, linewidth=0, zorder=5))
    ax.text(a.end - a.start, y_am - 0.41, "steps changed by the decoder", ha="right", va="center", fontsize=6, color=ACCENT)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.5, ys[0] + 0.9)
    ax.set_xlim(0, a.end - a.start)
    ax.set_xlabel("Time from the start of the excerpt (s)")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    drop_overflowing_labels(fig, ax)
    out = OUT / "decoding_example.pdf"
    fig.savefig(out)
    print(f"written {out} (P{a.fold + 1}, {a.start}-{a.end} s); steps changed in window: {len(changed)}")


if __name__ == "__main__":
    main()
