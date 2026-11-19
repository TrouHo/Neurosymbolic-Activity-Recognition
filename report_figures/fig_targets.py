"""Figure: the five prediction targets on an excerpt of a real recording (prepared data v2).

Usage (from code/):  .venv\\Scripts\\python.exe -m report_figures.fig_targets [--user user7 --start 308 --end 352]
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from emob import config as C
from report_figures.style import (ACT_EN, GROUP_EN, GRID, MAIN, MAIN_MID, MAIN_STRONG, OUT, POSTURE_EN, TEXTWIDTH_IN,
                                  V2, drop_overflowing_labels, setup, track)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="user7")
    ap.add_argument("--start", type=float, default=317.5)
    ap.add_argument("--end", type=float, default=342.5)
    a = ap.parse_args(argv)
    setup()
    d = np.load(V2 / f"{a.user}.npz")
    s, e = int(round(a.start * 10)), int(round(a.end * 10))
    rows = [
        ("Activity", d["l2"][s:e], [ACT_EN[c] for c in C.L2_CLASSES], MAIN_MID, (), 1.6),
        ("Group", d["l1"][s:e], [GROUP_EN[c] for c in C.L1_CLASSES], MAIN_MID, (), 1.6),
        ("Posture", d["s"][s:e], POSTURE_EN, MAIN_MID, (), 1.6),
        ("Walking", d["w"][s:e], None, MAIN_STRONG, (0,), 99),
        ("Transition", d["t"][s:e], None, MAIN_STRONG, (0,), 99),
    ]
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.2))
    for i, (name, v, names, fill, hide, minlab) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.axhline(y, color=GRID, linewidth=0.4, zorder=0)
        track(ax, y, v, names, 0.0, fill=fill, edge=MAIN, hide=hide, minlabel=minlab)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, a.end - a.start)
    ax.set_xlabel("Time from the start of the excerpt (s)")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    drop_overflowing_labels(fig, ax)
    out = OUT / "targets_excerpt.pdf"
    fig.savefig(out)
    print(f"written {out} ({a.user}, {a.start}-{a.end} s)")


if __name__ == "__main__":
    main()
