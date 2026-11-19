"""Figure: where the temporal rule losses act, on the real predicted probabilities of a trained
TCN-B2 (all devices, 100 % labels, seed 0) for one held-out participant.

Usage (from code/):  .venv\\Scripts\\python.exe -m report_figures.fig_rules [--fold 6 --start 27.5 --end 36]
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from emob import config as C
from emob.rules_spec import TRANSITION_IDX
from report_figures.rulecalc import probabilities, violations
from report_figures.style import (ACCENT, ACCENT_FILL, ACT_EN, GRID, MAIN, MAIN_MID, OUT, RESULTS, TEXTWIDTH_IN, V2,
                                  drop_overflowing_labels, setup, track)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=6)          # fold k tests participant k+1
    ap.add_argument("--start", type=float, default=27.5)
    ap.add_argument("--end", type=float, default=36.0)
    a = ap.parse_args(argv)
    setup()
    z = np.load(RESULTS / f"fold{a.fold}" / "test_logits.npz")
    p = probabilities({h: z[f"ALL_{h}"].astype(float) for h in ("l2", "l1", "s", "w", "t")})
    v = violations(p)
    d = np.load(V2 / f"user{a.fold + 1}.npz")
    s, e = int(round(a.start * 10)), int(round(a.end * 10))
    t = (np.arange(s, e) - s) / 10

    rows = [("A2: start of a\nnamed transition", p["l2"][TRANSITION_IDX].sum(0), "activity head: any named transition", v["A2"]),
            ("A3: posture change\nwithout transition", p["t"], "transition head", v["A3"]),
            ("A6: walking bout\nshorter than 1 s", p["w"], "walking head", v["A6"])]
    fig, axes = plt.subplots(len(rows) + 1, 1, figsize=(TEXTWIDTH_IN, 3.6), sharex=True,
                             gridspec_kw={"height_ratios": [0.55] + [1] * len(rows), "hspace": 0.25})
    ax0 = axes[0]
    track(ax0, 0, d["l2"][s:e], [ACT_EN[c] for c in C.L2_CLASSES], 0.0, minlabel=0.6)
    ax0.set_ylim(-0.5, 0.5); ax0.set_yticks([0]); ax0.set_yticklabels(["Annotation"])
    for sp in ("top", "right", "left", "bottom"):
        ax0.spines[sp].set_visible(False)
    ax0.tick_params(bottom=False)
    for i, (ax, (name, prob, pname, viol)) in enumerate(zip(axes[1:], rows)):
        ax.fill_between(t, 0, viol[s:e], color=ACCENT_FILL, edgecolor=ACCENT, linewidth=0.6, step="mid",
                        label="violation $v_t$ of the rule")
        ax.plot(t, prob[s:e], color=MAIN, linewidth=0.9, label="predicted probability")
        ax.text(0.01, 0.98, pname, transform=ax.transAxes, ha="left", va="top", fontsize=6, color=MAIN)
        ax.set_ylim(0, 1.02); ax.set_yticks([0, 1])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=7)
        ax.grid(axis="y", color=GRID, linewidth=0.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if i == 0:
            ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=6, frameon=False, ncol=2, borderaxespad=0.0)
    axes[-1].set_xlim(0, a.end - a.start)
    axes[-1].set_xlabel("Time from the start of the excerpt (s)")
    drop_overflowing_labels(fig, ax0)
    out = OUT / "rule_violations_excerpt.pdf"
    fig.savefig(out)
    print(f"written {out} (P{a.fold + 1}, {a.start}-{a.end} s); max v: "
          + ", ".join(f"{r} {v[r][s:e].max():.2f}" for r in ("A2", "A3", "A6", "A7")))


if __name__ == "__main__":
    main()
