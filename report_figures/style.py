"""Shared style of the data figures of the proposal (same palette as proposal/latex/figstyle.tex).

Figures are written as PDF into proposal/latex/figures/. Text is typeset by LaTeX when available
(usetex), so fonts match the document; otherwise a serif font is used.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from emob import config as C  # noqa: E402

OUT = C.PROJECT / "proposal" / "latex" / "figures"
V2 = C.PROCESSED
RESULTS = C.PROJECT / "data" / "results_local" / "TCN-B2_frac1_seed0"   # copied from ictserver11

MAIN, MAIN_FILL, MAIN_MID, MAIN_STRONG = "#1F5F5F", "#F0F8F8", "#D6EBEB", "#94C7C7"
ACCENT, ACCENT_FILL = "#CC6600", "#FFEBD9"
LINE, GRID, NEUTRAL = "#595959", "#D9D9D9", "#8C8C8C"
TEXTWIDTH_IN = 16.0 / 2.54

ACT_EN = {
    "Lever_Bras": "Raise one arm", "Baisser_Bras": "Lower one arm", "Lever_2_Bras": "Raise both arms",
    "Baisser_2_Bras": "Lower both arms", "Autre_Mouvement_Bras": "Other arm movement", "Marcher": "Walking",
    "Lever_Jambe": "Raise one leg", "Baisser_Jambe": "Lower one leg", "Autre_Mouvement_Jambe": "Other leg movement",
    "Debout_Assis": "Stand to sit", "Assis_Debout": "Sit to stand", "Assis_Couche": "Sit to lie",
    "Couche_Assis": "Lie to sit", "Debout_Penche": "Stand to bend", "Penche_Debout": "Bend to stand",
    "Debout_Agenou": "Stand to kneel", "Agenou_Debout": "Kneel to stand", "Debout_Accroupi": "Stand to squat",
    "Accroupi_Debout": "Squat to stand", "Autre_Transition": "Other posture transition"}
GROUP_EN = {"ARM": "Arm movements", "WALK": "Walking", "LIE_TRANSITION": "Lie transitions",
            "MODERATE": "Moderate movements", "SIT_TRANSITION": "Sit transitions", "LEG": "Leg movements",
            "OTHER_TRANSITION": "Other transition"}
POSTURE_EN = ["Standing", "Sitting", "Lying", "Bent", "Kneeling", "Squatting"]


def setup():
    usetex = shutil.which("latex") is not None
    plt.rcParams.update({
        "text.usetex": usetex, "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 8, "axes.edgecolor": LINE, "axes.linewidth": 0.6,
        "xtick.color": LINE, "ytick.color": "black", "xtick.major.width": 0.6, "ytick.major.width": 0,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    if usetex:
        plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}\usepackage[T1]{fontenc}"
    OUT.mkdir(parents=True, exist_ok=True)
    return usetex


def segments(a, empty=C.IGNORE):
    """Maximal runs (value, start, end_exclusive) of an integer array, empty values included."""
    out, i = [], 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[j + 1] == a[i]:
            j += 1
        out.append((int(a[i]), i, j + 1))
        i = j + 1
    return out


def track(ax, y, values, names, t0, step=0.1, height=0.62, fill=MAIN_MID, edge=MAIN, minlabel=1.3,
          unknown_hatch=True, hide=()):
    """Draw one row of labelled intervals: a box per run, its name inside when wide enough;
    IGNORE runs hatched (no target); values in `hide` left blank."""
    for v, s, e in segments(values):
        x0, w = t0 + s * step, (e - s) * step
        if v == C.IGNORE:
            if unknown_hatch:
                ax.add_patch(plt.Rectangle((x0, y - height / 2), w, height, facecolor="white", edgecolor=NEUTRAL,
                                           hatch="////", linewidth=0.3))
            continue
        if v in hide:
            continue
        ax.add_patch(plt.Rectangle((x0, y - height / 2), w, height, facecolor=fill, edgecolor=edge, linewidth=0.5))
        if w >= minlabel and names is not None and names[v]:
            txt = ax.text(x0 + w / 2, y, names[v], ha="center", va="center", fontsize=6)
            _labels.append((txt, x0, w))


_labels: list = []


def drop_overflowing_labels(fig, ax, margin=1.0):
    """Remove the interval names that do not fit inside their box (measured after layout)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for txt, x0, w in _labels:
        (px0, _), (px1, _) = ax.transData.transform([(x0, 0), (x0 + w, 0)])
        room = margin * (px1 - px0)
        if txt.get_window_extent(renderer=r).width <= room:
            continue
        words = txt.get_text().split(" ")
        if len(words) > 1:                       # try two lines, split near the middle
            k = min(range(1, len(words)), key=lambda i: abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))))
            txt.set_text(" ".join(words[:k]) + "\n" + " ".join(words[k:]))
            txt.set_linespacing(0.95)
            if txt.get_window_extent(renderer=r).width <= room:
                continue
        txt.remove()
    _labels.clear()
