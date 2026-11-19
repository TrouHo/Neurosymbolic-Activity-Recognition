"""02 Labels: classes, segments, overlaps and unannotated time.

Activity = column `etiq activite` (20 classes; protocol markers Debut/Fin and empty cells are not
labels). raw = the CSV as delivered; v2 = after label time alignment and masking of the abnormal
long annotations. Seconds of a class are counted on rows (1/50 s) unless the unit says step.
Movement tracks = body, arm and leg columns; a track is active on a row when its value is not empty
and not `inactif`.
"""
from __future__ import annotations

import numpy as np

from emob import config as C
from inspection_data.common import PNAME, ROW_S, STEP_S, Context, Registry, lengths

TRACKS = {"body": C.BODY_COL, "arm": "etiq mvt_bras", "leg": C.LEG_COL}
ARM_RAISE_LOWER = ["Lever_Bras", "Baisser_Bras", "Lever_2_Bras", "Baisser_2_Bras"]


def is_label(v) -> bool:
    return v != "" and v not in C.PROTOCOL_MARKERS


def run(ctx: Context, reg: Registry):
    reg.section = "02 Labels"
    U = ctx.users
    # ---- per class seconds
    raw_s = {c: sum(float((ctx.raw[u]["act"] == c).sum()) * ROW_S for u in U) for c in C.L2_CLASSES}
    v2_s = {c: sum(ctx.stats(u)["l2_s"][c] for u in U) for c in C.L2_CLASSES}
    v2_n = {c: sum(ctx.stats(u)["l2_s"][c] > 0 for u in U) for c in C.L2_CLASSES}
    for c in C.L2_CLASSES:
        reg.add(f"lab.class.{c}.raw_s", raw_s[c], "s", "raw", f"rows labelled {c} in the activity column")
        reg.add(f"lab.class.{c}.v2_s", v2_s[c], "s", "v2", f"rows labelled {c} after alignment and masking")
        reg.add(f"lab.class.{c}.v2_participants", v2_n[c], "participants", "v2", f"participants with at least one row of {c}")
    tot_raw, tot_v2 = sum(raw_s.values()), sum(v2_s.values())
    reg.add("lab.activity.raw_s", tot_raw, "s", "raw", "rows with one of the 20 activity classes")
    reg.add("lab.activity.v2_s", tot_v2, "s", "v2", "rows with a usable activity label (main target)")
    reg.add("lab.activity.v2_min", tot_v2 / 60, "min", "v2", "same, in minutes")
    for u in U:
        reg.add(f"lab.activity.{PNAME[u]}.v2_s", ctx.stats(u)["l2_supervised_s"], "s", "v2", f"{PNAME[u]}: rows with a usable activity label")
    reg.add("lab.share.Autre_Mouvement_Bras.v2_pct", 100 * v2_s["Autre_Mouvement_Bras"] / tot_v2, "%", "v2", "Other arm movement / all usable activity rows")
    reg.add("lab.share.Marcher.v2_pct", 100 * v2_s["Marcher"] / tot_v2, "%", "v2", "Walking / all usable activity rows")
    small = [c for c in C.L2_CLASSES if v2_s[c] < 20]
    reg.add("lab.n_classes_below_20s.v2", len(small), "classes", "v2", "classes with less than 20 s: " + ", ".join(small))
    reg.add("lab.arm_raise_lower.v2_s", sum(v2_s[c] for c in ARM_RAISE_LOWER), "s", "v2", "total of the four arm raise / lower classes")
    reg.add("lab.n_classes_below_100s.v2", sum(v2_s[c] < 100 for c in C.L2_CLASSES), "classes", "v2", "classes with less than 100 s")

    # ---- activity segments in three versions (the unit changes the count, not the data)
    a = np.concatenate([lengths(ctx.raw[u]["act"], ROW_S, keep=is_label) for u in U])
    b = np.concatenate([lengths(ctx.v2[u]["l2_raw"], STEP_S) for u in U])
    c = np.concatenate([lengths(ctx.v2[u]["l2"], STEP_S) for u in U])
    for tag, x, data, unit in (("raw_rows", a, "raw", "row"), ("raw_steps", b, "raw", "step"), ("v2_steps", c, "v2", "step")):
        reg.add(f"lab.segments.{tag}.n", len(x), "segments", data, f"activity segments, counted on {unit}s")
        reg.add(f"lab.segments.{tag}.below_1s_pct", 100 * (x < 1.0 - 1e-9).mean(), "%", data, f"share of activity segments shorter than 1 s ({unit}s)")
        reg.add(f"lab.segments.{tag}.median_s", np.median(x), "s", data, f"median activity segment duration ({unit}s)")

    # ---- named transitions: median duration per class, body track, raw rows
    med = {}
    for t in C.POSTURE_TRANSITIONS:
        d = np.concatenate([lengths(ctx.raw[u]["fam"][C.BODY_COL], ROW_S, keep=lambda v, t=t: v == t) for u in U])
        med[t] = float(np.median(d))
        reg.add(f"lab.transition.{t}.median_s", med[t], "s", "raw", f"median duration of {t} segments in the body track")
    reg.add("lab.transition.median_min_s", min(med.values()), "s", "raw", "smallest per-class median duration of the named transitions")
    reg.add("lab.transition.median_max_s", max(med.values()), "s", "raw", "largest per-class median duration of the named transitions")

    # ---- overlap of the movement tracks and unannotated time (raw rows, raw labelled region)
    ov = ann = gap = region_n = 0
    for u in U:
        fam = ctx.raw[u]["fam"]
        active = np.stack([(fam[col] != "") & (fam[col] != "inactif") for col in TRACKS.values()])
        any_track = np.any(np.stack([fam[col] != "" for col in C.FAMILY_COLS]), axis=0)
        idx = np.flatnonzero(any_track)
        region = slice(idx[0], idx[-1] + 1)
        k = active[:, region].sum(0)
        ann += int((k >= 1).sum())
        ov += int((k >= 2).sum())
        gap += int((~any_track[region]).sum())
        region_n += region.stop - region.start
    reg.add("lab.tracks.overlap_pct.raw", 100 * ov / ann, "% rows", "raw",
            "rows with two or more active movement tracks / rows with at least one (labelled region)")
    reg.add("lab.unannotated.raw_pct", 100 * gap / region_n, "% rows", "raw",
            "rows with no label on any track / rows of the labelled region")
    reg.add("lab.unannotated.raw_s", gap * ROW_S, "s", "raw", "rows with no label on any track inside the labelled region")
    reg.add("lab.labelled_region.raw_s", region_n * ROW_S, "s", "raw", "total length of the labelled regions (first to last labelled row)")
    # same on v2 (after alignment and masking), step grid
    g2 = r2 = 0
    for u in U:
        st = ctx.stats(u)
        g2 += st["labelled_region_s"] - st["annotated_s"]
        r2 += st["labelled_region_s"]
    reg.add("lab.unannotated.v2_pct", 100 * g2 / r2, "% rows", "v2", "rows with no label on any track / labelled region (after alignment and masking)")
