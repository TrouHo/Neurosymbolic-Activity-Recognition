"""07 Motion capture: checks of the labels and of the alignment against the Qualisys markers.

  gaps        unannotated rows inside the labelled region (aligned labels): share of the time with
              the fastest marker at or above 100 mm/s ("moving"), for gaps < 1 s, >= 1 s and all.
  bend        trunk angle around the Stand to bend / Bend to stand annotations (median over rows
              5-15 before the start and 5-15 after the end): share with a change > 15 deg in the
              expected direction, raw and aligned labels; transitions inside an annotation longer
              than 12 s are excluded.
  bent        rows whose derived posture is bent (aligned labels, not inside a transition): share
              with the trunk inclined > 30 deg.
  unlabelled  episodes >= 0.5 s with derived posture standing and trunk > 30 deg, hip not low
              (>= 0.75 x standing hip height) and more than 0.5 s from any posture transition
              annotation: bends that carry no body label.
  residual    lag between each device kept in v2 and its marker, after alignment (estimator of the
              sensor alignment; emob.data.check.residual_lags).
  tracks      agreement of the annotation-offset estimates of the body, arm and leg tracks
              (reports/data/2026-11-13_label_timing.json, the input of params/label_offsets.json).
  manual      results of a manual reading of motion-capture timelines; not computed here.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from emob import config as C
from emob.data.check import residual_lags
from emob.data.labels import annotated_mask, posture_states, runs
from inspection_data import mocap as M
from inspection_data.common import PNAME, ROW_S, Context, Registry

FS = C.FS
SUSPECT_S = 12.0
NAMED = set(C.POSTURE_TRANSITIONS)
AT = C.OTHER_TRANSITION
TIMING = C.PROJECT / "reports" / "data" / "2026-11-13_label_timing.json"


def _prep(ctx: Context, u: str, aligned: bool) -> dict:
    fam = M.families(ctx.raw[u], ctx.lab_off, u, aligned)
    ann = annotated_mask(fam)
    idx = np.flatnonzero(ann)
    reg = np.zeros(len(ann), bool)
    reg[idx[0]:idx[-1] + 1] = True
    body = fam[C.BODY_COL]
    ps = posture_states(body, ann)
    sus = np.zeros(len(body), bool)
    for name, s, e in runs(body):
        if name in NAMED | {AT} and (e - s) / FS > SUSPECT_S:
            sus[s:e] = True
    return dict(fam=fam, ann=ann, reg=reg, body=body, ps=ps, st=ps.state_all, sus=sus)


def run(ctx: Context, reg: Registry):
    reg.section = "07 Motion capture"
    U = ctx.users
    mk = {u: M.markers(str(ctx.files[u])) for u in U}
    ang = {u: M.trunk_angle(mk[u]) for u in U}
    A = {u: _prep(ctx, u, True) for u in U}
    R = {u: _prep(ctx, u, False) for u in U}

    # ---- gaps
    tot = {"lt1s": [0, 0], "ge1s": [0, 0]}
    for u in U:
        v = M.fastest_speed(mk[u])
        p = A[u]
        for _, s, e in runs(np.where(p["reg"] & ~p["ann"], "G", "").astype(object)):
            key = "ge1s" if (e - s) / FS >= 1.0 else "lt1s"
            x = v[s:e]
            x = x[np.isfinite(x)]
            tot[key][0] += int((x >= 100).sum())
            tot[key][1] += len(x)
    for key, (mv, n) in tot.items():
        reg.add(f"mocap.gaps.{key}.moving_pct", 100 * mv / n, "% rows", "v2+mocap",
                f"unannotated gaps {'>= 1 s' if key == 'ge1s' else '< 1 s'}: time with the fastest marker >= 100 mm/s")
        reg.add(f"mocap.gaps.{key}.s", n * ROW_S, "s", "v2+mocap", f"unannotated time in gaps {'>= 1 s' if key == 'ge1s' else '< 1 s'} (rows with markers)")
    mv = sum(x[0] for x in tot.values()); n = sum(x[1] for x in tot.values())
    reg.add("mocap.gaps.all.moving_pct", 100 * mv / n, "% rows", "v2+mocap", "all unannotated time: fastest marker >= 100 mm/s")

    # ---- trunk angle at the bend transitions
    for tag, V in (("raw", R), ("aligned", A)):
        ok_all = n_all = 0
        for name, sign in (("Debout_Penche", 1), ("Penche_Debout", -1)):
            ok = n = 0
            for u in U:
                a = ang[u]
                for nm_, s, e in V[u]["ps"].transitions:
                    if nm_ != name or V[u]["sus"][s:e].any():
                        continue
                    x = np.nanmedian(a[max(s - 15, 0): max(s - 5, 1)])
                    y = np.nanmedian(a[e + 5: e + 15])
                    if np.isfinite(x) and np.isfinite(y):
                        n += 1
                        ok += bool(sign * (y - x) > 15)
            reg.add(f"mocap.bend.{tag}.{name}", f"{ok}/{n}", "transitions", f"{tag}+mocap",
                    f"{name} annotations with a trunk-angle change > 15 deg in the expected direction ({100 * ok / n:.1f} %)")
            ok_all += ok; n_all += n
        reg.add(f"mocap.bend.{tag}.pct", 100 * ok_all / n_all, "%", f"{tag}+mocap",
                f"both bend transitions together ({ok_all}/{n_all})")

    # ---- bent rows inclined; unlabelled bends
    bent_n = bent_ok = 0
    iso = []
    for u in U:
        p, a = A[u], ang[u]
        intr = np.isin(p["body"], list(NAMED | {AT}))
        m = (p["st"] == C.S_CLASSES.index("Penche")) & p["reg"] & ~intr & np.isfinite(a)
        bent_n += int(m.sum()); bent_ok += int((a[m] > 30).sum())
        hip = M.hip_height(mk[u])
        hipref = float(np.nanmedian(hip[(p["st"] == 0) & p["ann"]]))
        mm = (p["st"] == 0) & ~intr & np.isfinite(a) & (a > 30) & p["reg"]
        anyt = [(s, e) for _, s, e in p["ps"].transitions]
        for _, s, e in runs(np.where(mm, "B", "").astype(object)):
            if (e - s) / FS < 0.5:
                continue
            if np.nanmin(hip[s:e]) / hipref < 0.75:
                continue
            da = min([max(0, s - b, aa - e) for aa, b in anyt] or [10 ** 9]) / FS
            if da > 0.5:
                iso.append((e - s) / FS)
    reg.add("mocap.bent.inclined_pct", 100 * bent_ok / bent_n, "% rows", "aligned+mocap", "derived posture bent: trunk inclined > 30 deg")
    reg.add("mocap.bent.s", bent_n * ROW_S, "s", "aligned+mocap", "derived posture bent, rows with a valid trunk angle")
    reg.add("mocap.unlabelled_bend.n", len(iso), "episodes", "aligned+mocap", "bends without body label (definition in the module header)")
    reg.add("mocap.unlabelled_bend.s", sum(iso), "s", "aligned+mocap", "their total duration")
    reg.add("mocap.unlabelled_bend.pct_of_bending", 100 * sum(iso) / (sum(iso) + bent_n * ROW_S), "%", "aligned+mocap",
            "unlabelled bending / (unlabelled bending + derived bent time)")

    # ---- residual sensor lag
    lags = residual_lags(ctx.v2)
    worst = 0.0
    for u, devs in lags.items():
        for dev, s in devs.items():
            if s is None:
                continue
            blk = C.DEVICES[dev][0]
            applied = ctx.stats(u)["sensors"][blk]["shift_s"]
            reg.add(f"mocap.residual.{PNAME[u]}.{dev}", f"{s['median']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}]", "s", "v2+mocap",
                    f"{PNAME[u]} {dev}: lag left after alignment (applied shift {applied:+.2f} s)")
            if applied:
                worst = max(worst, abs(s["median"]))
    reg.add("mocap.residual.max_corrected_s", worst, "s", "v2+mocap", "largest |residual lag| among the streams that were shifted")

    # ---- agreement of the annotation offsets across tracks
    raw_bytes = TIMING.read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != ctx.lab_off["source_sha256"]:
        raise RuntimeError("label_timing.json differs from the file used to build params/label_offsets.json")
    T = json.loads(raw_bytes.decode("utf-8"))
    spread = 0.0
    for u in U:
        est = [x["d"] for x in T["estimates"][u] if x["estimator"].startswith("signed") and x.get("reliable")]
        if len(est) >= 2:
            sp = max(est) - min(est)
            spread = max(spread, sp)
            reg.add(f"mocap.tracks.{PNAME[u]}.spread_s", sp, "s", "raw+mocap",
                    f"{PNAME[u]}: spread of the reliable signed offset estimates of {len(est)} tracks")
    reg.add("mocap.tracks.max_spread_s", spread, "s", "raw+mocap", "largest spread between tracks, over participants with >= 2 reliable tracks")

    # ---- manual reading (not computed)
    reg.add("manual.missing_transitions", 12, "pairs", "manual",
            "of the 18 non-connecting transition pairs, those where motion capture shows a posture change without annotation "
            "(manual reading of reports/data/2026-11-13_label_errors_evidence.txt, section B)")
