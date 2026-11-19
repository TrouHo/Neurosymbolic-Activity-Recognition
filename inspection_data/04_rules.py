"""04 Rules: agreement of rules A1-A7 with the prepared targets (definitions in emob/data/rule_stats.py,
shared with the gate check), extra facts used to justify the rules, and the rejected candidates.
"""
from __future__ import annotations

import numpy as np

from emob import config as C
from emob.data import rule_stats as RS
from emob.data.labels import S_INDEX, runs
from inspection_data.common import PNAME, STEP_S, Context, Registry

SUSPECT_S = 12.0     # specific labels longer than this are the abnormal annotations (excluded below)


def run(ctx: Context, reg: Registry):
    reg.section = "04 Rules"
    D, U = ctx.v2, ctx.users
    r = RS.a1(D)
    reg.add("rule.A1.ok", r["ok"], "steps", "v2", "labelled activity steps whose group target is the group of the activity")
    reg.add("rule.A1.n", r["n"], "steps", "v2", "labelled activity steps")
    r = RS.a2_pairs(ctx.manifest)
    reg.add("rule.A2.pairs_chain", r["chain"], "pairs", "v2", "successive named transitions of the body track that connect (X_Y then Y_Z)")
    reg.add("rule.A2.pairs_break", r["break"], "pairs", "v2", "successive named transitions that do not connect")
    reg.add("rule.A2.pairs_other", r["with_other"], "pairs", "v2", "pairs involving Other posture transition (not checkable)")
    reg.add("rule.A2.chain_pct", 100 * r["chain"] / (r["chain"] + r["break"]), "%", "v2", "connecting pairs / checkable pairs")
    # A2 at transition starts in the activity column
    ok = bad = unk = 0
    src = {c: S_INDEX[C.transition_states(c)[0]] for c in C.POSTURE_TRANSITIONS}
    for d in D.values():
        l2, s = d["l2"], d["s"]
        for t in range(1, len(l2)):
            if l2[t] >= 0 and C.L2_CLASSES[l2[t]] in src and l2[t - 1] != l2[t]:
                if s[t - 1] < 0:
                    unk += 1
                elif s[t - 1] == src[C.L2_CLASSES[l2[t]]]:
                    ok += 1
                else:
                    bad += 1
    reg.add("rule.A2.starts_ok", ok, "starts", "v2", "starts of a named transition in the activity column whose previous-step posture is its source")
    reg.add("rule.A2.starts_violated", bad, "starts", "v2", "same, previous-step posture known but different")
    reg.add("rule.A2.starts_unknown", unk, "starts", "v2", "same, previous-step posture unknown")
    r = RS.a3(D)
    reg.add("rule.A3.ok", r["ok"], "events", "v2+mocap", "head-height changes (> 15 % of standing height within 2 s) overlapping a transition target within +-1 s")
    reg.add("rule.A3.n", r["n"], "events", "v2+mocap", "head-height changes in annotated time, outside masked annotations")
    reg.add("rule.A3.pct", 100 * r["ok"] / r["n"], "%", "v2+mocap", "A3 agreement")
    for u, (k, n) in r["per_participant"].items():
        reg.add(f"rule.A3.{PNAME[u]}", f"{k}/{n}", "events", "v2+mocap", f"{PNAME[u]}: A3 agreement ({100 * k / n:.1f} %)")
    r = RS.a4_a5(D)
    reg.add("rule.A4.violations", r["a4_violations"], "steps", "v2", "walking steps while sitting or lying")
    reg.add("rule.A5.violations", r["a5_violations"], "steps", "v2", "walking steps while kneeling or squatting")
    reg.add("rule.A4A5.walking_known_posture", r["walking_steps_known_posture"], "steps", "v2", "walking steps with a known posture")
    for k, name in enumerate(C.S_CLASSES):
        sec = sum(int(((d["w"] == 1) & (d["s"] == k)).sum()) for d in D.values()) * STEP_S
        reg.add(f"rule.walking_in.{name}_s", sec, "s", "v2", f"walking time while the posture is {name}")
    r = RS.a6(D)
    reg.add("rule.A6.ok", r["ok"], "bouts", "v2", "walking bouts (maximal runs of walking target = 1) lasting >= 1 s")
    reg.add("rule.A6.n", r["n"], "bouts", "v2", "walking bouts")
    reg.add("rule.A6.ge_0_5s", r["ge_0_5s"], "bouts", "v2", "walking bouts lasting >= 0.5 s")
    reg.add("rule.A6.median_s", r["median_s"], "s", "v2", "median walking bout duration")
    r = RS.a7(D)
    reg.add("rule.A7.ok", r["ok"], "segments", "v2", "segments of the 16 specific classes lasting <= 5 s (activity column)")
    reg.add("rule.A7.n", r["n"], "segments", "v2", "segments of the 16 specific classes")
    for a, nm, u in r["longer"]:
        reg.add(f"rule.A7.longer.{PNAME[u]}.{nm}_s", a, "s", "v2", f"{PNAME[u]}: specific segment longer than 5 s")
    a, nm, u = r["longest_within_5s"]
    reg.add("rule.A7.longest_within_5s", a, "s", "v2", f"longest specific segment of at most 5 s ({nm}, {PNAME[u]})")

    # ---- rejected candidates (raw tracks; abnormal long annotations excluded as in the original check)
    raw = ctx.raw
    sus = {}
    for u in U:
        m = np.zeros(raw[u]["n"], bool)
        for col in (C.BODY_COL, "etiq mvt_bras", C.LEG_COL):
            for name, s, e in runs(raw[u]["fam"][col]):
                if name not in ("inactif", "Marcher", "Autre_Mouvement_Bras", "Autre_Mouvement_Jambe", "Debut", "Fin") and (e - s) / C.FS > SUSPECT_S:
                    m[s:e] = True
        sus[u] = m
    segs = ov = 0
    for u in U:
        tr = np.isin(raw[u]["fam"][C.BODY_COL], C.POSTURE_TRANSITIONS)
        for name, s, e in runs(raw[u]["fam"][C.LEG_COL]):
            if name == "Marcher" and not sus[u][s:e].any():
                segs += 1
                ov += bool(tr[s:e].any())
    reg.add("rule.rejected.walk_overlaps_transition", ov, "segments", "raw", "walking segments of the leg track overlapping a named transition (abnormal annotations excluded)")
    reg.add("rule.rejected.walk_segments", segs, "segments", "raw", "walking segments of the leg track (abnormal annotations excluded)")
    for track, col, labels in (("arm", "etiq mvt_bras", ["Lever_Bras", "Baisser_Bras", "Lever_2_Bras", "Baisser_2_Bras"]),
                               ("leg", C.LEG_COL, ["Lever_Jambe", "Baisser_Jambe"])):
        n = alt = 0
        for u in U:
            seq = [name for name, s, e in runs(raw[u]["fam"][col]) if name in labels and not sus[u][s:e].any()]
            for x, y in zip(seq[:-1], seq[1:]):
                n += 1
                alt += x.startswith("Lever") != y.startswith("Lever")
        reg.add(f"rule.rejected.alternation.{track}_pct", 100 * alt / n, "%", "raw",
                f"consecutive raise/lower labels of the {track} track that alternate direction ({alt}/{n} pairs)")
