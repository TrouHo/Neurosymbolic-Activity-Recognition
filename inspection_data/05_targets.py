"""05 Targets: coverage of the five prediction heads, posture episodes and class weights (v2, steps).

Heads: activity (L2), group (L1), posture (S), walking (W), transition (T); a step has a target for a
head when its value is not IGNORE. Posture episode = maximal run of one known posture state.
Class weights: count n_c of labelled activity steps per class over all 8 participants (the training
code computes them on the training participants of each fold); ratio = weight(rarest)/weight(most
frequent). Effective number (Cui et al. 2019): E_n = (1 - beta^n) / (1 - beta), weight ~ 1 / E_n.
"""
from __future__ import annotations

import numpy as np

from emob import config as C
from inspection_data.common import STEP_S, Context, Registry, segments

HEADS = {"l2": "activity", "l1": "group", "s": "posture", "w": "walking", "t": "transition"}
RECEPTIVE_FIELD_STEPS = 511


def run(ctx: Context, reg: Registry):
    reg.section = "05 Targets"
    D = ctx.v2
    total = sum(len(d["l2"]) for d in D.values())
    reg.add("tgt.total_steps", total, "steps", "v2", "all output steps of the 8 recordings")
    for h, name in HEADS.items():
        lab = sum(int((d[h] != C.IGNORE).sum()) for d in D.values())
        reg.add(f"tgt.{name}.labelled_s", lab * STEP_S, "s", "v2", f"steps with a {name} target")
        reg.add(f"tgt.{name}.labelled_pct", 100 * lab / total, "% steps", "v2", f"share of all steps with a {name} target")
        if h in ("w", "t"):
            pos = sum(int((d[h] == 1).sum()) for d in D.values())
            reg.add(f"tgt.{name}.positive_s", pos * STEP_S, "s", "v2", f"steps with {name} target = 1")
    # posture episodes
    ep = {k: [] for k in range(len(C.S_CLASSES))}
    for d in D.values():
        for v, s, e in segments(d["s"]):
            ep[int(v)].append((e - s) * STEP_S)
    allep = np.concatenate([np.array(x) for x in ep.values()])
    for k, name in enumerate(C.S_CLASSES):
        reg.add(f"tgt.posture.{name}.longest_episode_s", max(ep[k]), "s", "v2", f"longest {name} episode")
    reg.add("tgt.posture.n_episodes", len(allep), "episodes", "v2", "posture episodes (all states)")
    reg.add("tgt.posture.within_receptive_field_pct", 100 * (allep <= RECEPTIVE_FIELD_STEPS * STEP_S).mean(), "% episodes", "v2",
            f"posture episodes no longer than the TCN receptive field ({RECEPTIVE_FIELD_STEPS} steps)")
    # class weights
    n = np.array([sum(int((d["l2"] == i).sum()) for d in D.values()) for i in range(len(C.L2_CLASSES))])
    reg.add("tgt.class_steps.max", int(n.max()), "steps", "v2", f"labelled steps of the most frequent class ({C.L2_CLASSES[n.argmax()]})")
    reg.add("tgt.class_steps.min", int(n.min()), "steps", "v2", f"labelled steps of the rarest class ({C.L2_CLASSES[n.argmin()]})")
    reg.add("tgt.weight_ratio.inverse", n.max() / n.min(), "-", "v2", "weight ratio rarest/most frequent with 1/n")
    reg.add("tgt.weight_ratio.inverse_sqrt", np.sqrt(n.max() / n.min()), "-", "v2", "weight ratio with 1/sqrt(n)")
    for beta in (0.999, 0.9999):
        E = lambda m: (1 - beta ** m) / (1 - beta)
        reg.add(f"tgt.weight_ratio.effective_{beta}", E(n.max()) / E(n.min()), "-", "v2", f"weight ratio with the effective number, beta = {beta}")
