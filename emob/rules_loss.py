"""Differentiable losses for rules A1–A7 (approved 2026-11-16; formulas approved 2026-11-18).

Reference: reports/design/2026-11-17_model_architecture.md §4 and Explanation.md §2–5.

A rule loss only compares the model's own predicted probabilities with each other; it needs no
label. Truth values follow product fuzzy logic (Badreddine et al. 2022): "A and B" = p(A) p(B),
"not A" = 1 - p(A). A1 is a semantic loss (Xu et al. 2018). A7 uses the Gödel conjunction (min)
because a product over 51 steps vanishes even when the arg-max sequence violates the rule.

Per-step violation v(t), one step = 0.1 s:
  A1  -log sum_g p_L1(g,t) sum_{c in g} p_L2(c,t)                         (hierarchy, L1 <-> L2)
  A2  sum_{X_Y} p_L2(X_Y,t) (1 - p_L2(X_Y,t-1)) (1 - p_S(X,t-1))         (transition starts from X)
  A3  [1 - sum_k p_S(k,t-1) p_S(k,t)] [1 - max_{|tau|<=10} p_T(t+tau)]   (posture changes need T)
  A4  p_W(t) [p_S(Assis,t) + p_S(Couche,t)]                               (no walking sitting/lying)
  A5  p_W(t) [p_S(Agenou,t) + p_S(Accroupi,t)]                            (no walking kneeling/squatting)
  A6  sum_{k=1..9} (1 - p_W(t-1)) prod_{j<k} p_W(t+j) (1 - p_W(t+k))      (walking run of exactly k < 10 steps)
  A7  sum_{c specific} min_{j=0..50} p_L2(c,t+j)                          (specific class held > 5 s)

Each rule is averaged over the anchor steps t where the mask m(t) = 1 (NS1/NS2: labelled steps;
NS3: every step) and where the rule is computable inside the crop:
  A2, A3: t >= 1; A3 also needs the +-10-step window inside the crop (10 <= t <= T-11);
  A6: only runs that start and end inside the crop are counted (a run cut by the crop edge has an
      unknown length); A7: t <= T-51.
On one-hot probabilities the sum of v equals the number of violations counted by
emob.metrics.rule_violations (A7: the summed excess length, run length - 50, of the long runs);
tested in tests/test_rules_loss.py, so loss and metric measure the same thing.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from emob import config as C
from emob.data.labels import L2_TO_L1_IDX
from emob.rules_spec import (A3_HALF_WINDOW, GROUP, HEADS_NEEDED, KNEEL_SQUAT, MAX_SPECIFIC_STEPS,  # noqa: F401
                             MIN_WALK_STEPS, RULES, SIT_LIE, SPECIFIC_IDX, TRANSITION_FROM, TRANSITION_IDX)

EPS = 1e-6


def probabilities(logits: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Softmax for L2, L1, S -> (B, K, T); sigmoid for W, T -> (B, T)."""
    p = {}
    for h, z in logits.items():
        p[h] = torch.sigmoid(z[:, 0]) if h in C.BINARY_HEADS else torch.softmax(z, dim=1)
    return p


def violations(p: dict[str, torch.Tensor], rule: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-step violation v (B, T) and the bool mask of anchor steps where it is defined."""
    ref = next(iter(p.values()))
    B, T = ref.shape[0], ref.shape[-1]
    v = ref.new_zeros(B, T)
    ok = torch.zeros(B, T, dtype=torch.bool, device=ref.device)

    if rule == "A1":
        parent = torch.as_tensor(L2_TO_L1_IDX, device=ref.device)
        agree = (p["l2"] * p["l1"][:, parent]).sum(1)
        v = -torch.log(agree.clamp_min(EPS))
        ok[:] = True
    elif rule == "A2":
        pl = p["l2"][:, TRANSITION_IDX]                              # (B, 10, T)
        onset = pl[:, :, 1:] * (1 - pl[:, :, :-1])
        not_from = 1 - p["s"][:, TRANSITION_FROM, :-1]
        v[:, 1:] = (onset * not_from).sum(1)
        ok[:, 1:] = True
    elif rule == "A3":
        ps = p["s"]
        change = 1 - (ps[:, :, 1:] * ps[:, :, :-1]).sum(1)           # change between t-1 and t
        k = 2 * A3_HALF_WINDOW + 1
        t_near = F.max_pool1d(p["t"].unsqueeze(1), k, stride=1, padding=A3_HALF_WINDOW).squeeze(1)
        v[:, 1:] = change * (1 - t_near[:, 1:])
        ok[:, A3_HALF_WINDOW:T - A3_HALF_WINDOW] = True
    elif rule in ("A4", "A5"):
        states = SIT_LIE if rule == "A4" else KNEEL_SQUAT
        v = p["w"] * p["s"][:, states].sum(1)
        ok[:] = True
    elif rule == "A6":
        pw = p["w"]
        run = pw                                                     # run_k(t) = prod_{j<k} pw(t+j)
        for k in range(1, MIN_WALK_STEPS):
            if k > 1:
                run = run[:, :T - k + 1] * pw[:, k - 1:]
            if T - k - 1 <= 0:
                break
            # anchors t = 1 .. T-k-1: needs pw(t-1) and pw(t+k) inside the crop
            v[:, 1:T - k] = v[:, 1:T - k] + (1 - pw[:, :T - k - 1]) * run[:, 1:T - k] * (1 - pw[:, k + 1:])
        ok[:] = True
    elif rule == "A7":
        n = MAX_SPECIFIC_STEPS + 1
        if T >= n:
            pl = p["l2"][:, SPECIFIC_IDX]                            # (B, 16, T)
            run_min = -F.max_pool1d(-pl, n, stride=1)                # min over [t, t+50], (B, 16, T-50)
            v[:, :T - n + 1] = run_min.sum(1)
            ok[:, :T - n + 1] = True
    else:
        raise ValueError(rule)
    return v, ok


def rule_losses(logits: dict[str, torch.Tensor], mask: torch.Tensor,
                rules: list[str] | None = None) -> dict[str, torch.Tensor]:
    """Mean violation of each rule over the anchor steps with mask = 1 (mask: (B, T) bool)."""
    p = probabilities(logits)
    out = {}
    for r in rules or RULES:
        if not HEADS_NEEDED[r] <= set(p):
            raise ValueError(f"rule {r} needs heads {sorted(HEADS_NEEDED[r])}")
        v, ok = violations(p, r)
        w = (mask & ok).float()
        out[r] = (v * w).sum() / w.sum().clamp_min(1.0)
    return out


def weighted_sum(losses: dict[str, torch.Tensor], lam: dict[str, float]) -> torch.Tensor:
    """lam: {'hier': ..., 'hard': ..., 'soft': ...}; rule r gets lam[GROUP[r]]."""
    total = 0.0
    for r, x in losses.items():
        total = total + lam[GROUP[r]] * x
    return total
