"""Diagnostic (2026-11-21): is the low L2 macro-F1 a bias towards frequent classes?

Post-hoc logit adjustment (Menon et al. 2021, "Long-tail learning via logit adjustment"):
predict argmax_c [ z_c - tau * log prior_c ], prior = class frequency of L2 in the six
training participants of the fold. tau = 0 is the plain arg-max. No retraining.

Only test logits are saved, so tau is scanned on the test participants: this is a DIAGNOSTIC of
how much a frequency bias costs, never a reported result (choosing tau on test is leakage).

Main metric as in emob.evaluate: per-participant macro-F1 over the classes present in that
participant's labels, then the mean over participants.

    .venv\\Scripts\\python.exe -m diagnostics.logit_adjustment_check [--results DIR] [--head ALL]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from emob import config as C

V2 = Path(__file__).resolve().parents[2] / "data" / "processed" / "v2"
DEFAULT = Path(__file__).resolve().parents[2] / "data" / "results_local" / "TCN-B2_frac1_seed0"
TAUS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
N_CLASSES = len(C.L2_CLASSES)


def f1_per_class(g: np.ndarray, p: np.ndarray, classes) -> dict[int, float]:
    out = {}
    for c in classes:
        tp = ((g == c) & (p == c)).sum()
        den = (g == c).sum() + (p == c).sum()
        out[int(c)] = 2 * tp / den if den else 0.0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=DEFAULT)
    ap.add_argument("--config", default="ALL")
    a = ap.parse_args(argv)

    labels = {k: np.load(V2 / f"user{k + 1}.npz")["l2"] for k in range(8)}
    data = []
    for k in range(8):
        val = (k + 1) % 8
        train = [j for j in range(8) if j not in (k, val)]
        counts = np.ones(N_CLASSES)                      # +1 smoothing
        for j in train:
            y = labels[j]
            counts += np.bincount(y[y != C.IGNORE], minlength=N_CLASSES)
        logprior = np.log(counts / counts.sum())
        z = np.load(a.results / f"fold{k}" / "test_logits.npz")[f"{a.config}_l2"]
        g = labels[k]
        z = z[:, :len(g)]
        m = g != C.IGNORE
        data.append((g[m], z[:, m], logprior))

    print(f"results {a.results.name}, config {a.config}")
    print(f"{'tau':>5s} {'macroF1':>8s} {'acc':>6s} {'%pred Autre_Bras':>17s} {'#classes F1>0.1':>16s}")
    autre = C.L2_CLASSES.index("Autre_Mouvement_Bras")
    per_tau = {}
    for tau in TAUS:
        mf, G, P = [], [], []
        for g, z, lp in data:
            p = (z - tau * lp[:, None]).argmax(0)
            f = f1_per_class(g, p, np.unique(g))
            mf.append(np.mean(list(f.values())))
            G.append(g); P.append(p)
        g, p = np.concatenate(G), np.concatenate(P)
        pooled = f1_per_class(g, p, range(N_CLASSES))
        per_tau[tau] = pooled
        print(f"{tau:5.2f} {np.mean(mf):8.3f} {(g == p).mean():6.3f} {(p == autre).mean():17.1%} "
              f"{sum(v > 0.1 for v in pooled.values()):16d}")
    g = np.concatenate([d[0] for d in data])
    print("\nper-class F1 (pooled over the 8 test participants), by tau:")
    print(f"{'class':24s} {'sec':>6s} " + " ".join(f"{t:>5.2f}" for t in TAUS))
    order = np.argsort(-np.bincount(g, minlength=N_CLASSES))
    for c in order:
        n = (g == c).sum()
        if n == 0:
            continue
        print(f"{C.L2_CLASSES[c]:24s} {n / 10:6.1f} " + " ".join(f"{per_tau[t][int(c)]:5.2f}" for t in TAUS))


if __name__ == "__main__":
    main()
