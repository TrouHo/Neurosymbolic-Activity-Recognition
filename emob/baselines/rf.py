"""B0: hand-crafted window features + Random Forest (reference point).

For every 0.1-s step, features are computed on a 2-s window centred on the step
(normalised channels, same preprocessing as the neural models).

    python -m emob.baselines.rf --fold 0 --seed 0 --fraction 1.0 --out results/phase1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestClassifier

from emob import __version__, config as C
from emob.data.dataset import build_fold
from emob.data.labels import L2_TO_L1_IDX
from emob.data.splits import loso_folds
from emob.evaluate import evaluate

WIN = 100  # samples = 2 s at 50 Hz


def features(x: np.ndarray) -> np.ndarray:
    """x: (L, 4K) normalised channels + K flags -> (T, 9*3K + K) features, one row per 0.1-s step."""
    T = len(x) // C.STEP
    K = x.shape[1] // 4
    pad = np.pad(x, ((WIN // 2, WIN // 2), (0, 0)), mode="edge")
    centres = np.arange(T) * C.STEP + C.CENTRE
    w = sliding_window_view(pad, WIN, axis=0)[centres]          # (T, 4K, WIN)
    acc, flags = w[:, :3 * K, :], w[:, 3 * K:, :]
    spec = np.abs(np.fft.rfft(acc - acc.mean(-1, keepdims=True), axis=-1)) ** 2
    freqs = np.fft.rfftfreq(WIN, d=1 / C.FS)
    band = (freqs >= 0.5) & (freqs <= 3.0)
    feats = [acc.mean(-1), acc.std(-1), acc.min(-1), acc.max(-1),
             np.percentile(acc, 25, axis=-1), np.percentile(acc, 75, axis=-1),
             np.abs(np.diff(acc, axis=-1)).mean(-1),
             freqs[spec[..., 1:].argmax(-1) + 1],                 # dominant frequency (DC excluded)
             spec[..., band].sum(-1) / (spec.sum(-1) + 1e-9),     # share of energy in 0.5–3 Hz
             flags.mean(-1)]
    return np.concatenate(feats, axis=1).astype(np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--s-variant", choices=["s_all", "s"], default="s_all")
    ap.add_argument("--inputs", choices=["M", "MP"], default="MP")
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default="results/phase1")
    a = ap.parse_args(argv)
    rid = f"B0_{a.inputs}_frac{a.fraction:g}_{a.s_variant}_fold{a.fold}_seed{a.seed}"
    out = Path(a.out) / rid
    out.mkdir(parents=True, exist_ok=True)

    fold = loso_folds()[a.fold]
    data = build_fold(fold["train"], fold["val"], fold["test"], a.s_variant, a.fraction, a.seed, a.inputs)
    t0 = time.time()
    Xtr = np.concatenate([features(s.x) for s in data["train"]])
    y2 = np.concatenate([s.l2 for s in data["train"]])
    ys = np.concatenate([s.s for s in data["train"]])
    rf2 = RandomForestClassifier(a.trees, class_weight="balanced_subsample", n_jobs=a.jobs, random_state=a.seed)
    rf2.fit(Xtr[y2 != C.IGNORE], y2[y2 != C.IGNORE])
    rfs = RandomForestClassifier(a.trees, class_weight="balanced_subsample", n_jobs=a.jobs, random_state=a.seed)
    rfs.fit(Xtr[ys != C.IGNORE], ys[ys != C.IGNORE])
    train_time = time.time() - t0

    te = data["test"][0]
    Xte = features(te.x)
    p2 = rf2.classes_[rf2.predict_proba(Xte).argmax(1)]
    ps = rfs.classes_[rfs.predict_proba(Xte).argmax(1)]
    pred = {"l2": p2, "l1": L2_TO_L1_IDX[p2], "s": ps}
    metrics = evaluate(pred, {"l1": te.l1, "l2": te.l2, "s": te.s_masked, "s_all": te.s_all})
    result = {"run_id": rid, "code_version": __version__, "args": vars(a), "fold": fold,
              "n_features": int(Xtr.shape[1]), "n_train_l2": int((y2 != C.IGNORE).sum()),
              "train_time_s": train_time, "test": metrics}
    (out / "metrics.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"{rid}: test L2 {metrics['l2_macro_f1']:.3f}, L1 {metrics['l1_macro_f1']:.3f}, time {train_time:.0f}s")


if __name__ == "__main__":
    main()
