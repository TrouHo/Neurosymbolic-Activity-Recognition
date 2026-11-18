"""B0: hand-crafted window features + Random Forest (reference point, L2 only).

Reference: reports/design/2026-11-17_model_architecture.md §5; per-configuration training approved
2026-11-18 (Q5): the neural models are trained once with modality dropout, a Random Forest cannot
learn to cope with missing sensors that way, so one forest is trained per sensor configuration
(ALL / PHONE / HB) on the training participants' steps where at least one device of that
configuration is available. B0 is thus evaluated in its best condition for each configuration.

Features: for every 0.1-s step, a 2-s window centred on the step, on the same normalised input
channels as the neural models (13 orientation-invariant channels + 4 device flags):
mean, SD, min, max, 25th and 75th percentile, mean absolute first difference, dominant frequency
(DC excluded), share of spectral energy in 0.5–3 Hz (per channel), and the mean of each flag.

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
from emob.data.dataset import build_fold, eval_config, with_devices
from emob.data.features import DEVICES
from emob.data.splits import loso_folds
from emob.evaluate import evaluate

WIN = 100  # samples = 2 s at 50 Hz


def features(x: np.ndarray, n_channels: int) -> np.ndarray:
    """x: (L, K + D) normalised channels + device flags -> (T, 9 K + D), one row per 0.1-s step."""
    T = len(x) // C.STEP
    K = n_channels
    pad = np.pad(x.astype(np.float32), ((WIN // 2, WIN // 2), (0, 0)), mode="edge")
    centres = np.arange(T) * C.STEP + C.CENTRE
    w = sliding_window_view(pad, WIN, axis=0)[centres]          # (T, K + D, WIN)
    sig, flags = w[:, :K, :], w[:, K:, :]
    spec = np.abs(np.fft.rfft(sig - sig.mean(-1, keepdims=True), axis=-1)) ** 2
    freqs = np.fft.rfftfreq(WIN, d=1 / C.FS)
    band = (freqs >= 0.5) & (freqs <= 3.0)
    feats = [sig.mean(-1), sig.std(-1), sig.min(-1), sig.max(-1),
             np.percentile(sig, 25, axis=-1), np.percentile(sig, 75, axis=-1),
             np.abs(np.diff(sig, axis=-1)).mean(-1),
             freqs[spec[..., 1:].argmax(-1) + 1],
             spec[..., band].sum(-1) / (spec.sum(-1) + 1e-9),
             flags.mean(-1)]
    return np.concatenate(feats, axis=1).astype(np.float32)


def config_steps(seq, devices: list[str]) -> np.ndarray:
    """Steps where at least one device of the configuration is available (flag at the step centre)."""
    K = len(seq.channels)
    cols = [K + DEVICES.index(d) for d in devices]
    centres = np.arange(len(seq.y["l2"])) * C.STEP + C.CENTRE
    return seq.x[centres][:, cols].max(1) > 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--rep", choices=["invariant", "raw"], default="invariant")
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default="results/phase1")
    a = ap.parse_args(argv)
    rid = f"RF-B0_frac{a.fraction:g}_fold{a.fold}_seed{a.seed}" + (f"_{a.rep}" if a.rep != "invariant" else "")
    out = Path(a.out).expanduser() / rid
    out.mkdir(parents=True, exist_ok=True)

    fold = loso_folds()[a.fold]
    data = build_fold(fold["train"], fold["val"], fold["test"], a.fraction, a.seed, a.rep)
    K = len(data["norm"]["channels"])
    te = data["test"][0]
    test, info = {}, {}
    t0 = time.time()
    for cfg, devs in C.EVAL_CONFIGS.items():
        seq_te = eval_config(te, cfg)
        if seq_te is None:
            continue
        Xs, ys = [], []
        for s in data["train"]:
            on = config_steps(s, devs) & (s.y["l2"] != C.IGNORE)
            if on.any():
                Xs.append(features(with_devices(s, devs).x, K)[on])
                ys.append(s.y["l2"][on])
        X, y = np.concatenate(Xs), np.concatenate(ys)
        rf = RandomForestClassifier(a.trees, class_weight="balanced_subsample", n_jobs=a.jobs, random_state=a.seed)
        rf.fit(X, y)
        pred = rf.classes_[rf.predict_proba(features(seq_te.x, K)).argmax(1)]
        test[cfg] = evaluate({"l2": pred.astype(np.int64)}, te.y)
        info[cfg] = {"n_train_steps": int(len(y)), "n_train_participants": len(Xs)}
    train_time = time.time() - t0
    result = {"run_id": rid, "code_version": __version__, "args": vars(a), "fold": fold,
              "n_features": 9 * K + len(DEVICES), "per_config": info, "train_time_s": train_time, "test": test}
    (out / "metrics.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"{rid}: test L2 " + ", ".join(f"{c} {m['l2_macro_f1']:.3f}" for c, m in test.items())
          + f", time {train_time:.0f}s")


if __name__ == "__main__":
    main()
