"""Per-channel z-score fitted on training participants only; unusable rows -> 0 + flag.

Works for K sensor blocks of 3 axes: raw (N, 3K) + usable (N, K) -> (N, 4K).
"""
from __future__ import annotations

import numpy as np


def fit_zscore(accs: list[np.ndarray], usables: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of each channel over the usable rows of the training recordings."""
    K = usables[0].shape[1]
    mean, std = np.zeros(3 * K), np.ones(3 * K)
    for s in range(K):
        cols = slice(3 * s, 3 * s + 3)
        vals = np.concatenate([a[u[:, s], cols] for a, u in zip(accs, usables)])
        if len(vals) == 0:
            raise ValueError(f"sensor {s}: no usable rows in the training set")
        mean[cols] = vals.mean(axis=0)
        std[cols] = vals.std(axis=0)
    if np.any(std <= 0):
        raise ValueError("zero variance channel")
    return mean, std


def fit_channels(xs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pipeline v2: per-channel mean and std over the finite values of the training recordings
    (unavailable samples are NaN, so a participant without a device does not contribute to it)."""
    allx = np.concatenate(xs, axis=0)
    finite = np.isfinite(allx)
    if not finite.any(axis=0).all():
        raise ValueError(f"channels without any value in the training set: {np.flatnonzero(~finite.any(axis=0))}")
    mean = np.nanmean(allx, axis=0)
    std = np.nanstd(allx, axis=0)
    if np.any(std <= 0):
        raise ValueError("zero variance channel")
    return mean, std


def apply_channels(x: np.ndarray, flags: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """(N, K) channels (NaN = unavailable) + (N, D) device flags -> (N, K + D) float32;
    normalised channels are 0 where unavailable (no interpolation)."""
    z = np.nan_to_num((x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([z, flags.astype(z.dtype)], axis=1).astype(np.float32)


def apply_zscore(acc: np.ndarray, usable: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """(N, 3K) raw -> (N, 4K): 3K normalised channels (0 where unusable) + K availability flags."""
    K = usable.shape[1]
    z = (acc - mean) / std
    for s in range(K):
        z[~usable[:, s], 3 * s:3 * s + 3] = 0.0
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([z, usable.astype(z.dtype)], axis=1).astype(np.float32)
