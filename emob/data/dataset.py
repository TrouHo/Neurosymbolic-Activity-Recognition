"""Fold-level data assembly: normalised inputs, label masks, random training crops."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from emob import config as C
from emob.data.normalize import apply_zscore, fit_zscore
from emob.data.splits import BLOCK_STEPS, label_fraction_mask


@dataclass
class Seq:
    participant: str
    x: np.ndarray            # (L, 8) float32 at 50 Hz, L multiple of 5
    l1: np.ndarray           # (T,) int64, T = L / 5
    l2: np.ndarray
    s: np.ndarray            # S labels of the chosen variant ("s_all" or "s")
    s_masked: np.ndarray     # "s" (for evaluation of both variants)
    s_all: np.ndarray
    annotated: np.ndarray    # (T,) bool


def load_npz(participant: str) -> dict:
    d = np.load(C.PROCESSED / f"{participant}.npz")
    return {k: d[k] for k in d.files}


INPUTS = {
    "M": "hand ACC + belt ACC",
    "MP": "hand ACC + belt ACC + pocket-phone ACC + pocket-phone GYR",
}


def raw_inputs(d: dict, inputs: str) -> tuple[np.ndarray, np.ndarray]:
    """(N, 3K) raw signals and (N, K) usable flags for an input configuration."""
    if inputs == "M":
        return d["acc50"].astype(np.float64), d["usable50"]
    if inputs == "MP":
        return (np.concatenate([d["acc50"], d["phone50"]], axis=1).astype(np.float64),
                np.concatenate([d["usable50"], d["phone_usable50"]], axis=1))
    raise ValueError(inputs)


def build_fold(train: list[str], val: str, test: str, s_variant: str = "s_all",
               fraction: float = 1.0, seed: int = 0, inputs: str = "MP") -> dict[str, list[Seq] | dict]:
    """Normalise with training statistics; hide labels of training participants by `fraction`."""
    raw = {p: load_npz(p) for p in train + [val, test]}
    sig = {p: raw_inputs(raw[p], inputs) for p in raw}
    mean, std = fit_zscore([sig[p][0] for p in train], [sig[p][1] for p in train])

    def make(p: str, hide: bool) -> Seq:
        d = raw[p]
        T = len(d["l2"])
        L = T * C.STEP
        x = apply_zscore(sig[p][0][:L], sig[p][1][:L], mean, std)
        l1, l2 = d["l1"].astype(np.int64), d["l2"].astype(np.int64)
        s_sel = d[s_variant].astype(np.int64)
        if hide and fraction < 1.0:
            # blocks are chosen from L2 supervision only, so the mask is identical for every
            # model and every S variant; inside a kept block all heads keep their labels
            keep = label_fraction_mask(l2 != C.IGNORE, fraction, p, seed)
            block_keep = np.zeros(T, dtype=bool)
            for b in range(0, T, BLOCK_STEPS):
                block_keep[b:b + BLOCK_STEPS] = keep[b:b + BLOCK_STEPS].any()
            l1, l2, s_sel = (np.where(block_keep, a, C.IGNORE) for a in (l1, l2, s_sel))
        return Seq(p, x, l1, l2, s_sel, d["s"].astype(np.int64), d["s_all"].astype(np.int64), d["annotated"])

    return {"train": [make(p, True) for p in train], "val": [make(val, False)], "test": [make(test, False)],
            "norm": {"mean": mean.tolist(), "std": std.tolist()}}


def class_weights(seqs: list[Seq], head: str, n_classes: int) -> np.ndarray:
    """Inverse square root of class frequency in the (label-masked) training data, mean 1 over
    present classes; 0 for absent classes."""
    lab = np.concatenate([getattr(s, head) for s in seqs])
    counts = np.bincount(lab[lab != C.IGNORE], minlength=n_classes).astype(float)
    w = np.zeros(n_classes)
    present = counts > 0
    w[present] = 1.0 / np.sqrt(counts[present])
    w[present] /= w[present].mean()
    return w


def sample_crops(seqs: list[Seq], n: int, crop_steps: int, rng: np.random.Generator):
    """n random crops of `crop_steps` steps; probability proportional to recording length."""
    lens = np.array([len(s.l2) for s in seqs])
    probs = np.maximum(lens - crop_steps, 1) / np.maximum(lens - crop_steps, 1).sum()
    xs, l1s, l2s, ss = [], [], [], []
    for i in rng.choice(len(seqs), size=n, p=probs):
        sq = seqs[i]
        T = len(sq.l2)
        t0 = int(rng.integers(0, max(T - crop_steps, 0) + 1))
        t1 = t0 + crop_steps
        xs.append(sq.x[t0 * C.STEP:t1 * C.STEP])
        l1s.append(sq.l1[t0:t1]); l2s.append(sq.l2[t0:t1]); ss.append(sq.s[t0:t1])
    return np.stack(xs), np.stack(l1s), np.stack(l2s), np.stack(ss)


def modality_dropout(x: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """x: (B, L, 4K). With probability p per crop, zero one available sensor block
    (3 channels + its flag), only if at least one other block stays available."""
    x = x.copy()
    K = x.shape[-1] // 4
    for b in range(len(x)):
        if rng.random() >= p:
            continue
        avail = [s for s in range(K) if x[b, :, 3 * K + s].max() > 0]
        if len(avail) < 2:
            continue
        s = int(rng.choice(avail))
        x[b, :, 3 * s:3 * s + 3] = 0.0
        x[b, :, 3 * K + s] = 0.0
    return x
