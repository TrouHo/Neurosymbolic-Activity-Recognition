"""Fold-level data assembly for training (pipeline v2 data -> model inputs and targets).

Reference: reports/design/2026-11-17_model_architecture.md §1 (inputs) and §6 (training).
- Inputs: orientation-invariant channels (features.py) + one availability flag per device;
  z-score fitted on the training participants of the fold only; unavailable = 0, no interpolation.
- Targets per 0.1-s step: l2, l1, s, w, t (C.IGNORE = no label). The posture variant is `s`
  (state implied by the two bounding transitions, also on unannotated rows; chosen 2026-11-05) or
  `s_annotated` (sensitivity run).
- Label fraction: labels of the training participants are hidden in whole 60-s blocks, the same
  blocks for every head and every model (blocks chosen from L2 supervision).
- Modality dropout (training) and evaluation configurations ALL / PHONE / HB (test).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from emob import config as C
from emob.data.features import DEVICES, channel_device, represent
from emob.data.normalize import apply_channels, fit_channels
from emob.data.splits import BLOCK_STEPS, label_fraction_mask

HEAD_KEYS = list(C.HEADS)


@dataclass
class Seq:
    participant: str
    x: np.ndarray                    # (L, K + D) float32 at 50 Hz; last D columns = device flags; L = 5 T
    y: dict[str, np.ndarray]         # head -> (T,) int64, C.IGNORE = no label
    annotated: np.ndarray            # (T,) bool: at least one label column non-empty
    masked_anomaly: np.ndarray       # (T,) bool: inside a masked part of a long label run
    channels: list[str] = field(default_factory=list)

    @property
    def n_dev(self) -> int:
        return len(DEVICES)


def load_npz(participant: str) -> dict:
    d = np.load(C.PROCESSED / f"{participant}.npz")
    return {k: d[k] for k in d.files}


def targets(d: dict, s_variant: str = "s") -> dict[str, np.ndarray]:
    return {"l2": d["l2"], "l1": d["l1"], "s": d[s_variant], "w": d["w"], "t": d["t"]}


def build_fold(train: list[str], val: str, test: str, fraction: float = 1.0, seed: int = 0,
               rep: str = "invariant", s_variant: str = "s") -> dict:
    """Normalise with training statistics; hide training labels by `fraction` (nested 60-s blocks)."""
    raw = {p: load_npz(p) for p in train + [val, test]}
    feats = {p: represent(raw[p]["x50"], raw[p]["avail50"], rep) for p in raw}
    names = feats[train[0]][2]
    mean, std = fit_channels([feats[p][0] for p in train])

    def make(p: str, hide: bool) -> Seq:
        d = raw[p]
        T = len(d["l2"])
        x = apply_channels(feats[p][0], feats[p][1], mean, std)[:T * C.STEP]
        y = {h: v.astype(np.int64) for h, v in targets(d, s_variant).items()}
        if hide and fraction < 1.0:
            keep = label_fraction_mask(y["l2"] != C.IGNORE, fraction, p, seed)
            block = np.zeros(T, bool)
            for b in range(0, T, BLOCK_STEPS):
                block[b:b + BLOCK_STEPS] = keep[b:b + BLOCK_STEPS].any()
            y = {h: np.where(block, v, C.IGNORE) for h, v in y.items()}
        return Seq(p, x, y, d["annotated"].astype(bool), d["masked_anomaly"].astype(bool), names)

    return {"train": [make(p, True) for p in train], "val": [make(val, False)], "test": [make(test, False)],
            "norm": {"channels": names, "mean": mean.tolist(), "std": std.tolist()}}


# ------------------------------------------------------------------ device switches
def device_columns(seq: Seq, device: str) -> list[int]:
    """Columns of x that belong to a device: its channels and its flag."""
    j = DEVICES.index(device)
    K = len(seq.channels)
    return list(np.flatnonzero(channel_device(seq.channels) == j)) + [K + j]


def with_devices(seq: Seq, devices: list[str]) -> Seq:
    """Copy of seq where every device not in `devices` is switched off (channels 0, flag 0)."""
    x = seq.x.copy()
    for dvc in DEVICES:
        if dvc not in devices:
            x[:, device_columns(seq, dvc)] = 0.0
    return replace(seq, x=x)


def eval_config(seq: Seq, name: str) -> Seq | None:
    """Evaluation configuration ALL / PHONE / HB; None when the participant has none of its devices
    (HB is reported only on participants with both hand and belt somewhere in the recording)."""
    devs = C.EVAL_CONFIGS[name]
    K = len(seq.channels)
    present = [d for d in devs if seq.x[:, K + DEVICES.index(d)].any()]
    if name == "HB" and len(present) < 2:
        return None
    if not present:
        return None
    return with_devices(seq, devs)


# ------------------------------------------------------------------ training helpers
def class_weights(seqs: list[Seq], head: str, n_classes: int) -> np.ndarray:
    """Inverse square root of class frequency in the (label-masked) training data, mean 1 over
    present classes; 0 for absent classes."""
    lab = np.concatenate([s.y[head] for s in seqs])
    counts = np.bincount(lab[lab != C.IGNORE], minlength=n_classes).astype(float)
    w = np.zeros(n_classes)
    present = counts > 0
    w[present] = 1.0 / np.sqrt(counts[present])
    w[present] /= w[present].mean()
    return w


def sample_crops(seqs: list[Seq], n: int, crop_steps: int, rng: np.random.Generator):
    """n random crops of `crop_steps` steps; recording chosen with probability ∝ its length.
    Returns x (n, crop_steps*5, K+D) and {head: (n, crop_steps)}."""
    lens = np.array([len(s.y["l2"]) for s in seqs])
    probs = np.maximum(lens - crop_steps, 1) / np.maximum(lens - crop_steps, 1).sum()
    xs, ys = [], {h: [] for h in HEAD_KEYS}
    for i in rng.choice(len(seqs), size=n, p=probs):
        sq = seqs[i]
        T = len(sq.y["l2"])
        t0 = int(rng.integers(0, max(T - crop_steps, 0) + 1))
        t1 = t0 + crop_steps
        xs.append(sq.x[t0 * C.STEP:t1 * C.STEP])
        for h in HEAD_KEYS:
            ys[h].append(sq.y[h][t0:t1])
    return np.stack(xs), {h: np.stack(v) for h, v in ys.items()}


def modality_dropout(x: np.ndarray, channels: list[str], p: float, rng: np.random.Generator) -> np.ndarray:
    """x: (B, L, K + D). For each crop, every available device is switched off (channels and flag
    set to 0) with probability p; at least one available device is always kept (Neverova et al.
    2016, ModDrop; p = 0.2, rationale register §2)."""
    x = x.copy()
    K = len(channels)
    dev_of = channel_device(channels)
    for b in range(len(x)):
        avail = [j for j in range(len(DEVICES)) if x[b, :, K + j].max() > 0]
        if len(avail) < 2:
            continue
        drop = [j for j in avail if rng.random() < p]
        if len(drop) == len(avail):
            drop.remove(int(rng.choice(drop)))
        for j in drop:
            x[b, :, np.flatnonzero(dev_of == j)] = 0.0
            x[b, :, K + j] = 0.0
    return x
