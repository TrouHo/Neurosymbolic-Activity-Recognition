"""Leave-one-subject-out folds and label-fraction masks."""
from __future__ import annotations

import zlib

import numpy as np

PARTICIPANTS = [f"user{i}" for i in range(1, 9)]
BLOCK_STEPS = 600   # 60 s blocks for hiding labels (= training crop length; see rationale register)


def loso_folds(participants=PARTICIPANTS) -> list[dict]:
    """Fold k: test = participant k, validation = participant k+1 (cyclic), train = the other 6."""
    n = len(participants)
    folds = []
    for k in range(n):
        test = participants[k]
        val = participants[(k + 1) % n]
        train = [p for p in participants if p not in (test, val)]
        folds.append({"fold": k, "test": test, "val": val, "train": train})
    return folds


def _rng(participant: str, seed: int) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f"{participant}|{seed}".encode()))


def label_fraction_mask(labelled: np.ndarray, fraction: float, participant: str, seed: int,
                        block_steps: int = BLOCK_STEPS) -> np.ndarray:
    """Keep labels only in a random subset of contiguous blocks.

    - The timeline is cut into blocks of `block_steps` steps; only blocks containing
      at least one labelled step are candidates.
    - One random permutation per (participant, seed); a fraction keeps the first
      round(fraction * n_candidates) blocks (at least 1). Hence the masks are nested:
      10 % ⊂ 25 % ⊂ 50 % ⊂ 100 %, and identical for every model.
    Returns a bool mask (True = label kept) of the same length as `labelled`.
    """
    labelled = np.asarray(labelled, dtype=bool)
    if fraction >= 1.0:
        return labelled.copy()
    n = len(labelled)
    n_blocks = int(np.ceil(n / block_steps))
    cand = [b for b in range(n_blocks) if labelled[b * block_steps:(b + 1) * block_steps].any()]
    order = _rng(participant, seed).permutation(len(cand))
    k = max(1, int(round(fraction * len(cand))))
    keep = np.zeros(n, dtype=bool)
    for idx in order[:k]:
        b = cand[idx]
        keep[b * block_steps:(b + 1) * block_steps] = True
    return keep & labelled
