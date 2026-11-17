"""Pipeline v2 — build the training data from the raw 2022 CSV files.

Plan: reports/design/2026-11-17_pipeline_v2_plan.md (approved 2026-11-17).
Reads only data/raw (read-only) and the frozen parameters in params/*.json; contains no value
estimated from the data.

  S1 read + format checks           (io: ts_SQ grid, column layout, known label values)
  S2 sensor time alignment          (sync.py       <- params/sensor_offsets.json)
  S4 mask anomalous label runs      (anomalies.py  <- params/long_runs.json; in CSV label time)
  S3 label time alignment           (alignment.py  <- params/label_offsets.json)
  S5 targets L2, L1, S, W, T        (labels.py)
  S6 0.1-s step grid                (label = centre sample of each 5-sample step)
  S7 files: userX.npz, manifest.json, data_card.md

S4 runs before S3 because the long-run boundaries are expressed in the label time of the CSV;
masks are then shifted together with the labels.

Usage (from code/):  .venv\\Scripts\\python.exe -m emob.data.build [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from emob import __version__, config as C
from emob.data import alignment, anomalies, sync
from emob.data.labels import (L2_INDEX, annotated_mask, encode_l2, l1_from_l2, mask_posture_after,
                              posture_states, step_centres, transition_targets, walk_targets)
from emob.data.splits import PARTICIPANTS

KNOWN_ACTIVITE = set(C.L2_CLASSES) | C.PROTOCOL_MARKERS | {""}


# ---------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def content_sha256(arrays: dict) -> str:
    """Hash of the array contents (independent of zip timestamps), for the reproducibility check."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def git_state() -> dict:
    cands = ["git"] + sorted(glob(str(Path.home() / "AppData/Local/GitHubDesktop/app-*/resources/app/git/cmd/git.exe")))[::-1]
    for g in cands:
        try:
            rev = subprocess.run([g, "rev-parse", "HEAD"], cwd=C.CODE, capture_output=True, text=True, check=True).stdout.strip()
            dirty = subprocess.run([g, "status", "--porcelain"], cwd=C.CODE, capture_output=True, text=True, check=True).stdout.strip()
            return {"commit": rev, "dirty": bool(dirty)}
        except (OSError, subprocess.CalledProcessError):
            continue
    return {"commit": None, "dirty": None}


def str_col(df, c):
    return df[c].astype("string").fillna("").to_numpy(object)


# ---------------------------------------------------------------- S1
def read(path: Path) -> dict:
    df = pd.read_csv(path, low_memory=False)
    ts = pd.to_numeric(df["ts_SQ"], errors="coerce").to_numpy(float)
    if not np.all(np.diff(ts) == C.TS_STEP):
        raise ValueError(f"{path.name}: ts_SQ is not a regular grid of step {C.TS_STEP}")
    cols = list(df.columns)
    blocks = {}
    for b in C.BLOCK_ORDER:
        tcol, xyz, _ = C.BLOCKS[b]
        i = cols.index(tcol)
        if cols[i + 1:i + 4] != xyz:
            raise ValueError(f"{path.name}: unexpected columns after {tcol}: {cols[i + 1:i + 4]}")
        x = df[xyz].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        ok = np.isfinite(x).all(1) & ~(x == 0).all(1)
        nu = max((len(np.unique(x[ok, k])) for k in range(3)), default=0) if ok.any() else 0
        if nu < C.MIN_UNIQUE:
            ok[:] = False                                   # block without signal (user1, 2, 4 SB)
        blocks[b] = (x, ok)
    act = str_col(df, C.ACT_COL)
    bad = set(np.unique(act)) - KNOWN_ACTIVITE
    if bad:
        raise ValueError(f"{path.name}: unknown activite values {bad}")
    return dict(ts=ts.astype(np.int64), blocks=blocks, fam={c: str_col(df, c) for c in C.FAMILY_COLS},
                cat=str_col(df, C.CAT_COL), act=act, n=len(df))


# ---------------------------------------------------------------- one participant
def process(user: str, raw: dict, sens_off: dict, lab_off: dict, long_runs: list, vindex: dict):
    n = raw["n"]
    stats = {"participant": user}

    # S2 sensor alignment
    x50, avail = [], []
    stats["sensors"] = {}
    for b in C.BLOCK_ORDER:
        dev = C.BLOCKS[b][2]
        plan = sync.device_plan(sens_off, user, dev)
        x, a = sync.align_block(*raw["blocks"][b], plan)
        x50.append(x); avail.append(a)
        stats["sensors"][b] = dict(device=dev, status=plan["status"], shift_s=plan["shift_s"] if plan["usable"] else None)

    # S4 anomaly masks (CSV label time), applied to the family columns
    fam_mask, act_mask = anomalies.masks(long_runs, user, n, raw["act"])
    fam = {c: np.where(fam_mask[c], "", raw["fam"][c]).astype(object) for c in C.FAMILY_COLS}

    # S3 label alignment (labels and masks move together)
    d = alignment.shift_s(lab_off, user)
    fam = {c: alignment.shift_labels(v, d) for c, v in fam.items()}
    act = alignment.shift_labels(raw["act"], d)
    act_mask = alignment.shift_labels(act_mask, d, False)
    fam_mask = {cc: alignment.shift_labels(m, d, False) for cc, m in fam_mask.items()}
    body_mask = fam_mask[C.BODY_COL]
    any_mask = act_mask | np.any(np.stack(list(fam_mask.values())), axis=0)
    stats["label_shift_s"] = d

    # S5 targets at 50 Hz
    ann = annotated_mask(fam)
    l2s = encode_l2(act)
    l2s[act_mask] = C.IGNORE
    post = posture_states(fam[C.BODY_COL], ann)
    s_all = mask_posture_after(post.state_all, body_mask, post.transitions)
    s_ann = s_all.copy(); s_ann[~ann] = C.IGNORE
    w = walk_targets(fam[C.LEG_COL])
    t = transition_targets(fam[C.BODY_COL])

    # S6 step grid
    T = n // C.STEP
    L = T * C.STEP
    c = step_centres(n)
    l2 = l2s[c]
    arrays = {
        "ts": raw["ts"][:L],
        "x50": np.concatenate(x50, axis=1)[:L].astype(np.float32),          # 5 blocks x 3 axes, NaN = unavailable
        "avail50": np.stack(avail, axis=1)[:L],                               # (L, 5) bool
        "l2": l2.astype(np.int8), "l1": l1_from_l2(l2).astype(np.int8),
        "s": s_all[c].astype(np.int8), "s_annotated": s_ann[c].astype(np.int8),
        "w": w[c].astype(np.int8), "t": t[c].astype(np.int8),
        "annotated": ann[c], "masked_anomaly": any_mask[c],
        "fam": np.stack([[vindex[v] for v in fam[col][c]] for col in C.FAMILY_COLS], axis=1).astype(np.int16),
        "l2_raw": encode_l2(raw["act"])[c].astype(np.int8),                   # activite as in the CSV (no shift, no mask)
    }

    # statistics (seconds at 50 Hz on the label rows, steps for the 0.1-s grid)
    lab = np.flatnonzero(ann)
    region = slice(lab[0], lab[-1] + 1)
    sec = lambda m: float(np.sum(m) / C.FS)
    stats.update({
        "samples": n, "steps": T, "duration_s": n / C.FS,
        "labelled_region_s": (region.stop - region.start) / C.FS, "annotated_s": sec(ann),
        "availability_in_labelled_region_pct": {b: float(100 * avail[i][region].mean()) for i, b in enumerate(C.BLOCK_ORDER)},
        "l2_s": {k: sec(l2s == i) for k, i in L2_INDEX.items()},
        "l2_supervised_s": sec(l2s != C.IGNORE),
        "activite_masked_s": sec(act_mask), "family_masked_s": {cc: sec(fam_mask[cc]) for cc in C.FAMILY_COLS},
        "s_s": {C.S_CLASSES[i]: sec(s_all == i) for i in range(len(C.S_CLASSES))},
        "s_supervised_s": sec(s_all != C.IGNORE),
        "w_s": {"walk": sec(w == 1), "not_walk": sec(w == 0)}, "t_s": {"transition": sec(t == 1), "no_transition": sec(t == 0)},
        "posture_pairs": {"chain": post.consistent_pairs, "break": post.inconsistent_pairs, "with_autre_transition": post.pairs_with_other},
        "content_sha256": content_sha256(arrays),
    })
    return arrays, stats


# ---------------------------------------------------------------- data card
def data_card(man: dict) -> str:
    P = man["participants"]
    users = [p["participant"] for p in P]
    L = ["# Data card — processed data v2", "",
         f"Built {man['created']} from git commit `{man['git']['commit']}`" + (" (uncommitted changes!)" if man["git"]["dirty"] else "") + ".",
         "Plan: reports/design/2026-11-17_pipeline_v2_plan.md. Seconds are counted on 50-Hz rows.", "",
         "## Per participant", "",
         "| | " + " | ".join(users) + " |", "|---|" + "---|" * len(users)]
    row = lambda name, f: L.append(f"| {name} | " + " | ".join(f(p) for p in P) + " |")
    row("duration (s)", lambda p: f"{p['duration_s']:.0f}")
    row("annotated (s)", lambda p: f"{p['annotated_s']:.0f}")
    row("L2 supervised (s)", lambda p: f"{p['l2_supervised_s']:.0f}")
    row("activite masked, anomaly (s)", lambda p: f"{p['activite_masked_s']:.1f}")
    row("S supervised (s)", lambda p: f"{p['s_supervised_s']:.0f}")
    row("W walk / not (s)", lambda p: f"{p['w_s']['walk']:.0f} / {p['w_s']['not_walk']:.0f}")
    row("T transition / not (s)", lambda p: f"{p['t_s']['transition']:.0f} / {p['t_s']['no_transition']:.0f}")
    row("label shift (s)", lambda p: f"{p['label_shift_s']:+.2f}")
    for b in C.BLOCK_ORDER:
        row(f"{b}: shift (s) / available %", lambda p, b=b: (
            f"{p['sensors'][b]['shift_s']:+.2f} / {p['availability_in_labelled_region_pct'][b]:.0f}"
            if p["sensors"][b]["shift_s"] is not None else "not used"))
    L += ["", "Availability = share of the labelled region where the block has a valid, aligned sample.", "",
          "## L2 classes (seconds) — main target `activite`", "",
          "| class | " + " | ".join(users) + " | total | participants |", "|---|" + "---|" * (len(users) + 2)]
    for k in C.L2_CLASSES:
        v = [p["l2_s"][k] for p in P]
        L.append(f"| {k} | " + " | ".join(f"{x:.1f}" for x in v) + f" | {sum(v):.1f} | {sum(x > 0 for x in v)} |")
    L += ["", "## Posture S (seconds)", "", "| state | " + " | ".join(users) + " | total |", "|---|" + "---|" * (len(users) + 1)]
    for s in C.S_CLASSES:
        v = [p["s_s"][s] for p in P]
        L.append(f"| {s} | " + " | ".join(f"{x:.0f}" for x in v) + f" | {sum(v):.0f} |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.PROJECT / "data" / "processed" / "v2"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sens_off, lab_off, long_runs = sync.load_offsets(), alignment.load_offsets(), anomalies.load_runs()
    files = {p: next(C.RAW_2022.glob(f"{p}-*.csv")) for p in PARTICIPANTS}
    raws = {p: read(files[p]) for p in PARTICIPANTS}
    vocab = sorted({v for r in raws.values() for c in C.FAMILY_COLS for v in r["fam"][c]})
    vindex = {v: i for i, v in enumerate(vocab)}

    man = {"created": datetime.now().isoformat(timespec="seconds"), "pipeline": "v2", "code_version": __version__,
           "git": git_state(), "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
           "raw": {p: {"file": files[p].name, "sha256": sha256_file(files[p])} for p in PARTICIPANTS},
           "params": {f.name: sha256_file(f) for f in sorted(C.PARAMS.glob("*.json"))},
           "blocks": C.BLOCK_ORDER, "block_columns": {b: C.BLOCKS[b][1] for b in C.BLOCK_ORDER},
           "family_cols": C.FAMILY_COLS, "family_vocab": vocab,
           "l2_classes": C.L2_CLASSES, "l1_classes": C.L1_CLASSES, "s_classes": C.S_CLASSES,
           "step_samples": C.STEP, "ignore": C.IGNORE, "participants": []}
    for p in PARTICIPANTS:
        arrays, stats = process(p, raws[p], sens_off, lab_off, long_runs, vindex)
        np.savez_compressed(out / f"{p}.npz", **arrays)
        man["participants"].append(stats)
        print(f"{p}: {stats['steps']} steps; L2 {stats['l2_supervised_s']:.0f} s; S {stats['s_supervised_s']:.0f} s; "
              f"masked activite {stats['activite_masked_s']:.1f} s; sha {stats['content_sha256'][:12]}")
    (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "data_card.md").write_text(data_card(man), encoding="utf-8")
    print(f"written {out}")


if __name__ == "__main__":
    main()
