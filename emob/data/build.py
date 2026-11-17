"""Phase 0: build one .npz per 2022 participant and a gate-G0 report.

Usage (from D:\\Projects\\EmoB\\code):
    .venv\\Scripts\\python.exe -m emob.data.build
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from emob import __version__, config as C
from emob.data.io import load_recording
from emob.data.labels import (annotated_mask, centre_vs_majority, encode_l2, l1_from_l2,
                              posture_states, step_centres)
from emob.data.rules import (hierarchy_agreement, last_valid_state_check, lookback_check,
                             walk_segments, walking_state_counts)
from emob.data.splits import PARTICIPANTS

DELTAS = [0.1, 0.2, 0.3, 0.5, 1.0]


def family_vocab(recs) -> list[str]:
    vocab = set()
    for r in recs:
        for c in C.FAMILY_COLS:
            vocab |= set(r.families[c])
    return sorted(vocab)                               # "" is index 0


def process(rec, vocab_index):
    ann = annotated_mask(rec.families)
    post = posture_states(rec.families[C.BODY_COL], ann)
    l2_samples = encode_l2(rec.activite)
    centres = step_centres(rec.n)

    l2 = l2_samples[centres]
    arrays = {
        "ts": rec.ts.astype(np.int64),
        "acc50": rec.acc.astype(np.float32),
        "usable50": rec.usable,
        "phone50": rec.extra.astype(np.float32),          # phone ACC xyz, phone GYR xyz
        "phone_usable50": rec.extra_usable,
        "l2": l2.astype(np.int8),
        "l1": l1_from_l2(l2).astype(np.int8),
        "s": post.state[centres].astype(np.int8),          # masked on unannotated rows
        "s_all": post.state_all[centres].astype(np.int8),  # not masked on unannotated rows
        "annotated": ann[centres],
        "fam": np.stack([[vocab_index[v] for v in rec.families[c][centres]] for c in C.FAMILY_COLS],
                        axis=1).astype(np.int16),
    }

    # --- statistics for the G0 report ------------------------------------------------
    lab = np.flatnonzero(ann)
    region = slice(lab[0], lab[-1] + 1) if len(lab) else slice(0, 0)
    n_hier, agree = hierarchy_agreement(rec.activite, rec.categorie, rec.families)
    n_walk, walk_ok = walk_segments(rec.families[C.LEG_COL])
    n_sup, differ = centre_vs_majority(l2_samples)
    stats = {
        "participant": rec.participant, "source": rec.source, "sha256": rec.sha256,
        "samples": rec.n, "steps": len(centres),
        "duration_s": rec.n / C.FS,
        "labelled_region_s": (region.stop - region.start) / C.FS,
        "unusable_in_labelled_region_pct": {
            **{k: float(100 * (~rec.usable[region, i]).mean()) for i, k in enumerate(C.SENSOR_ORDER)},
            **{k: float(100 * (~rec.extra_usable[region, i]).mean()) for i, k in enumerate(C.EXTRA_ORDER)}},
        "degenerate_sensor": rec.degenerate,
        "max_unique_values_per_axis": rec.n_unique,
        "annotated_rows_in_region_pct": float(100 * ann[region].mean()),
        "steps_with_l2": int((l2 != C.IGNORE).sum()),
        "steps_with_s": int((arrays["s"] != C.IGNORE).sum()),
        "steps_annotated": int(arrays["annotated"].sum()),
        "l2_step_counts": {C.L2_CLASSES[i]: int((l2 == i).sum()) for i in range(len(C.L2_CLASSES))},
        "s_step_counts": {C.S_CLASSES[i]: int((arrays["s"] == i).sum()) for i in range(len(C.S_CLASSES))},
        "steps_with_s_all": int((arrays["s_all"] != C.IGNORE).sum()),
        "s_all_step_counts": {C.S_CLASSES[i]: int((arrays["s_all"] == i).sum()) for i in range(len(C.S_CLASSES))},
        "hierarchy_rows": n_hier, "hierarchy_agree": agree,
        "sm_consistent_pairs": post.consistent_pairs, "sm_inconsistent_pairs": post.inconsistent_pairs,
        "sm_pairs_with_other": post.pairs_with_other,
        "walk_segments": n_walk, "walk_segments_ge_1s": walk_ok,
        **walking_state_counts(rec.families[C.LEG_COL], post.state),
        "walking_rows_sitting_or_lying_s_all":
            walking_state_counts(rec.families[C.LEG_COL], post.state_all)["walking_rows_sitting_or_lying"],
        "centre_supervised_steps": n_sup, "centre_differs_from_majority": differ,
        "lookback": {str(k): v for k, v in lookback_check(post.transitions, post.state, DELTAS).items()},
        "last_valid_state": last_valid_state_check(post.transitions, post.state),
    }
    return arrays, stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(C.RAW_2022))
    ap.add_argument("--out", default=str(C.PROCESSED))
    args = ap.parse_args(argv)
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = {p: next(raw.glob(f"{p}-*.csv")) for p in PARTICIPANTS}
    recs = [load_recording(files[p]) for p in PARTICIPANTS]
    vocab = family_vocab(recs)
    vindex = {v: i for i, v in enumerate(vocab)}

    report = {"created": datetime.now().isoformat(timespec="seconds"), "code_version": __version__,
              "family_vocab": vocab, "l2_classes": C.L2_CLASSES, "l1_classes": C.L1_CLASSES,
              "s_classes": C.S_CLASSES, "participants": []}
    for rec in recs:
        arrays, stats = process(rec, vindex)
        np.savez_compressed(out / f"{rec.participant}.npz", **arrays)
        report["participants"].append(stats)
        print(f"{rec.participant}: {stats['steps']} steps, L2 on {stats['steps_with_l2']}, S on {stats['steps_with_s']}")

    tot = lambda k: sum(p[k] for p in report["participants"])
    report["totals"] = {k: tot(k) for k in [
        "steps", "steps_with_l2", "steps_with_s", "steps_with_s_all", "hierarchy_rows", "hierarchy_agree",
        "sm_consistent_pairs", "sm_inconsistent_pairs", "sm_pairs_with_other",
        "walk_segments", "walk_segments_ge_1s", "walking_rows", "walking_rows_known_state",
        "walking_rows_sitting_or_lying", "walking_rows_sitting_or_lying_s_all", "centre_supervised_steps", "centre_differs_from_majority"]}
    (out / "phase0_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["totals"], indent=1))


if __name__ == "__main__":
    main()
