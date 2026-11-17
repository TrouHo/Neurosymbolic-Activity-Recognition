"""Aggregate metrics.json files into tables (mean ± SD over folds x seeds, and per fold).

    python -m emob.summarize --results ~/emob/results/phase1 --out summary.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

KEYS = ["l2_macro_f1", "l2_macro_f1_no_autre", "l1_macro_f1", "S_macro_f1_vs_s_all", "S_macro_f1_vs_s"]
SEG = ["edit", "f1@10", "f1@25", "f1@50"]
RULES = ["A1", "A2", "A3", "A4"]


def collect(root: Path) -> list[dict]:
    rows = []
    for f in sorted(root.glob("*/metrics.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        a, t = m["args"], m["test"]
        model = m["run_id"].split("_")[0]
        row = {"model": model, "inputs": a.get("inputs", "MP"), "fraction": a["fraction"],
               "s_variant": a["s_variant"], "fold": a["fold"], "seed": a["seed"],
               "test": m["fold"]["test"], "train_time_s": m.get("train_time_s"),
               "epochs": m.get("epochs_run"), "best_epoch": m.get("best_epoch")}
        for k in KEYS:
            row[k] = t.get(k)
        for k in SEG:
            row["seg_" + k] = t["l2_segments"][k]
        row["n_seg_pred"] = t["l2_segments"]["n_seg_pred"]
        row["n_seg_gt"] = t["l2_segments"]["n_seg_gt"]
        for k in RULES:
            row["viol_" + k] = t["rule_violations_pred"].get(k)
        row["l2_per_class_f1"] = t["l2_per_class_f1"]
        rows.append(row)
    return rows


def summarise(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["inputs"], r["fraction"], r["s_variant"])].append(r)
    out = {}
    for key, rs in sorted(groups.items()):
        name = "{}|{}|frac{}|{}".format(*key)
        g = {"n_runs": len(rs)}
        numeric = [k for k in rs[0] if isinstance(rs[0][k], (int, float)) and k not in ("fold", "seed", "fraction")]
        for k in numeric:
            v = np.array([r[k] for r in rs if r[k] is not None], float)
            if len(v):
                g[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
        # per test participant (mean over seeds)
        per = defaultdict(list)
        for r in rs:
            per[r["test"]].append(r["l2_macro_f1"])
        g["l2_macro_f1_per_participant"] = {p: float(np.mean(v)) for p, v in sorted(per.items())}
        # per-class F1 (mean over runs where defined)
        cls = defaultdict(list)
        for r in rs:
            for c, f in r["l2_per_class_f1"].items():
                if f is not None:
                    cls[c].append(f)
        g["l2_per_class_f1_mean"] = {c: float(np.mean(v)) for c, v in cls.items()}
        out[name] = g
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = collect(Path(a.results).expanduser())
    s = summarise(rows)
    Path(a.out).expanduser().write_text(json.dumps({"rows": rows, "summary": s}, indent=1), encoding="utf-8")
    print(f"{len(rows)} runs")
    for name, g in s.items():
        f = lambda k: f"{g[k]['mean']:.3f}±{g[k]['sd']:.3f}" if k in g else "  -  "
        print(f"{name:32s} n={g['n_runs']:2d}  L2 {f('l2_macro_f1')}  L2-noAutre {f('l2_macro_f1_no_autre')}  "
              f"L1 {f('l1_macro_f1')}  S(s_all) {f('S_macro_f1_vs_s_all')}  edit {f('seg_edit')}  "
              f"time {g['train_time_s']['mean']:.0f}s")


if __name__ == "__main__":
    main()
