"""Collect metrics.json files, select lambdas on validation, and compare models.

Reference: Explanation.md §1 (lambda selection) and §7 (paired comparison); Demšar 2006 (JMLR 7).

1. collect(): one row per run and per sensor configuration (ALL / PHONE / HB).
2. select_lambda(): for each (backbone, NS model, fraction, fold), the lambda set with the highest
   validation macro-F1 of L2, averaged over seeds; ties -> the smaller lambdas. Uses validation only.
   Selection is a separate step on saved files: changing the rule needs no retraining.
   NS3 is not selected: it inherits NS2's selected lambdas (INHERIT_LAMBDA).
3. compare(): paired comparison on the test participants. Unit = test participant, score = mean
   over seeds (seeds of one participant are not independent). Reported: mean and SD of the
   difference, number of participants improved, exact two-sided Wilcoxon signed-rank p, and
   Holm-adjusted p over the comparisons of one (configuration, fraction).

    python -m emob.summarize --results ~/emob/results --out ~/emob/results/summary.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from emob.metrics import macro_f1
from emob.rules_spec import TRANSITION_IDX

SCORES = ["l2_macro_f1", "l2_macro_f1_no_autre", "l2_macro_f1_transitions", "l1_macro_f1", "s_macro_f1",
          "w_f1", "t_f1"]
SEG = ["edit", "f1@10", "f1@25", "f1@50"]
RULES = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
MAIN = "l2_macro_f1"
# Secondary metrics, declared after phases 1-2 and BEFORE phase 3 (label fractions, NS3) and NS4
# (decision 2026-11-18). All are reported, favourable or not; none replaces the main metric.
# Rationale: the rules act on posture transitions and on segment structure, which the 20-class
# frame macro-F1 (dominated by 13 classes with F1 <= 0.2 and a few seconds per test participant)
# hardly reflects.
SECONDARY = ["l2_macro_f1_transitions", "seg_f1@25", "seg_edit", "l1_macro_f1"] + [f"viol_{r}" for r in RULES]
# Fixed lambdas for the fairness check "NS2 without selection" (middle of the grid, chosen a priori)
FIXED = {"NS1": (0.3, 0.0, 0.0), "NS2": (0.3, 0.3, 0.3), "NS3": (0.3, 0.3, 0.3)}
# lam_hard values outside the valid range (2026-11-18): at lam_hard = 10 the walking head collapses
# (trivial solution of A4-A6, W F1 0.72 -> about 0.12). Excluded from selection, kept in "lambda_grid".
EXCLUDED_LAM_HARD = {10.0}
COMPARISONS = [("B2", "B1"), ("NS1", "B2"), ("NS2", "B2"), ("NS3", "NS2"), ("NS2-fixed", "B2"),
               ("NS3-fixed", "NS2-fixed"), ("B1", "B0")]
# NS3 is not selected on its own (decision 2026-11-21, restoring the declared design): it uses the
# lambdas selected for NS2 in the same backbone, fold and fraction, so NS3 - NS2 differs only by the
# steps on which the rule losses are computed. Until 2026-11-21 NS3 was re-selected between NS2's
# lambdas and the fixed set (8/32 TCN configurations then used the fixed set).
INHERIT_LAMBDA = {"NS3": "NS2"}
# NS4 (decoding-time rules, emob.run_ns4): own Holm family, so adding it leaves the Holm-adjusted
# p-values of the phase 1-3 comparisons unchanged. Main: NS4 - B2.
COMPARISONS_NS4 = [("NS4", "B2"), ("NS2V", "NS2"), ("NS2V", "NS4")]


# ----------------------------------------------------------------------------- collect
def collect(root: Path) -> list[dict]:
    rows = []
    for f in sorted(root.glob("**/metrics.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        a = m["args"]
        backbone, model = m["run_id"].split("_")[0].split("-")
        base = {"run_id": m["run_id"], "backbone": backbone, "model": model,
                "fraction": float(a.get("fraction", 1.0)), "fold": int(a["fold"]), "seed": int(a["seed"]),
                "test_participant": m["fold"]["test"],
                "lam": (a.get("lam_hier", 0.0), a.get("lam_hard", 0.0), a.get("lam_soft", 0.0)),
                "variant": "_".join(sorted(a.get("no_rule", []))) + ("" if a.get("rep", "invariant") == "invariant" else a["rep"])
                           + ("" if a.get("s_variant", "s") == "s" else a["s_variant"]),
                "val_l2_macro_f1": m.get("val_l2_macro_f1"), "train_time_s": m.get("train_time_s"),
                "best_epoch": m.get("best_epoch")}
        for cfg, t in m["test"].items():
            r = dict(base, config=cfg)
            for k in SCORES:
                r[k] = t.get(k)
            if r["l2_macro_f1_transitions"] is None:          # runs evaluated before the key existed
                r["l2_macro_f1_transitions"] = macro_f1(np.array(t["l2_confusion"]), TRANSITION_IDX)
            for k in SEG:
                r["seg_" + k] = t["l2_segments"][k]
            for k in RULES:
                v = t["rules_pred"].get(k)
                r["viol_" + k] = v["rate"] if v else None
            r["l2_per_class_f1"] = t["l2_per_class_f1"]
            rows.append(r)
    return rows


# ----------------------------------------------------------------------------- lambda selection
def select_lambda(rows: list[dict], backbone: str, model: str, fold: int, fraction: float = 1.0,
                  variant: str = "") -> tuple | None:
    """Lambda set with the best mean validation L2 macro-F1 over seeds (configuration-independent)."""
    by = defaultdict(dict)
    for r in rows:
        if (r["backbone"], r["model"], r["fold"], r["fraction"], r["variant"]) == (backbone, model, fold, fraction, variant) \
                and r["lam"][1] not in EXCLUDED_LAM_HARD:
            by[tuple(r["lam"])][r["seed"]] = r["val_l2_macro_f1"]
    if not by:
        return None
    score = {lam: np.mean(list(s.values())) for lam, s in by.items()}
    return max(score, key=lambda lam: (round(score[lam], 12), [-x for x in lam]))


def selected_rows(rows: list[dict]) -> list[dict]:
    """Rows of the models as compared: B*, NS* with the selected lambda, NS*-fixed with FIXED."""
    out, cache = [], {}
    for r in rows:
        if r["variant"]:
            continue
        if r["model"] in ("B0", "B1", "B2"):
            out.append(r)
            continue
        if tuple(r["lam"]) == FIXED.get(r["model"]):
            out.append(dict(r, model=r["model"] + "-fixed"))
        key = (r["backbone"], INHERIT_LAMBDA.get(r["model"], r["model"]), r["fold"], r["fraction"])
        if key not in cache:
            cache[key] = select_lambda(rows, *key)
        if tuple(r["lam"]) == cache[key]:
            out.append(r)
    return out


# ----------------------------------------------------------------------------- statistics
def wilcoxon_exact(d: np.ndarray) -> float | None:
    """Exact two-sided Wilcoxon signed-rank p (Demšar 2006 §3.1.3). Zero differences are dropped;
    tied |d| get average ranks; p = share of the 2^n sign patterns whose R+ is at least as far from
    its mean n(n+1)/4 as the observed R+."""
    d = np.asarray([x for x in d if x != 0], float)
    n = len(d)
    if n == 0:
        return None
    order = np.argsort(np.abs(d))
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    for v in np.unique(np.abs(d)):                       # average ranks for ties
        idx = np.abs(d) == v
        ranks[idx] = ranks[idx].mean()
    centre = ranks.sum() / 2
    obs = abs(ranks[d > 0].sum() - centre)
    hits = sum(abs(sum(r for r, s in zip(ranks, signs) if s) - centre) >= obs - 1e-12
               for signs in itertools.product((0, 1), repeat=n))
    return hits / 2 ** n


def holm(pvals: list[float | None]) -> list[float | None]:
    """Holm step-down adjusted p-values (Demšar 2006 §3.2.2)."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    m = len(idx)
    adj = [None] * len(pvals)
    running = 0.0
    for k, i in enumerate(sorted(idx, key=lambda i: pvals[i])):
        running = max(running, min(1.0, (m - k) * pvals[i]))
        adj[i] = running
    return adj


def per_participant(rows: list[dict], metric: str) -> dict:
    """{(backbone, model, config, fraction): {participant: mean over seeds}}."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get(metric) is not None:
            acc[(r["backbone"], r["model"], r["config"], r["fraction"])][r["test_participant"]].append(r[metric])
    return {k: {p: float(np.mean(v)) for p, v in sorted(d.items())} for k, d in acc.items()}


def compare(rows: list[dict], metric: str = "l2_macro_f1", comparisons=None) -> list[dict]:
    comparisons = COMPARISONS if comparisons is None else comparisons
    lstm_block = comparisons is COMPARISONS
    pp = per_participant(rows, metric)
    out = []
    groups = sorted({(k[2], k[3]) for k in pp})
    for cfg, frac in groups:
        block = []
        backbones = sorted({k[0] for k in pp if k[0] != "RF"})
        for bb in backbones:
            for a, b in comparisons:
                ka = (bb if a != "B0" else "RF", a, cfg, frac)
                kb = (bb if b != "B0" else "RF", b, cfg, frac)
                if ka in pp and kb in pp:
                    block.append((f"{bb}-{a} - {kb[0]}-{b}", pp[ka], pp[kb]))
        for model in sorted({k[1] for k in pp}) if lstm_block else []:   # backbone dependence
            if ("LSTM", model, cfg, frac) in pp and ("TCN", model, cfg, frac) in pp:
                block.append((f"LSTM-{model} - TCN-{model}", pp[("LSTM", model, cfg, frac)], pp[("TCN", model, cfg, frac)]))
        res = []
        for name, A, B in block:
            common = sorted(set(A) & set(B))
            d = np.array([A[p] - B[p] for p in common])
            res.append({"comparison": name, "config": cfg, "fraction": frac, "metric": metric,
                        "n": len(common), "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                        "improved": int((d < 0).sum() if metric.startswith("viol_") else (d > 0).sum()), "per_participant": dict(zip(common, d.round(4).tolist())),
                        "wilcoxon_p": wilcoxon_exact(d)})
        for r, p in zip(res, holm([r["wilcoxon_p"] for r in res])):
            r["holm_p"] = p
        out += res
    return out


def table(rows: list[dict]) -> dict:
    """Mean +- SD over folds x seeds per (backbone, model, config, fraction)."""
    g = defaultdict(list)
    for r in rows:
        g[(r["backbone"], r["model"], r["config"], r["fraction"])].append(r)
    out = {}
    for key, rs in sorted(g.items()):
        e = {"n_runs": len(rs)}
        for k in SCORES + ["seg_" + s for s in SEG] + ["viol_" + r for r in RULES] + ["train_time_s"]:
            v = np.array([r[k] for r in rs if r.get(k) is not None], float)
            if len(v):
                e[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
        out["|".join(map(str, key))] = e
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, nargs="+", help="one or more result folders")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = [r for root in a.results for r in collect(Path(root).expanduser())]
    sel = selected_rows(rows)
    lam = {f"{r['backbone']}-{r['model']}|frac{r['fraction']:g}|fold{r['fold']}": list(r["lam"])
           for r in sel if r["model"].startswith("NS") and not r["model"].endswith("fixed")}
    res = {"n_runs": len({r["run_id"] for r in rows}), "selected_lambda": lam, "table": table(sel),
           "per_participant_l2": {"|".join(map(str, k)): v for k, v in per_participant(sel, "l2_macro_f1").items()},
           "comparisons": compare(sel, MAIN),
           "comparisons_secondary": {m: compare(sel, m) for m in SECONDARY},
           "comparisons_ns4": compare(sel, MAIN, COMPARISONS_NS4),
           "comparisons_ns4_secondary": {m: compare(sel, m, COMPARISONS_NS4) for m in SECONDARY},
           "variants": table([dict(r, model=f"{r['model']}[{r['variant']}]") for r in rows if r["variant"]]),
           "lambda_grid": table([dict(r, model=f"{r['model']}{list(r['lam'])}") for r in rows
                                 if r["model"].startswith("NS")])}
    Path(a.out).expanduser().write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"{res['n_runs']} runs")
    for k, e in res["table"].items():
        f = lambda m: f"{e[m]['mean']:.3f}±{e[m]['sd']:.3f}" if m in e else "  -  "
        print(f"{k:34s} n={e['n_runs']:3d}  L2 {f('l2_macro_f1')}  noAutre {f('l2_macro_f1_no_autre')}  "
              f"L1 {f('l1_macro_f1')}  S {f('s_macro_f1')}  W {f('w_f1')}  edit {f('seg_edit')}")
    print()
    for c in res["comparisons"]:
        p = lambda x: "-" if x is None else f"{x:.4f}"
        print(f"{c['config']:5s} frac{c['fraction']:<4g} {c['comparison']:28s} diff {c['mean_diff']:+.3f}±{c['sd_diff']:.3f}  "
              f"improved {c['improved']}/{c['n']}  p {p(c['wilcoxon_p'])}  Holm {p(c['holm_p'])}")
    if res["comparisons_ns4"]:
        print("\nNS4 (decoding-time rules; own Holm family):")
    for c in res["comparisons_ns4"]:
        p = lambda x: "-" if x is None else f"{x:.4f}"
        print(f"{c['config']:5s} frac{c['fraction']:<4g} {c['comparison']:28s} diff {c['mean_diff']:+.3f}±{c['sd_diff']:.3f}  "
              f"improved {c['improved']}/{c['n']}  p {p(c['wilcoxon_p'])}  Holm {p(c['holm_p'])}")
    print("\nSecondary metrics (config ALL; 'improved' = lower for violation rates):")
    for m, cs in list(res["comparisons_secondary"].items()) + list(res["comparisons_ns4_secondary"].items()):
        for c in cs:
            if c["config"] == "ALL" and "B0" not in c["comparison"]:
                print(f"{m:26s} frac{c['fraction']:<4g} {c['comparison']:28s} diff {c['mean_diff']:+.4f}  "
                      f"improved {c['improved']}/{c['n']}  p {c['wilcoxon_p'] if c['wilcoxon_p'] is None else round(c['wilcoxon_p'], 4)}")


if __name__ == "__main__":
    main()
