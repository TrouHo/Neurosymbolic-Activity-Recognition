"""Do the rule losses change the model? TCN-B2 vs TCN-NS2 on training and test participants.

Question (2026-11-18, after phase 2b): NS2 equals B2 on L2 at 100 % labels. Two explanations:
  (a) on the training participants the rules are already satisfied once CE fits the labels, so the
      rule losses add almost no gradient; the violations happen on the unseen test participant;
  (b) the rule losses are too weak (averaged over every step, lambdas <= 1) to move the model.
Check: for each fold (seed 0), load the saved weights of TCN-B2 and of TCN-NS2 with the lambdas
selected on validation, predict the full recordings of the six training participants and of the
test participant (configuration ALL), and report
  - the soft rule losses on labelled steps (exactly the training objective, emob.rules_loss);
  - the arg-max violation rates (emob.metrics.rule_violations).
If NS2 lowers the training rule losses clearly below B2 but not the test violations -> (a) / a
generalisation gap; if NS2 does not lower the training rule losses either -> (b).

Read-only (needs torch; run on the training server).
Usage (from code/):  PYTHONPATH=. python diagnostics/rule_loss_train_vs_test.py --results ~/emob/results/v2 \
                     --summary ~/emob/results/v2/summary_phase2b.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from emob import config as C
from emob.data.dataset import build_fold
from emob.data.splits import loso_folds
from emob.evaluate import decide
from emob.metrics import rule_violations
from emob.models.heads import ALL_HEADS
from emob.models.tcn import TCN
from emob.rules_loss import RULES, rule_losses

SHOW = ["A1", "A2", "A3", "A6"]          # rules with room to act (phase 1: A4, A5, A7 about 0 %)


def load(run_dir: Path, device) -> TCN:
    m = TCN(in_ch=17, heads=ALL_HEADS).to(device)
    m.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    return m.eval()


@torch.no_grad()
def measure(model, seqs, device) -> dict:
    soft = {r: [] for r in RULES}
    viol = {r: [0, 0] for r in RULES}
    for s in seqs:
        logits = model(torch.from_numpy(s.x).unsqueeze(0).to(device))
        mask = torch.from_numpy(s.y["l2"] != C.IGNORE).unsqueeze(0).to(device)
        for r, v in rule_losses(logits, mask).items():
            soft[r].append(float(v))
        pred = decide({k: v[0].float().cpu().numpy() for k, v in logits.items()})
        for r, v in rule_violations(**pred).items():
            viol[r][0] += v["viol"]
            viol[r][1] += v["n"]
    return {"soft": {r: float(np.mean(v)) for r, v in soft.items()},
            "viol": {r: (v[0] / v[1] if v[1] else None) for r, v in viol.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.results).expanduser()
    lam = json.loads(Path(a.summary).expanduser().read_text())["selected_lambda"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for f in loso_folds():
        k = f["fold"]
        h, d, s = lam[f"TCN-NS2|frac1|fold{k}"]
        runs = {"B2": root / f"TCN-B2_frac1_fold{k}_seed{a.seed}",
                "NS2": root / f"TCN-NS2_frac1_fold{k}_seed{a.seed}_lam{h:g}-{d:g}-{s:g}"}
        data = build_fold(f["train"], f["val"], f["test"])
        out[k] = {}
        for name, rd in runs.items():
            m = load(rd, device)
            out[k][name] = {"train": measure(m, data["train"], device), "test": measure(m, data["test"], device)}
        print(f"fold {k} (test {f['test']}, NS2 lambdas {h:g}/{d:g}/{s:g})")
        for part in ("train", "test"):
            for name in ("B2", "NS2"):
                r = out[k][name][part]
                print(f"  {part:5s} {name:3s} soft " + " ".join(f"{x} {r['soft'][x]:.4f}" for x in SHOW)
                      + " | viol " + " ".join(f"{x} {100 * r['viol'][x]:.1f}%" for x in SHOW if r['viol'][x] is not None))
    print("\nMean over folds (soft loss on labelled steps; arg-max violation %):")
    for part in ("train", "test"):
        for name in ("B2", "NS2"):
            soft = {x: np.mean([out[k][name][part]["soft"][x] for k in out]) for x in SHOW}
            viol = {x: np.mean([out[k][name][part]["viol"][x] for k in out if out[k][name][part]["viol"][x] is not None]) for x in SHOW}
            print(f"  {part:5s} {name:3s} soft " + " ".join(f"{x} {v:.4f}" for x, v in soft.items())
                  + " | viol " + " ".join(f"{x} {100 * v:.1f}%" for x, v in viol.items()))
    Path(root / "diag_rule_loss_train_vs_test.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
