"""Train and evaluate one neural run (one backbone, model, fold, seed, label fraction, lambda set).

Reference: reports/design/2026-11-17_model_architecture.md §4–§6; formulas and choices approved
2026-11-18 (Explanation.md §1–7).

Models (backbone given separately, TCN or DeepConvLSTM):
  B1   heads L2, L1          loss = CE
  B2   heads L2, L1, S, W, T loss = CE                       (B1 + auxiliary heads)
  NS1  as B2                 loss = CE + lam_hier * A1
  NS2  as B2                 loss = CE + lam_hier * A1 + lam_hard * (A2 + A4..A7) + lam_soft * A3,
                             rule losses on labelled steps only
  NS3  as NS2                rule losses on every step (also unlabelled ones)
CE: softmax heads (L2, L1, S) with class weights 1/sqrt(frequency) on the training labels (Cui et
al. 2019 p.2); binary heads (W, T) without weights. Each head skips its own unlabelled steps.

Training: random 60-s crops, modality dropout p = 0.2, AdamW; early stopping on the macro-F1 of L2
on the validation participant (configuration ALL). Test: the full recording of the test participant
under each sensor configuration ALL / PHONE / HB.

Example:
    python -m emob.train --model NS2 --backbone tcn --fold 0 --seed 0 --lam-hier 0.3 --lam-hard 1 --lam-soft 0.3
Writes <out>/<run_id>/metrics.json, test_logits.npz, val_logits.npz, model.pt.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from emob import __version__, config as C
from emob.data.dataset import build_fold, class_weights, eval_config, modality_dropout, sample_crops
from emob.data.splits import loso_folds
from emob.evaluate import decide, evaluate
from emob.metrics import confusion, macro_f1
from emob.models.deepconvlstm import DeepConvLSTM
from emob.models.heads import ALL_HEADS, B1_HEADS
from emob.models.tcn import TCN
from emob.rules_loss import RULES, rule_losses, weighted_sum

MODELS = {
    "B1": {"heads": B1_HEADS, "rules": []},
    "B2": {"heads": ALL_HEADS, "rules": []},
    "NS1": {"heads": ALL_HEADS, "rules": ["A1"]},
    "NS2": {"heads": ALL_HEADS, "rules": RULES},
    "NS3": {"heads": ALL_HEADS, "rules": RULES},
}
BACKBONES = {"tcn": "TCN", "lstm": "LSTM"}
WEIGHTED_HEADS = {"l2", "l1", "s"}          # class weights (approved 2026-11-18)


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--backbone", choices=list(BACKBONES), default="tcn")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--lam-hier", type=float, default=0.0)
    ap.add_argument("--lam-hard", type=float, default=0.0)
    ap.add_argument("--lam-soft", type=float, default=0.0)
    ap.add_argument("--no-rule", action="append", default=[], choices=RULES,
                    help="leave-one-rule-out: drop this rule (repeatable)")
    ap.add_argument("--rep", choices=["invariant", "raw"], default="invariant")
    ap.add_argument("--s-variant", choices=["s", "s_annotated"], default="s")
    ap.add_argument("--out", default="results/phase1")
    ap.add_argument("--crop-steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--moddrop", type=float, default=0.2)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    rules = [r for r in MODELS[a.model]["rules"] if r not in a.no_rule]
    if a.model in ("B1", "B2") and (a.lam_hier or a.lam_hard or a.lam_soft):
        ap.error(f"{a.model} has no rule loss; do not pass lambdas")
    if a.model == "NS1" and (a.lam_hard or a.lam_soft):
        ap.error("NS1 uses lam_hier only")
    a.rules = rules
    return a


def run_id(a) -> str:
    rid = f"{BACKBONES[a.backbone]}-{a.model}_frac{a.fraction:g}_fold{a.fold}_seed{a.seed}"
    if a.model == "NS1":
        rid += f"_lam{a.lam_hier:g}"
    elif a.model in ("NS2", "NS3"):
        rid += f"_lam{a.lam_hier:g}-{a.lam_hard:g}-{a.lam_soft:g}"
    if a.no_rule:
        rid += "_no" + "".join(sorted(a.no_rule))
    if a.rep != "invariant":
        rid += f"_{a.rep}"
    if a.s_variant != "s":
        rid += f"_{a.s_variant}"
    return rid


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=C.CODE, capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def make_model(a, in_ch: int):
    heads = MODELS[a.model]["heads"]
    if a.backbone == "tcn":
        return TCN(in_ch=in_ch, heads=heads)
    return DeepConvLSTM(in_ch=in_ch, heads=heads)


@torch.no_grad()
def predict(model, seq, device) -> dict[str, np.ndarray]:
    """Full recording -> {head: (K, T) logits}."""
    model.eval()
    x = torch.from_numpy(seq.x).unsqueeze(0).to(device)
    return {k: v[0].float().cpu().numpy() for k, v in model(x).items()}


def supervised_loss(logits, y, weights) -> tuple[torch.Tensor, dict[str, float]]:
    total, parts = 0.0, {}
    for h, z in logits.items():
        t = y[h]
        known = t != C.IGNORE
        if not known.any():
            continue
        if h in C.BINARY_HEADS:
            ce = F.binary_cross_entropy_with_logits(z[:, 0][known], t[known].float())
        else:
            ce = F.cross_entropy(z, t, weight=weights.get(h), ignore_index=C.IGNORE)
        total = total + ce
        parts[h] = float(ce)
    return total, parts


def main(argv=None):
    a = parse(argv)
    out = Path(a.out).expanduser() / run_id(a)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    fold = loso_folds()[a.fold]
    data = build_fold(fold["train"], fold["val"], fold["test"], a.fraction, a.seed, a.rep, a.s_variant)
    channels = data["norm"]["channels"]
    in_ch = data["train"][0].x.shape[1]
    heads = MODELS[a.model]["heads"]
    weights = {h: torch.tensor(class_weights(data["train"], h, C.HEADS[h]), dtype=torch.float32, device=a.device)
               for h in heads if h in WEIGHTED_HEADS}
    lam = {"hier": a.lam_hier, "hard": a.lam_hard, "soft": a.lam_soft}
    rules_on = [r for r in a.rules if lam[{"A1": "hier", "A3": "soft"}.get(r, "hard")] > 0]

    model = make_model(a, in_ch).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    total_steps = sum(len(s.y["l2"]) for s in data["train"])
    crops_per_epoch = math.ceil(total_steps / a.crop_steps)

    val = data["val"][0]
    best, best_epoch, best_state, history = -1.0, -1, None, []
    t_start = time.time()
    for epoch in range(a.max_epochs):
        model.train()
        x, ys = sample_crops(data["train"], crops_per_epoch, a.crop_steps, rng)
        x = modality_dropout(x, channels, a.moddrop, rng)
        logs = {"train_loss": [], "train_ce": [], **{r: [] for r in rules_on}}
        for i in range(0, crops_per_epoch, a.batch):
            xb = torch.from_numpy(x[i:i + a.batch]).to(a.device)
            yb = {h: torch.from_numpy(ys[h][i:i + a.batch]).to(a.device) for h in heads}
            logits = model(xb)
            loss, _ = supervised_loss(logits, yb, weights)
            if not isinstance(loss, float):
                logs["train_ce"].append(float(loss))
            if rules_on:
                mask = (torch.ones_like(yb["l2"], dtype=torch.bool) if a.model == "NS3"
                        else yb["l2"] != C.IGNORE)
                rl = rule_losses(logits, mask, rules_on)
                loss = loss + weighted_sum(rl, lam)
                for r, v in rl.items():
                    logs[r].append(float(v))
            if isinstance(loss, float):          # crop without any label (possible at low fractions)
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            logs["train_loss"].append(float(loss))
        pv = decide(predict(model, val, a.device))
        f1 = macro_f1(confusion(val.y["l2"], pv["l2"], C.HEADS["l2"]))
        f1 = f1 if np.isfinite(f1) else 0.0
        history.append({"epoch": epoch, "val_l2_macro_f1": f1,
                        **{k: (float(np.mean(v)) if v else None) for k, v in logs.items()}})
        if f1 > best:
            best, best_epoch = f1, epoch
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        elif epoch - best_epoch >= a.patience:
            break
    train_time = time.time() - t_start

    model.load_state_dict(best_state)
    val_logits = predict(model, val, a.device)
    val_metrics = evaluate(decide(val_logits), val.y)
    te = data["test"][0]
    test, test_logits = {}, {}
    for cfg in C.EVAL_CONFIGS:
        seq = eval_config(te, cfg)
        if seq is None:
            continue
        lg = predict(model, seq, a.device)
        test[cfg] = evaluate(decide(lg), te.y)
        test_logits.update({f"{cfg}_{h}": v.astype(np.float16) for h, v in lg.items()})

    np.savez_compressed(out / "test_logits.npz", **test_logits)
    np.savez_compressed(out / "val_logits.npz", **{h: v.astype(np.float16) for h, v in val_logits.items()})
    torch.save(best_state, out / "model.pt")
    result = {
        "run_id": run_id(a), "code_version": __version__, "git_commit": git_commit(),
        "args": {k: v for k, v in vars(a).items()}, "fold": fold, "heads": heads, "rules_on": rules_on,
        "n_params": n_params, "crops_per_epoch": crops_per_epoch, "best_epoch": best_epoch,
        "epochs_run": len(history), "val_l2_macro_f1": best, "val": val_metrics,
        "train_time_s": train_time, "normalisation": data["norm"],
        "class_weights": {h: w.cpu().tolist() for h, w in weights.items()},
        "test": test, "history": history,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    summary = ", ".join(f"{c} {m['l2_macro_f1']:.3f}" for c, m in test.items())
    print(f"{run_id(a)}: val {best:.3f} (epoch {best_epoch}), test L2 {summary}, time {train_time:.0f}s")


if __name__ == "__main__":
    main()
