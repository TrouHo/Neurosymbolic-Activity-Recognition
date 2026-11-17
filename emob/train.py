"""Train and evaluate one neural run (one fold, one seed, one label fraction).

Example:
    python -m emob.train --model B1+S --fold 0 --seed 0 --fraction 1.0 --out results/phase1
All settings and the outcome are written to <out>/<run_id>/metrics.json.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from emob import __version__, config as C
from emob.data.dataset import build_fold, class_weights, modality_dropout, sample_crops
from emob.data.splits import loso_folds
from emob.evaluate import evaluate
from emob.metrics import confusion, macro_f1
from emob.models.tcn import TCN

MODELS = {"B1": {"use_s": False}, "B1+S": {"use_s": True}}


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--s-variant", choices=["s_all", "s"], default="s_all")
    ap.add_argument("--inputs", choices=["M", "MP"], default="MP")
    ap.add_argument("--out", default="results/phase1")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--crop-steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--moddrop", type=float, default=0.2)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args(argv)


def run_id(a) -> str:
    return f"{a.model}_{a.inputs}_frac{a.fraction:g}_{a.s_variant}_fold{a.fold}_seed{a.seed}"


@torch.no_grad()
def predict(model, seq, device) -> dict[str, np.ndarray]:
    model.eval()
    x = torch.from_numpy(seq.x).unsqueeze(0).to(device)
    return {k: v[0].float().cpu().numpy() for k, v in model(x).items()}   # (K, T) logits


def main(argv=None):
    a = parse(argv)
    out = Path(a.out) / run_id(a)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    fold = loso_folds()[a.fold]
    data = build_fold(fold["train"], fold["val"], fold["test"], a.s_variant, a.fraction, a.seed, a.inputs)
    in_ch = data["train"][0].x.shape[1]
    use_s = MODELS[a.model]["use_s"]
    heads = ["l1", "l2"] + (["s"] if use_s else [])
    ncls = {"l1": len(C.L1_CLASSES), "l2": len(C.L2_CLASSES), "s": len(C.S_CLASSES)}
    weights = {h: torch.tensor(class_weights(data["train"], h, ncls[h]), dtype=torch.float32, device=a.device)
               for h in heads}

    model = TCN(in_ch=in_ch, ch=a.channels, layers=a.layers, dropout=a.dropout, use_s=use_s).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    total_steps = sum(len(s.l2) for s in data["train"])
    crops_per_epoch = math.ceil(total_steps / a.crop_steps)

    best, best_epoch, best_state, history = -1.0, -1, None, []
    t_start = time.time()
    for epoch in range(a.max_epochs):
        model.train()
        x, l1, l2, s = sample_crops(data["train"], crops_per_epoch, a.crop_steps, rng)
        x = modality_dropout(x, a.moddrop, rng)
        labels = {"l1": l1, "l2": l2, "s": s}
        losses = []
        for i in range(0, crops_per_epoch, a.batch):
            xb = torch.from_numpy(x[i:i + a.batch]).to(a.device)
            logits = model(xb)
            loss = 0.0
            for h in heads:
                y = torch.from_numpy(labels[h][i:i + a.batch]).to(a.device)
                if (y != C.IGNORE).any():
                    loss = loss + F.cross_entropy(logits[h], y, weight=weights[h], ignore_index=C.IGNORE)
            if isinstance(loss, float):
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        # validation: L2 macro-F1 on the validation participant
        v = data["val"][0]
        pv = predict(model, v, a.device)["l2"].argmax(0)
        f1 = macro_f1(confusion(v.l2, pv, ncls["l2"]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None, "val_l2_f1": f1})
        if f1 > best:
            best, best_epoch = f1, epoch
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        elif epoch - best_epoch >= a.patience:
            break
    train_time = time.time() - t_start

    model.load_state_dict(best_state)
    te = data["test"][0]
    logits = predict(model, te, a.device)
    pred = {h: logits[h].argmax(0) for h in heads}
    metrics = evaluate(pred, {"l1": te.l1, "l2": te.l2, "s": te.s_masked, "s_all": te.s_all})

    np.savez_compressed(out / "test_logits.npz", **{h: logits[h].astype(np.float16) for h in heads})
    torch.save(best_state, out / "model.pt")
    result = {
        "run_id": run_id(a), "code_version": __version__, "args": vars(a), "fold": fold,
        "n_params": n_params, "receptive_field_steps": TCN.receptive_field_steps(a.layers),
        "crops_per_epoch": crops_per_epoch, "best_epoch": best_epoch, "epochs_run": len(history),
        "val_l2_macro_f1": best, "train_time_s": train_time, "normalisation": data["norm"],
        "class_weights": {h: w.cpu().tolist() for h, w in weights.items()},
        "test": metrics, "history": history,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"{run_id(a)}: val {best:.3f} (epoch {best_epoch}), test L2 {metrics['l2_macro_f1']:.3f}, "
          f"L1 {metrics['l1_macro_f1']:.3f}, time {train_time:.0f}s")


if __name__ == "__main__":
    main()
