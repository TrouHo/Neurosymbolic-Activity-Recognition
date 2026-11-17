"""Walking-vs-rest and 20-class RF with orientation-invariant features from RAW signals.
Per block: |a| (raw) and, for accelerometers, the vertical/horizontal split w.r.t. gravity estimated
by a 2-s moving average."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from emob import config as C
from emob.data.dataset import load_npz
from emob.data.splits import loso_folds

WALK = C.L2_CLASSES.index("Marcher")


def inv_channels(d):
    blocks = [(d["acc50"][:, 0:3], d["usable50"][:, 0], True), (d["acc50"][:, 3:6], d["usable50"][:, 1], True),
              (d["phone50"][:, 0:3], d["phone_usable50"][:, 0], True), (d["phone50"][:, 3:6], d["phone_usable50"][:, 1], False)]
    out = []
    for x, u, is_acc in blocks:
        x = np.where(u[:, None], np.nan_to_num(x.astype(float)), 0.0)
        mag = np.linalg.norm(x, axis=1)
        if is_acc:
            k = np.ones(100) / 100
            g = np.stack([np.convolve(x[:, j], k, "same") for j in range(3)], 1)
            gn = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)
            dyn = x - g
            vert = (dyn * gn).sum(1)
            hor = np.linalg.norm(dyn - vert[:, None] * gn, axis=1)
            tilt = gn                                  # still orientation dependent; excluded below
            out += [mag, vert, hor]
        else:
            out += [mag]
        out.append(u.astype(float))
    return np.stack(out, 1)


def feats(ch, T):
    pad = np.pad(ch, ((50, 50), (0, 0)), mode="edge")
    w = sliding_window_view(pad, 100, axis=0)[np.arange(T) * 5 + 2]
    spec = np.abs(np.fft.rfft(w - w.mean(-1, keepdims=True), axis=-1)) ** 2
    fr = np.fft.rfftfreq(100, 1 / 50)
    return np.concatenate([w.mean(-1), w.std(-1), w.min(-1), w.max(-1), np.abs(np.diff(w, axis=-1)).mean(-1),
                           fr[spec[..., 1:].argmax(-1) + 1], spec[..., (fr >= .5) & (fr <= 3)].sum(-1) / (spec.sum(-1) + 1e-9)], 1)


data = {}
for i in range(1, 9):
    d = load_npz(f"user{i}"); T = len(d["l2"])
    data[f"user{i}"] = (feats(inv_channels(d)[: T * 5], T), d["l2"].astype(int))

f1w, f1m = [], []
for f in loso_folds():
    Xtr = np.concatenate([data[p][0] for p in f["train"]]); y = np.concatenate([data[p][1] for p in f["train"]])
    Xte, yte = data[f["test"]]; m = yte >= 0
    rf = RandomForestClassifier(200, class_weight="balanced_subsample", n_jobs=16, random_state=0, min_samples_leaf=5)
    rf.fit(Xtr[y >= 0], y[y >= 0]); p = rf.predict(Xte)
    present = np.unique(yte[m])
    a = f1_score(yte[m] == WALK, p[m] == WALK); b = f1_score(yte[m], p[m], labels=present, average="macro")
    f1w.append(a); f1m.append(b)
    print(f"test {f['test']}: walk F1 {a:.3f}  20-class macro-F1 {b:.3f}  acc {np.mean(yte[m]==p[m]):.3f}")
print(f"mean walk F1 {np.mean(f1w):.3f}, mean macro-F1 {np.mean(f1m):.3f}")
