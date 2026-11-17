"""Does the signal change with the label? Activity of each sensor per L2 class / S state."""
import json
import numpy as np
from emob import config as C
from emob.data.dataset import load_npz

for p in ["user6", "user3"]:
    d = load_npz(p)
    T = len(d["l2"])
    acc = np.concatenate([d["acc50"], d["phone50"]], axis=1)[: T * 5].reshape(T, 5, 12)
    us = np.concatenate([d["usable50"], d["phone_usable50"]], axis=1)[: T * 5].reshape(T, 5, 4).all(1)
    # activity = std over a 1-s window of |a| for each block
    mag = np.stack([np.linalg.norm(acc[:, :, 3*k:3*k+3], axis=2).mean(1) for k in range(4)], 1)  # (T,4)
    act = np.stack([np.convolve(np.abs(np.diff(mag[:, k], prepend=mag[0, k])), np.ones(10) / 10, "same") for k in range(4)], 1)
    print(p, "blocks: hand, belt, phone_acc, phone_gyr")
    for c in ["Marcher", "Autre_Mouvement_Bras", "Debout_Assis", "Lever_Bras"]:
        i = C.L2_CLASSES.index(c)
        m = d["l2"] == i
        print(f"  {c:22s} n={m.sum():5d} activity={act[m].mean(0).round(3)}")
    for j, s in enumerate(C.S_CLASSES):
        m = d["s_all"] == j
        if m.sum():
            # mean phone gravity direction (orientation) per state
            g = acc[:, :, 6:9].mean(1)[m].mean(0)
            gb = acc[:, :, 3:6].mean(1)[m].mean(0)
            print(f"  S {s:9s} n={m.sum():5d} phone_acc_mean={g.round(2)} belt_acc_mean={gb.round(2)}")
    m = (d["l2"] == C.IGNORE) & d["annotated"]
    print("  annotated but no L2:", m.sum(), "| no annotation:", (~d["annotated"]).sum(),
          "activity there", act[~d["annotated"]].mean(0).round(3))
