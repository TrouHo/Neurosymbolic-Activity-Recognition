"""Effective resolution of the signals: repeated samples, spectrum during walking."""
import numpy as np
from emob import config as C
from emob.data.dataset import load_npz
from emob.data.labels import runs

walk = C.L2_CLASSES.index("Marcher")
for i in (3, 5, 6, 8):
    d = load_npz(f"user{i}")
    sig = {"hand": d["acc50"][:, 0:3], "belt": d["acc50"][:, 3:6], "phone_acc": d["phone50"][:, 0:3],
           "phone_gyr": d["phone50"][:, 3:6]}
    us = {"hand": d["usable50"][:, 0], "belt": d["usable50"][:, 1], "phone_acc": d["phone_usable50"][:, 0],
          "phone_gyr": d["phone_usable50"][:, 1]}
    print(f"user{i}")
    # longest walking segments (step grid -> samples)
    segs = sorted([(e - s, s, e) for v, s, e in runs(np.where(d["l2"] == walk, "w", "").astype(object))], reverse=True)[:5]
    for name, x in sig.items():
        if not us[name].any():
            print(f"   {name:9s} unusable"); continue
        x = x.astype(float)
        rep = np.mean(np.all(np.diff(x, axis=0) == 0, axis=1)[us[name][1:]])
        # run length of identical consecutive samples
        same = np.all(np.diff(x, axis=0) == 0, axis=1)
        rl = [e - s for v, s, e in runs(np.where(same, "s", "").astype(object))]
        # spectrum of |x| during the longest walking segments
        P = np.zeros(129); n = 0
        for _, s, e in segs:
            seg = np.linalg.norm(x[s * 5:e * 5], axis=1)
            for j in range(0, len(seg) - 256, 128):
                w = seg[j:j + 256] - seg[j:j + 256].mean()
                P += np.abs(np.fft.rfft(w * np.hanning(256))) ** 2; n += 1
        fr = np.fft.rfftfreq(256, 1 / 50)
        if n:
            P /= n
            band = lambda lo, hi: P[(fr >= lo) & (fr < hi)].sum() / P[fr > 0.3].sum()
            peak = fr[1:][P[1:].argmax()]
            print(f"   {name:9s} repeated-sample rate {rep:.2f}, median hold {np.median(rl) if rl else 0:.0f}+1 samples; "
                  f"walking |x| peak {peak:.2f} Hz; power share 0.3-1 Hz {band(.3,1):.2f}, 1-3 Hz {band(1,3):.2f}, 3-25 Hz {band(3,25.1):.2f}")
