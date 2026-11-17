"""Local lag between walking labels and torso speed in sliding windows (does the offset drift?)."""
import glob, os
import numpy as np, pandas as pd

LAGS = np.arange(-40, 40.1, 0.2)


def smooth(x, n=50):
    return np.convolve(x, np.ones(n) / n, "same")


for f in sorted(glob.glob(r"D:\Projects\EmoB\data\raw\capture_2022-01-26\*.csv")):
    u = os.path.basename(f)[:5]
    if u not in ("user1", "user2", "user3", "user4", "user5", "user7"):
        continue
    df = pd.read_csv(f, sep=None, engine="python", usecols=["x_sq_Torse", "y_sq_Torse", "z_sq_Torse", "etiq mvt_jambe"])
    tor = df[["x_sq_Torse", "y_sq_Torse", "z_sq_Torse"]].to_numpy(float)
    tok = np.isfinite(tor).all(1) & ~(tor == 0).all(1)
    v = np.zeros(len(tor)); v[1:] = np.linalg.norm(np.diff(tor[:, :2], axis=0), axis=1) * 50
    vok = tok & np.roll(tok, 1); v = np.clip(np.where(vok, v, 0), 0, 5000)
    speed = smooth(v); sok = smooth(vok.astype(float)) > 0.99
    walk = (df["etiq mvt_jambe"] == "Marcher").to_numpy(float)
    W, S = 150 * 50, 60 * 50
    out = []
    for s0 in range(0, len(walk) - W, S):
        best = (np.nan, -1)
        for L in LAGS:
            k = int(round(L * 50))
            a0, a1 = s0, s0 + W
            b0, b1 = a0 + k, a1 + k
            if b0 < 0 or b1 > len(walk):
                continue
            x, y, m = walk[a0:a1], speed[b0:b1], sok[b0:b1]
            if m.sum() < W * 0.5 or x[m].std() == 0:
                continue
            r = np.corrcoef(x[m], y[m])[0, 1]
            if r > best[1]:
                best = (L, r)
        if np.isfinite(best[0]):
            out.append(f"{s0/50:4.0f}s:{best[0]:+5.1f}({best[1]:.2f})")
    print(u, " ".join(out))
