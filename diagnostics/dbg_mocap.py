"""Three-way timing check per recording: walking labels vs torso speed (Qualisys) vs phone energy."""
import glob, os
import numpy as np, pandas as pd

LAGS = np.arange(-60, 60.1, 0.2)  # seconds


def best_lag(a, b, ok):
    """lag (s) of b relative to a maximising Pearson correlation (positive = b later)."""
    res = []
    for L in LAGS:
        k = int(round(L * 50))
        if k >= 0:
            x, y, m = a[: len(a) - k], b[k:], ok[: len(a) - k] & ok[k:]
        else:
            x, y, m = a[-k:], b[: len(b) + k], ok[-k:] & ok[: len(b) + k]
        x, y = x[m], y[m]
        res.append(np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else np.nan)
    res = np.array(res); j = np.nanargmax(res); z = np.argmin(np.abs(LAGS))
    return LAGS[j], res[j], res[z]


def smooth(x, n=50):
    return np.convolve(x, np.ones(n) / n, "same")


for f in sorted(glob.glob(r"D:\Projects\EmoB\data\raw\capture_2022-01-26\*.csv")):
    df = pd.read_csv(f, sep=None, engine="python")
    tor = df[["x_sq_Torse", "y_sq_Torse", "z_sq_Torse"]].to_numpy(float)
    tok = np.isfinite(tor).all(1) & ~(tor == 0).all(1)
    v = np.zeros(len(tor)); v[1:] = np.linalg.norm(np.diff(tor[:, :2], axis=0), axis=1) * 50  # horizontal speed
    vok = tok & np.roll(tok, 1)
    v = np.where(vok, v, 0); v = np.clip(v, 0, 5000)
    speed = smooth(v)
    ph = df[["x_sf", "y_sf", "z_sf"]].to_numpy(float)
    mag = np.linalg.norm(ph, axis=1)
    energy = np.sqrt(smooth((mag - smooth(mag)) ** 2))
    walk = (df["etiq mvt_jambe"] == "Marcher").to_numpy(float)
    ann = df[["etiq standard", "etiq mvt_corps", "etiq mvt_bras", "etiq mvt_jambe"]].notna().any(axis=1).to_numpy()
    sok = smooth(vok.astype(float)) > 0.99
    l1 = best_lag(walk, speed, ann & sok)
    l2 = best_lag(speed, energy, sok)
    l3 = best_lag(walk, energy, ann)
    sp_w = speed[(walk == 1) & sok].mean(); sp_n = speed[(walk == 0) & sok & ann].mean()
    print(f"{os.path.basename(f)[:5]}: torso coverage {100*vok.mean():4.1f}% | speed walk/non-walk {sp_w:6.0f}/{sp_n:6.0f} mm/s | "
          f"labels->mocap lag {l1[0]:+5.1f}s (r {l1[1]:.2f}, at 0: {l1[2]:.2f}) | mocap->phone lag {l2[0]:+5.1f}s (r {l2[1]:.2f}, at 0: {l2[2]:.2f}) | "
          f"labels->phone lag {l3[0]:+5.1f}s (r {l3[1]:.2f}, at 0: {l3[2]:.2f})")
