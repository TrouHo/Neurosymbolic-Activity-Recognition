"""Follow-up checks to audit_full.py (read-only, 2026-11-09).

1. Walking label vs torso speed, old method (dbg_mocap.py: Pearson, clip 5000 mm/s) next to
   robust methods (Pearson with clip 2000, Spearman, AUC) -> where the 05/11 r .25-.32 came from.
2. user3 SB hand/belt/leg: lag search over +-60 s against Qualisys (is there a fixable offset?).
3. Phone lag vs Qualisys over +-60 s for every participant.
4. Cross-tab of 'etiq standard' vs whether 'etiq activite' is empty, and value counts of activite.
"""
import glob
import os

import numpy as np
import pandas as pd

FS = 50


def smooth(x, n=FS):
    return pd.Series(x).rolling(n, center=True, min_periods=n // 2).mean().to_numpy()


def env(v):
    m = np.linalg.norm(v, axis=1)
    m[(v == 0).all(axis=1)] = np.nan
    return pd.Series(m).rolling(FS, center=True, min_periods=FS // 2).std().to_numpy()


def mspeed(xyz, clip):
    s = np.r_[np.nan, np.linalg.norm(np.diff(xyz, axis=0), axis=1) * FS]
    s[s > clip] = np.nan
    return smooth(s)


def corr(a, b, rank=False):
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if rank:
        a, b = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return np.corrcoef(a, b)[0, 1] if len(a) > 100 and a.std() > 0 and b.std() > 0 else np.nan


def lag_scan(a, b, max_s=60, step_s=0.2):
    """Spearman r of a[t] vs b[t+lag]; returns (best lag s, best r, r at 0, 2nd-best peak r >5 s away)."""
    a = pd.Series(a).rank().to_numpy(); b = pd.Series(b).rank().to_numpy()
    lags = np.arange(-max_s, max_s + 1e-9, step_s); rs = []
    for L in lags:
        k = int(round(L * FS))
        x, y = (a[:len(a) - k], b[k:]) if k >= 0 else (a[-k:], b[:len(b) + k])
        rs.append(corr(x, y))
    rs = np.array(rs); j = np.nanargmax(rs)
    far = np.abs(lags - lags[j]) > 5
    return float(lags[j]), float(rs[j]), float(rs[np.argmin(np.abs(lags))]), float(np.nanmax(rs[far]))


def auc(score, pos):
    ok = np.isfinite(score); s, p = score[ok], pos[ok]
    n1, n0 = p.sum(), (~p).sum()
    r = pd.Series(s).rank().to_numpy()
    return (r[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
  for f in sorted(glob.glob(r"D:\Projects\EmoB\data\raw\capture_2022-01-26\*.csv")):
    u = os.path.basename(f)[:5]
    df = pd.read_csv(f, low_memory=False)
    fam = ["etiq standard", "etiq mvt_corps", "etiq mvt_bras", "etiq mvt_jambe", "etiq activite"]
    lab = df[fam].astype("string").fillna("")
    idx = np.flatnonzero((lab != "").any(axis=1).to_numpy()); R = slice(idx[0], idx[-1] + 1)

    tor = df[["x_sq_Torse", "y_sq_Torse", "z_sq_Torse"]].to_numpy(float)
    walk = (lab["etiq activite"] == "Marcher").to_numpy()[R]
    print(f"\n=== {u}")
    for clip in (5000, 2000):
        sp = mspeed(tor, clip)[R]
        print(f"  walk vs torso speed, clip {clip}: Pearson {corr(walk.astype(float), sp):.2f}  "
              f"Spearman {corr(walk.astype(float), sp, rank=True):.2f}  AUC {auc(sp, walk):.3f}  "
              f"rows clipped {np.mean(np.r_[np.nan, np.linalg.norm(np.diff(tor, axis=0), axis=1) * FS][R] > clip):.4f}")

    ref = {"phone ACC": (["x_sf", "y_sf", "z_sf"], "D_hanche"),
           "SB hand ACC": (["x_sb.6", "y_sb.6", "z_sb.6"], "D_main"),
           "SB belt ACC": (["x_sb", "y_sb", "z_sb"], "Bassin"),
           "SB leg ACC": (["x_sb.3", "y_sb.3", "z_sb.3"], "D_genou")}
    for name, (cols, mk) in ref.items():
        v = df[cols].to_numpy(float).copy()
        ok = np.isfinite(v).all(1) & ~(v == 0).all(1)
        if ok.mean() < 0.5 or max(len(np.unique(v[ok, k])) for k in range(3)) < 50:
            print(f"  {name:12s} vs {mk}: no signal"); continue
        e = env(v)[R]
        s = mspeed(df[[f"x_sq_{mk}", f"y_sq_{mk}", f"z_sq_{mk}"]].to_numpy(float), 2000)[R]
        L, rb, r0, rfar = lag_scan(s, e)
        print(f"  {name:12s} vs {mk:8s}: best lag {L:+5.1f}s r {rb:.2f} | r at 0 {r0:.2f} | best r >5s from peak {rfar:.2f}")

    a = lab["etiq activite"].to_numpy(object)[R]; st = lab["etiq standard"].to_numpy(object)[R]
    print("  standard x (activite empty?):", pd.crosstab(st, a == "").to_dict())
    print("  activite top values:", pd.Series(a).value_counts().head(6).to_dict())
    print("  categorie values:", df["etiq categorie"].astype("string").iloc[R].value_counts().to_dict())


if __name__ == "__main__":
    main()
