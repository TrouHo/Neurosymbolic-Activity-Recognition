"""Read-only: segment-duration statistics on the 2022 capture, to justify time-scale choices."""
import glob, os
import numpy as np, pandas as pd

DT = 0.02  # ts_SQ step of 20 -> 50 Hz grid (verified earlier)
POST = {"Debout_Assis","Assis_Debout","Assis_Couche","Couche_Assis","Debout_Penche","Penche_Debout",
        "Debout_Agenou","Agenou_Debout","Debout_Accroupi","Accroupi_Debout","Autre_Transition"}
COLS = ["ts_SQ","etiq activite","etiq mvt_corps"]

def runs(values):
    """Contiguous runs of equal non-empty values -> list of (value, start_idx, end_idx_exclusive)."""
    out = []; v = values
    n = len(v); i = 0
    while i < n:
        j = i
        while j + 1 < n and v[j+1] == v[i]: j += 1
        if isinstance(v[i], str) and v[i] != "": out.append((v[i], i, j+1))
        i = j + 1
    return out

seg = {}; state_eps = []; rec_info = []
for f in sorted(glob.glob(r"D:\Projects\EmoB\data\raw\capture_2022-01-26\*.csv")):
    u = os.path.basename(f).split("-")[0]
    df = pd.read_csv(f, usecols=COLS, sep=None, engine="python", dtype={"etiq activite": str, "etiq mvt_corps": str})
    ts = df["ts_SQ"].to_numpy(float)
    assert np.all(np.diff(ts) == 20), u
    act = df["etiq activite"].fillna("").to_numpy(object)
    r = runs(act)
    lab = [x for x in r if x[0] not in ("Debut","Fin")]
    first = min(s for _, s, _ in lab); last = max(e for _, _, e in lab)
    rec_info.append((u, len(df)*DT, (last-first)*DT))
    for name, s, e in lab:
        seg.setdefault(name, []).append((e - s) * DT)
    # posture episodes: time between consecutive posture transitions in the body column
    body = df["etiq mvt_corps"].fillna("").to_numpy(object)
    tr = [x for x in runs(body) if x[0] in POST]
    for (n1, s1, e1), (n2, s2, e2) in zip(tr[:-1], tr[1:]):
        to_state = n1.split("_")[-1] if n1 != "Autre_Transition" else "?"
        state_eps.append((u, to_state, (s2 - e1) * DT))

print("Recording length / labelled span (s):")
for u, L, S in rec_info: print(f"  {u}: total {L:7.1f}  labelled span {S:7.1f}")

print("\nSegment durations of `etiq activite` (s), 2022, all users:")
print(f"{'class':24s} {'n':>4s} {'min':>6s} {'p5':>6s} {'p25':>6s} {'med':>6s} {'p75':>6s} {'max':>7s}")
allv = []
for k in sorted(seg, key=lambda k: np.median(seg[k])):
    v = np.array(seg[k]); allv += list(v)
    print(f"{k:24s} {len(v):4d} {v.min():6.2f} {np.percentile(v,5):6.2f} {np.percentile(v,25):6.2f} {np.median(v):6.2f} {np.percentile(v,75):6.2f} {v.max():7.1f}")
allv = np.array(allv)
print(f"ALL segments n={len(allv)}: share < 0.2 s = {np.mean(allv<0.2):.3f}, < 0.5 s = {np.mean(allv<0.5):.3f}, < 1 s = {np.mean(allv<1):.3f}")

e = pd.DataFrame(state_eps, columns=["user","state","dur"])
print("\nPosture episodes = time between end of a transition and start of the next one (s):")
for st, g in e.groupby("state"):
    v = g["dur"].to_numpy()
    print(f"  after ->{st:9s} n={len(v):3d} med={np.median(v):6.1f} p75={np.percentile(v,75):6.1f} p90={np.percentile(v,90):6.1f} max={v.max():6.1f}")
v = e["dur"].to_numpy()
print(f"  ALL n={len(v)} med={np.median(v):.1f} p75={np.percentile(v,75):.1f} p90={np.percentile(v,90):.1f} p95={np.percentile(v,95):.1f} max={v.max():.1f}; share <= 51 s: {np.mean(v<=51):.3f}; <= 25.5 s: {np.mean(v<=25.5):.3f}")
neg = (v < 0).sum(); print(f"  negative gaps (overlapping transitions): {neg}")
