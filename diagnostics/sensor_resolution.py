"""Read-only: resolution / scale of every sensor block in the 2022 CSVs, inside the annotated region."""
import glob, os
import numpy as np, pandas as pd

FAM = ["etiq standard", "etiq mvt_corps", "etiq mvt_bras", "etiq mvt_jambe"]
for f in sorted(glob.glob(r"D:\Projects\EmoB\data\raw\capture_2022-01-26\*.csv")):
    df = pd.read_csv(f, sep=None, engine="python")
    ann = df[FAM].notna().any(axis=1).to_numpy()
    idx = np.flatnonzero(ann); reg = slice(idx[0], idx[-1] + 1)
    cols = list(df.columns)
    print(os.path.basename(f))
    for i, c in enumerate(cols):
        if not c.startswith("ts_S"):
            continue
        v = df.iloc[reg, i + 1:i + 4].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        ok = np.isfinite(v).all(1) & ~(v == 0).all(1)
        w = v[ok]
        if len(w) == 0:
            print(f"   {c:32s} usable 0%"); continue
        mag = np.linalg.norm(w, axis=1)
        nun = max(len(np.unique(w[:, k])) for k in range(3))
        print(f"   {c:32s} usable {100*ok.mean():5.1f}%  max-unique/axis {nun:6d}  |v| median {np.median(mag):9.4f}  axis std {w.std(0).round(3)}")
