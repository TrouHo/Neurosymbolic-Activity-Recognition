"""Print raw CSV text (no numeric conversion) as evidence for the 2026-11-09 audit (read-only).

Part A: 20 consecutive rows during a walking bout, for every SB ACC block claimed 'no signal' or
        'empty', next to the phone ACC of the same rows (working sensor, for comparison).
Part B: value counts of the raw text over the whole labelled region for those columns and for
        every SB GYR / MAG column (so the 20 rows are not cherry-picked).
Part C: one reference participant (user6) printed the same way.
Usage: .venv\\Scripts\\python.exe diagnostics\\evidence_raw_rows.py
"""
import glob
import os

import numpy as np
import pandas as pd

RAW = r"D:\Projects\EmoB\data\raw\capture_2022-01-26"
# pandas renames repeated header names with .1, .2 ... in file order
BLOCKS = {
    "phone ACC": ["x_sf", "y_sf", "z_sf"],
    "SB belt ACC": ["x_sb", "y_sb", "z_sb"], "SB belt GYR": ["x_sb.1", "y_sb.1", "z_sb.1"],
    "SB belt MAG": ["x_sb.2", "y_sb.2", "z_sb.2"],
    "SB leg ACC": ["x_sb.3", "y_sb.3", "z_sb.3"], "SB leg GYR": ["x_sb.4", "y_sb.4", "z_sb.4"],
    "SB leg MAG": ["x_sb.5", "y_sb.5", "z_sb.5"],
    "SB hand ACC": ["x_sb.6", "y_sb.6", "z_sb.6"], "SB hand GYR": ["x_sb.7", "y_sb.7", "z_sb.7"],
    "SB hand MAG": ["x_sb.8", "y_sb.8", "z_sb.8"],
}
CLAIMS = {  # participant -> ACC blocks marked x in the table
    "user1": ["SB hand ACC", "SB belt ACC", "SB leg ACC"],
    "user2": ["SB hand ACC", "SB belt ACC", "SB leg ACC"],
    "user4": ["SB hand ACC", "SB belt ACC", "SB leg ACC"],
    "user5": ["SB belt ACC"],
    "user6": ["SB hand ACC", "SB belt ACC", "SB leg ACC"],   # reference, not a claim
}
GYRMAG = [b for b in BLOCKS if b.endswith(("GYR", "MAG"))]
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

for f in sorted(glob.glob(os.path.join(RAW, "*.csv"))):
    u = os.path.basename(f)[:5]
    df = pd.read_csv(f, dtype=str, keep_default_na=False)          # raw text, '' = empty cell
    fam = ["etiq standard", "etiq mvt_corps", "etiq mvt_bras", "etiq mvt_jambe", "etiq activite"]
    idx = np.flatnonzero((df[fam] != "").any(axis=1).to_numpy())
    R = df.iloc[idx[0]: idx[-1] + 1]
    print(f"\n{'=' * 100}\n{os.path.basename(f)}  (labelled region: rows {idx[0]}..{idx[-1]}, {len(R)} rows)")

    if u in CLAIMS:
        w = np.flatnonzero((R["etiq activite"] == "Marcher").to_numpy())
        start = w[len(w) // 3]                       # a row inside a walking bout, not hand-picked
        rows = R.iloc[start: start + 20]
        cols = ["ts_SQ", "etiq activite"] + BLOCKS["phone ACC"] + sum((BLOCKS[b] for b in CLAIMS[u]), [])
        show = rows[cols].copy()
        show.columns = ["ts_SQ", "activite", "phone_x", "phone_y", "phone_z"] + [
            f"{b.split()[1]}_{a}" for b in CLAIMS[u] for a in "xyz"]
        print(f"-- Part A{' (REFERENCE)' if u == 'user6' else ''}: 20 consecutive raw rows while walking (file row {rows.index[0]}..{rows.index[-1]})")
        print(show.to_string(index=False))
        print("-- Part B: raw value counts over the whole labelled region")
        for b in CLAIMS[u]:
            for c, a in zip(BLOCKS[b], "xyz"):
                vc = R[c].replace("", "<empty>").value_counts()
                top = ", ".join(f"'{k}':{v}" for k, v in vc.head(8).items())
                print(f"   {b} {a}: {len(vc)} distinct | {top}{' ...' if len(vc) > 8 else ''}")
    # GYR / MAG of the SB: claimed empty for everyone
    out = []
    for b in GYRMAG:
        vals = pd.unique(R[BLOCKS[b]].to_numpy().ravel())
        out.append(f"{b}: {sorted(set(v if v else '<empty>' for v in vals))[:5]}")
    print("-- SB GYR/MAG, all distinct raw values in labelled region:\n   " + "\n   ".join(out))
