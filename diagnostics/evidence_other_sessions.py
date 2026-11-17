"""Raw-text evidence for the 2021, run1 and run2 audit findings (read-only, 2026-11-09)."""
import glob
import os
import re

import numpy as np
import pandas as pd

RAW = r"D:\Projects\EmoB\data\raw"
pd.set_option("display.width", 250)

print("#" * 100 + "\n# 2021 capture\n" + "#" * 100)
for f in sorted(glob.glob(os.path.join(RAW, "capture_2021-05-31", "*.csv")),
                key=lambda p: int(re.search(r"\d+", os.path.basename(p)).group())):
    df = pd.read_csv(f, dtype=str, keep_default_na=False)
    cols = list(df.columns)
    present, empty = [], []
    for i, c in enumerate(cols):
        if c.startswith("ts_S") and c != "ts_SQ" and not c.endswith(("LAC", "MAG")):
            vals = set(pd.unique(df[cols[i + 1:i + 4]].to_numpy().ravel()))
            (empty if vals <= {"", "0", "NaN", "nan", "0.0"} else present).append(c[3:])
    nlab = int((df["etiq activite"] != "").sum())
    print(f"\n{os.path.basename(f)}: {len(df)} rows, activite non-empty on {nlab} rows")
    print("  blocks with values :", present)
    print("  blocks only 0/NaN/empty:", empty)

print("\n-- 2021 SB accelerometer unit (raw text, part2, first 3 rows with a non-zero torso value): |a| near 1000 = milli-g")
df = pd.read_csv(os.path.join(RAW, "capture_2021-05-31", "part2_data.csv"), dtype=str, keep_default_na=False)
i = list(df.columns).index("ts_SB - Torse - ACC")
blk = df.iloc[:, i + 1:i + 4]
nz = np.flatnonzero((blk.apply(pd.to_numeric, errors="coerce").fillna(0) != 0).any(axis=1).to_numpy())
print(df.iloc[nz[1000:1003], i:i + 4].to_string())

print("\n" + "#" * 100 + "\n# run1 (Session 3)\n" + "#" * 100)
for f in sorted(glob.glob(os.path.join(RAW, "run1", "acc_data", "*.csv"))):
    lines = open(f, encoding="utf-8", errors="replace").read().splitlines()
    pos = re.search(r"Position (\w+)", lines[0]).group(1)
    data = lines[3:]
    bad = [(j + 4, l) for j, l in enumerate(data) if not re.fullmatch(r"[-0-9.eE]+(,[-0-9.eE]+){7}", l)]
    first, last = data[0].split(",")[0], data[-1].split(",")[0]
    print(f"{os.path.basename(f):26s} {pos:6s} {lines[2]} | data rows {len(data)} | timestamp_pc first {first} last {last} ms"
          + (f" | malformed: {bad[:2]}" if bad else ""))

print("\n-- run1 participant3 label rows with problems (raw text)")
blob = open(os.path.join(RAW, "run1", "labels", "participant3_label.csv"), "rb").read().decode("cp1252")
for l in blob.splitlines():
    if "baiss" in l and "se lever depuis" in l:
        print("  ", l)

print("\n" + "#" * 100 + "\n# run2 (Session 4)\n" + "#" * 100)
f = os.path.join(RAW, "run2", "acc_data", "output1_poignet.txt")
lines = open(f, encoding="utf-8", errors="replace").read().splitlines()
print("-- output1_poignet.txt, first rows (raw): columns 6-8 = raw X Y Z, columns 9-11 = supplied scaled values")
for l in lines[1:8]:
    print("  ", l)
print("   Example: Z raw 59006 -> supplied 3599.37 (= 59006 x 0.061). As a signed 16-bit number 59006 = 59006 - 65536 = -6530.")
for name in ("output1_ceinture.txt", "output5_ceinture.txt"):
    ll = open(os.path.join(RAW, "run2", "acc_data", name), encoding="utf-8", errors="replace").read().splitlines()
    ts = [l for l in ll if ";Timestamp;" in l][:3]
    print(f"-- {name}: first clock rows: {ts}")
present = {}
for p in glob.glob(os.path.join(RAW, "run2", "acc_data", "*.txt")):
    k, site = re.search(r"output(\d+)_(\w+)", p).groups()
    present.setdefault(int(k), []).append(site)
print("-- sensor logs present per participant (Qualisys exists for 1-15):")
for k in range(1, 16):
    print(f"   participant {k:2d}: {sorted(present.get(k, [])) or 'NO SENSOR FILE'}")
