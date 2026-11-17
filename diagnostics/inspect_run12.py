"""Read-only inspection of run1 (Session 3) and run2 (Session 4) raw files."""
import sys, re, glob, os
import numpy as np
import pandas as pd

RAW = r"D:\Projects\EmoB\data\raw"

print("=== run1 sensor files ===")
for f in sorted(glob.glob(os.path.join(RAW, "run1", "acc_data", "*.csv"))):
    with open(f, encoding="utf-8", errors="replace") as fh:
        head = [fh.readline().strip() for _ in range(3)]
    df = pd.read_csv(f, skiprows=3, header=None, dtype=str)
    df.columns = ["ts_pc", "millis", "ax", "ay", "az", "gx", "gy", "gz"][: df.shape[1]]
    df = df.apply(pd.to_numeric, errors="coerce")
    bad = df.isna().any(axis=1)
    if bad.any():
        print(f"   [{os.path.basename(f)}] non-numeric rows: {bad.sum()} at row idx {list(np.where(bad)[0][:5])}")
    df = df[~bad]
    ms = df["millis"].to_numpy(float)
    pc = df["ts_pc"].to_numpy(float)
    dms = np.diff(ms); dpc = np.diff(pc)
    acc = df[["ax", "ay", "az"]].to_numpy(float)
    gyr = df[["gx", "gy", "gz"]].to_numpy(float)
    mag = np.linalg.norm(acc, axis=1)
    pos = re.search(r"Position (\w+)", head[0]).group(1)
    print(f"{os.path.basename(f):28s} pos={pos:6s} n={len(df):6d} "
          f"dur_millis={(ms[-1]-ms[0])/1000:7.1f}s dur_pc={(pc[-1]-pc[0])/1000:7.1f}s "
          f"rate~{len(df)/((ms[-1]-ms[0])/1000):5.1f}Hz dt_med={np.median(dms):5.1f} dt_p95={np.percentile(dms,95):6.1f} "
          f"dt_max={dms.max():8.0f} back_pc={(dpc<0).sum()} back_ms={(dms<0).sum()} "
          f"|acc|med={np.median(mag):5.2f} gyr_allzero={np.all(gyr==0)} gyr_std={gyr.std(0).round(2)} start='{head[2][-26:]}'")

print("\n=== run1 label files ===")
def tosec(s):
    h, m, rest = s.split(":"); sec, ms = rest.split(",")
    return int(h)*3600 + int(m)*60 + int(sec) + int(ms)/1000
for f in sorted(glob.glob(os.path.join(RAW, "run1", "labels", "*.csv"))):
    rows = []
    for line in open(f, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line: continue
        parts = re.findall(r'"+([^"]*)"+', line)
        name = line.split(",")[0].strip('"')
        times = re.findall(r"\d\d:\d\d:\d\d,\d{3}", line)
        rows.append((name, tosec(times[0]), tosec(times[1])))
    st = np.array([r[1] for r in rows]); en = np.array([r[2] for r in rows])
    print(f"{os.path.basename(f):24s} intervals={len(rows):3d} first_start={st.min():7.1f}s last_end={en.max():7.1f}s "
          f"labelled_total={np.sum(en-st):6.1f}s median_len={np.median(en-st):4.2f}s")

print("\n=== run2 sensor logs ===")
for f in sorted(glob.glob(os.path.join(RAW, "run2", "acc_data", "*.txt")),
                key=lambda p: (re.search(r"output(\d+)_(\w+)", p).group(2), int(re.search(r"output(\d+)", p).group(1)))):
    ts_vals, acc_raw, other = [], [], {}
    acc_per_ts, cnt = [], 0
    with open(f, encoding="utf-8", errors="replace") as fh:
        fh.readline(); fh.readline()
        for line in fh:
            p = line.strip().split(";")
            if len(p) < 5: continue
            kind = p[4]
            if kind == "Timestamp":
                ts_vals.append(int(p[5])); acc_per_ts.append(cnt); cnt = 0
            elif kind == "Accelerometer":
                b = p[2].split()
                vals = [int(b[i+1] + b[i], 16) for i in (0, 2, 4)]  # little endian XL XH
                acc_raw.append(vals + [int(p[5]), int(p[6]), int(p[7])]); cnt += 1
            else:
                other[kind] = other.get(kind, 0) + 1
    a = np.array(acc_raw)
    match = np.mean(np.all(a[:, :3] == a[:, 3:6], axis=1))
    u = a[:, :3].astype(float)
    s = np.where(u >= 32768, u - 65536, u)
    mag_u = np.linalg.norm(u, axis=1); mag_s = np.linalg.norm(s, axis=1)
    ts = np.array(ts_vals); dts = np.diff(ts)
    apt = np.array(acc_per_ts[1:])
    print(f"{os.path.basename(f):22s} acc_rows={len(a):6d} ts_rows={len(ts):5d} bytes==rawXYZ:{match:.3f} "
          f"ts_first={ts[0]} ts_last={ts[-1]} dts_med={np.median(dts) if len(dts) else None} dts_neg={(dts<0).sum()} "
          f"acc/ts_med={np.median(apt) if len(apt) else None} other={other} "
          f"|u|med={np.median(mag_u):7.0f} |u|IQR={np.subtract(*np.percentile(mag_u,[75,25])):7.0f} "
          f"|s|med={np.median(mag_s):7.0f} |s|IQR={np.subtract(*np.percentile(mag_s,[75,25])):7.0f} "
          f"frac_any_neg_signed={np.mean((s<0).any(1)):.2f}")
