"""E2 — freeze the per-participant label shift (decided 2026-11-13, confirmed 2026-11-17 as alignment).

The shifts were estimated on 2026-11-13 by diagnostics/check_label_timing.py (labels against the
signed vertical velocity of the head during body transitions; arm and leg families agree within
0.08 s). Rule approved by the user: one shift per participant for every label column, taken from
the "signed: 10 named transitions" body estimate, applied only when its whole 90 % interval lies
outside +-0.1 s. This script only copies the values and the rule into params/label_offsets.json,
with the SHA-256 of the source file, so the build never reads a report.

Sign: d < 0 means the movement happens before the label (labels late); aligning moves the labels
by d, i.e. label_aligned[t] = label_raw[t - d].
Usage (from code/): .venv\\Scripts\\python.exe -m estimate.label_offsets
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime

from emob import config as C

SOURCE = C.PROJECT / "reports" / "data" / "2026-11-13_label_timing.json"
ESTIMATOR = "signed: 10 named"
THRESHOLD = 0.1


def main():
    raw = SOURCE.read_bytes()
    T = json.loads(raw)
    out = {"created": datetime.now().isoformat(timespec="seconds"), "script": "estimate/label_offsets.py",
           "source": "reports/data/2026-11-13_label_timing.json", "source_sha256": hashlib.sha256(raw).hexdigest(),
           "rule": f"estimator '{ESTIMATOR}...'; shift only if the whole 90 % interval is outside +-{THRESHOLD} s; "
                   "same shift for every label column of a participant",
           "sign": "d < 0: labels late; label_aligned[t] = label_raw[t - d]", "participants": {}}
    for u in sorted(T["estimates"]):
        r = next(x for x in T["estimates"][u] if x["estimator"].startswith(ESTIMATOR))
        apply = r["lo"] > THRESHOLD or r["hi"] < -THRESHOLD
        out["participants"][u] = dict(estimate_s=r["d"], lo=r["lo"], hi=r["hi"], applied=bool(apply),
                                      shift_s=float(r["d"]) if apply else 0.0)
        print(f"{u}: estimate {r['d']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] -> shift {out['participants'][u]['shift_s']:+.2f} s")
    p = C.PARAMS / "label_offsets.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("written", p)


if __name__ == "__main__":
    main()
