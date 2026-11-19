"""03 Preparation: what each preparation step changed, from the frozen parameters and the rebuild.

Sensor time alignment: params/sensor_offsets.json (estimate/sensor_offsets.py); lag > 0 = device
late relative to Qualisys. Label time alignment: params/label_offsets.json (estimate/label_offsets.py);
shift < 0 moves annotations earlier. Abnormal long annotations: params/long_runs.json
(estimate/long_runs.py). "applied" values are the ones used by the build (rounded to rows).
"""
from __future__ import annotations

from emob import config as C
from inspection_data.common import PNAME, Context, Registry

DEV = {"phone": "phone_acc", "hand": "hand", "waist": "belt", "leg": "leg"}
PARAM_DEV = {"phone": "phone", "hand": "hand", "waist": "belt", "leg": "leg"}


def run(ctx: Context, reg: Registry):
    reg.section = "03 Preparation"
    U = ctx.users
    # ---- sensor time alignment
    phone = []
    for u in U:
        P, st = PNAME[u], ctx.stats(u)
        for dev, blk in DEV.items():
            s = st["sensors"][blk]
            if s["shift_s"] is not None:
                reg.add(f"prep.sensor.{P}.{dev}.applied_shift_s", s["shift_s"], "s", "params",
                        f"{P} {dev}: applied time shift ({s['status']}); > 0 = device was late")
            d = ctx.sens_off["participants"][u]["devices"][PARAM_DEV[dev]]
            if d.get("signal") and d.get("median") is not None:
                reg.add(f"prep.sensor.{P}.{dev}.measured_lag_s", d["median"], "s", "params",
                        f"{P} {dev}: measured lag (median over 120-s windows)")
                if d.get("lo") is not None:
                    reg.add(f"prep.sensor.{P}.{dev}.interval_width_s", d["hi"] - d["lo"], "s", "params",
                            f"{P} {dev}: width of the 90 % bootstrap interval of the lag")
            fb = d.get("fallback") or {}
            if fb.get("usable"):
                reg.add(f"prep.sensor.{P}.{dev}.fallback_windows", f"{fb['n_verified']}/{fb['n_windows_with_data']}",
                        "60-s windows", "params", f"{P} {dev}: windows retained by the window-level fallback")
        phone.append(st["sensors"]["phone_acc"]["shift_s"])
    reg.add("prep.sensor.phone.applied_min_s", min(phone), "s", "params", "smallest applied phone shift (8 participants)")
    reg.add("prep.sensor.phone.applied_max_s", max(phone), "s", "params", "largest applied phone shift (8 participants)")
    g = ctx.sens_off["gate"]
    reg.add("prep.sensor.gate.r_min", g["r_min"], "-", "params", "minimum correlation peak for a recording-wide correction")
    reg.add("prep.sensor.gate.ci_width_max_s", g["ci_width_max_s"], "s", "params", "maximum width of the 90 % interval")
    reg.add("prep.sensor.gate.drift_max_s", g["drift_max_s"], "s", "params", "maximum drift over the recording")

    # ---- label time alignment
    shifted = []
    for u in U:
        P, d = PNAME[u], ctx.lab_off["participants"][u]
        reg.add(f"prep.label.{P}.estimate_s", d["estimate_s"], "s", "params", f"{P}: estimated annotation offset (< 0 = labels late)")
        reg.add(f"prep.label.{P}.interval", f"[{d['lo']:+.2f}, {d['hi']:+.2f}]", "s", "params", f"{P}: 90 % interval of the estimate")
        reg.add(f"prep.label.{P}.applied_shift_s", d["shift_s"], "s", "params", f"{P}: applied annotation shift (0 = not shifted)")
        if d["applied"]:
            shifted.append(d["shift_s"])
    reg.add("prep.label.n_shifted", len(shifted), "participants", "params", "participants whose annotations were shifted")
    reg.add("prep.label.shift_min_s", min(shifted), "s", "params", "most negative applied annotation shift")
    reg.add("prep.label.shift_max_s", max(shifted), "s", "params", "most positive applied annotation shift")

    # ---- abnormal long annotations
    runs = ctx.long_runs["runs"] if isinstance(ctx.long_runs, dict) else ctx.long_runs
    for k, r in enumerate(runs, 1):
        P = PNAME[r["user"]]
        tag = f"prep.longrun.{k}.{P}.{r['label']}"
        reg.add(f"{tag}.original_s", r["length_s"], "s", "params", f"{P} {r['label']}: original annotation length")
        reg.add(f"{tag}.candidate_s", r["kept_length_s"] if r["kept_length_s"] is not None else "-", "s", "params",
                "initial part supported by motion capture")
        reg.add(f"{tag}.others_max_s", r["others_max_s"], "s", "params", "longest regular instance of the label in the 7 other recordings")
        reg.add(f"{tag}.order", r["previous_followed_by_label_in"], "participants", "params",
                "other recordings where the preceding label is followed by this label")
        reg.add(f"{tag}.decision", r["decision"], "-", "params", "decision")
    lens = [r["length_s"] for r in runs]
    reg.add("prep.longrun.n", len(runs), "annotations", "params", "abnormally long annotations")
    reg.add("prep.longrun.length_min_s", min(lens), "s", "params", "shortest of them")
    reg.add("prep.longrun.length_max_s", max(lens), "s", "params", "longest of them")
    reg.add("prep.longrun.others_max_max_s", max(r["others_max_s"] for r in runs), "s", "params",
            "largest 'longest regular instance' among the six labels")
    masked = {u: ctx.stats(u)["activite_masked_s"] for u in U}
    reg.add("prep.masked.activity_s", sum(masked.values()), "s", "v2", "activity-column rows masked as unsupported")
    for u in U:
        if masked[u]:
            reg.add(f"prep.masked.{PNAME[u]}.activity_s", masked[u], "s", "v2", f"{PNAME[u]}: masked activity rows")
