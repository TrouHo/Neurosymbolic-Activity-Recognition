"""Print compact tables from the JSON written by audit_full.py (read-only)."""
import json
import sys

res = json.load(open(sys.argv[1], encoding="utf-8"))

print("== recordings ==")
for r in res:
    print(f"{r['file']:24s} start {r['ts_SQ_start_utc'][:19]}  dur {r['duration_s']:6.0f}s  "
          f"labelled region {r['label_region_s']:5.0f}s  annotated {r['label_annotated_frac_in_region']:.2f}  "
          f"ts_SQ whole-second frac {r['ts_SQ_whole_seconds_frac']:.3f} steps {r['ts_SQ_steps']}")

names = [n for n in res[0]["blocks"] if not n.startswith("SQ ")]
for r in res[1:]:
    for n in r["blocks"]:
        if not n.startswith("SQ ") and n not in names:
            names.append(n)

print("\n== IMU blocks (inside labelled region) ==")
print("file | block | good | distinct | |v| median [IQR] | repeat_prev | ts=SQ | ts offset ms | ts repeat | sync r0/rbest@lag | quarter lags")
for r in res:
    for n in names:
        b = r["blocks"].get(n)
        if b is None:
            print(f"{r['file'][:5]} | {n:28s} | ABSENT"); continue
        nm = "-" if b["norm_median"] is None else f"{b['norm_median']:.2f} [{b['norm_iqr'][0]:.2f},{b['norm_iqr'][1]:.2f}]"
        rp = "-" if b["repeat_prev_frac"] is None else f"{b['repeat_prev_frac']:.2f}"
        te = "-" if b.get("ts_equals_SQ_frac") is None else f"{b['ts_equals_SQ_frac']:.2f}"
        to = "-" if b.get("ts_offset_median_ms") is None else f"{b['ts_offset_median_ms']:.0f}"
        tr = "-" if b.get("ts_repeat_frac") is None else f"{b['ts_repeat_frac']:.2f}"
        sy, ql = "", ""
        if "sync" in b:
            w = b["sync"]["whole"]
            sy = f"{w['r_at_0']:.2f}/{w['r_best']:.2f}@{w['lag_best_s']:+.1f}s"
            ql = " ".join(f"{q['lag_best_s']:+.1f}" if q['lag_best_s'] == q['lag_best_s'] else "nan" for q in b["sync"]["quarters"])
        print(f"{r['file'][:5]} | {n:28s} | {b['good_frac']:.2f} | {b['distinct_max']:5d} | {nm:20s} | {rp} | {te} | {to:>6s} | {tr} | {sy} | {ql}")

print("\n== Qualisys markers (inside labelled region): good frac / repeat_prev ==")
for r in res:
    sq = {n[3:]: b for n, b in r["blocks"].items() if n.startswith("SQ ")}
    print(r["file"][:5], " ".join(f"{k}:{v['good_frac']:.2f}/{v['repeat_prev_frac'] if v['repeat_prev_frac'] is None else round(v['repeat_prev_frac'],2)}" for k, v in sq.items()))

print("\n== label track ==")
for r in res:
    print(r["file"][:5], f"label ts == ts_SQ: {r['label_ts_equals_SQ_frac']:.3f}",
          f"activite empty in region: {r['activite_empty_frac_in_region']:.3f}",
          "standard where activite empty:", r["standard_values_where_activite_empty"],
          "| standard in region:", r["standard_values_in_region"])

print("\n== label checks vs Qualisys ==")
for r in res:
    c = r["label_checks"]
    sh = c.get("walk_auc_vs_label_shift_s", {})
    best = max(sh, key=lambda k: sh[k]) if sh else None
    print(r["file"][:5],
          f"walk AUC(torso speed) {c.get('walk_auc_torso_speed', float('nan')):.3f}",
          f"best label shift {best}s (AUC {sh.get(best, float('nan')):.3f})",
          "AUC by shift:", {k: round(v, 3) for k, v in sh.items()},
          f"| torso mm/s walk/not {[round(x) for x in c.get('torso_speed_mm_s_walk_vs_not') or []]}",
          f"| arm AUC(hand speed vs empty) {c.get('arm_auc_hand_speed_vs_empty', float('nan')):.3f}",
          f"hand mm/s arm/empty {[round(x) for x in c.get('hand_speed_mm_s_arm_vs_empty') or []]}")
