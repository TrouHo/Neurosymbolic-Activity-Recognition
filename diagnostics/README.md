# Diagnostic scripts (read-only)

Kept for traceability of facts in `knowledge_base/01_project_facts.md`. None of them modifies data.
Run from `D:\Projects\EmoB\code` with `$env:PYTHONPATH="."` when they import `emob`.

| Script | Purpose (date) |
|---|---|
| `durations_2022.py` | Segment and posture-episode durations (2026-11-05) |
| `inspect_run12.py` | First inspection of run1 / run2 (Sessions 3–4) (2026-11-05) |
| `sensor_resolution.py` | Distinct values and scale of every sensor block → SB ACC of user1/2/4 has no signal (2026-11-05) |
| `dbg_align.py` | Sensor activity per label and orientation per posture state (2026-11-05) |
| `dbg_orient2.py` | RF with orientation-invariant features from raw signals, 8 folds (needs sklearn; run on server) (2026-11-05) |
| `dbg_lag.py` | Lag between walking labels and phone energy (2026-11-05) |
| `dbg_smooth.py` | Repeated samples and gait spectrum during walking (2026-11-05) |
| `dbg_mocap.py` | Three-way timing: walking labels vs torso speed (Qualisys) vs phone energy (2026-11-05) |
| `dbg_drift.py` | Windowed lag between walking labels and torso speed (2026-11-05) — **superseded**: Pearson-based, conclusion withdrawn 2026-11-09 |
| `audit_full.py` + `audit_summary.py` | Full audit, every block of every CSV: presence, plausibility, sampling, timestamps, timing vs Qualisys; labels vs Qualisys (2026-11-09). `--year 2021` also works |
| `audit_followup.py` | Old vs robust label/mocap measures; ±60 s lag search per sensor; `standard` vs empty rows (2026-11-09) |
| `audit_drift.py` | Sensor lag vs Qualisys per 120-s window (is it constant?) (2026-11-09; imports `audit_followup`) |
| `audit_label_windows.py` | Walking-label timing per 120-s window, AUC-based (2026-11-09) |
| `evidence_raw_rows.py` | Raw CSV text (20 rows while walking + value counts) for every block marked ✗ (2026-11-09) |
| `evidence_lag_curve.py` | Match with Qualisys vs sensor time shift (what "lag" means) (2026-11-09) |
| `evidence_lag_timeline.py` | Timeline label / torso speed / IMU activity around a walking onset (2026-11-09) |
| `lagtools.py` | Shared timing method (activity vs marker speed, FFT lag scan, windows, AUC, mp4 header reader) (2026-11-09) |
| `audit_capture.py` | Timing + labels for the synchronised CSV captures; `2021` or `2022` (validated on 2022) (2026-11-09) |
| `audit_2021_labels.py` | 2021: do label tracks belong to their recording (±900 s, all pairs); label drift (2026-11-09) |
| `audit_run1.py` | run1 / Session 3: sensors, wall-clock alignment with Qualisys, labels vs Qualisys (2026-11-09) |
| `run2io.py` | Readers for run2 logs and Qualisys files (2026-11-09) |
| `audit_run2.py` | run2 / Session 4: log structure, clocks, signed values, timing at the stated 40 kHz clock (2026-11-09) |
| `audit_run2_clock.py`, `audit_run2_refine.py` | run2: clock factor search, refined alignment, wrist–belt consistency (2026-11-09) |
| `evidence_other_sessions.py` | Raw-text evidence for 2021 / run1 / run2 findings (2026-11-09) |
| `check_autre_transition.py` | How often the posture is unknown after `Autre_Transition`; context of each run; fill backwards / forwards vs Qualisys heights (2026-11-13; needs `PYTHONPATH=.`) |
| `check_bent.py` | Derived posture "bent" vs trunk angle from Qualisys; bends without label inside "standing"; label timing from bends (2026-11-13; needs `PYTHONPATH=.`) |
| `check_label_timing.py` | Time offset of the labels vs Qualisys per participant and family (signed / rank scores, bootstrap over runs, odd / even and thirds); writes `reports/data/2026-11-13_label_timing.json` (2026-11-13; needs `PYTHONPATH=.`) |
| `labelshift.py` | Shared: label families of a recording, shifted by the approved per-participant offsets when `EMOB_LABEL_SHIFT=1` (used by `check_label_timing.py`, `check_bent.py`, `check_autre_transition.py`) (2026-11-13) |
| `check_label_errors.py` | Long runs (first / last real movement, activity in the middle), transitions that do not chain, trunk inclined on "standing" rows; timelines per case (2026-11-13; needs `PYTHONPATH=.`, run with `EMOB_LABEL_SHIFT=1`) |
| `check_empty_rows.py` | Are gaps without label motionless? Marker speeds in gaps vs labelled movement, per gap length, posture and participant; posture stability in gaps (2026-11-13; needs `PYTHONPATH=.` and `2026-11-13_label_timing.json`) |
| `labelrepair.py` | Shared: loads the 2022 recordings with shifted labels, recomputes the boundaries of the six long runs (method of `check_label_errors.py`) and returns the label families under the proposed treatment (middle = not annotated); prints what follows each long run (2026-11-15) |
| `check_single_label_rules.py` | One label per time step from three families: how `activite` chooses; seconds per class under 8 candidate rules; options U1–U3 for rows where a family is not annotated; fragmentation and classes per participant (2026-11-15; needs `PYTHONPATH=.`, run with `EMOB_LABEL_SHIFT=1`) |
| `check_rules_shifted.py` | Agreement of rules A1–A4, B1, B2, B5a with the labels, as labelled and after the proposed treatment of the long runs (2026-11-15; needs `PYTHONPATH=.`; run with and without `EMOB_LABEL_SHIFT=1`) |
| `rule_loss_train_vs_test.py` | Rule losses (soft, labelled steps) and arg-max violations of TCN-B2 vs TCN-NS2 on training and test participants; output `reports/data/2026-11-18_rule_loss_train_vs_test.json` (2026-11-18; needs torch, `PYTHONPATH=.`) |
| `audit_posture.py` | Posture states (`s_all`) vs head/hip height (2026-11-09; needs `PYTHONPATH=.`) |
