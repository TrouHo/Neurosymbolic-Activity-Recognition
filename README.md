# eMOB — neurosymbolic activity recognition from wearable sensors

Master research project (LIMOS, Université Clermont Auvergne; eMOB project with CHU Clermont-Ferrand and LPC).

**Question.** Does adding prior knowledge about human activity, written as logic rules, to a deep sequence model improve activity recognition from wearable sensors?

**Setup.**
- Sequence labelling: one prediction every 0.1 s.
- Backbones: TCN (main) and DeepConvLSTM (control).
- Main output: 20 activity classes (`activite`).
- Auxiliary outputs: 7 coarse groups, 6 postures, walking, posture transition.
- Rules A1–A7 are turned into differentiable losses.
- Evaluation: leave-one-subject-out over 8 participants.

> Status (2026-11-18): data pipeline v2 complete (gate G 10/10); training, rule losses A1–A7, evaluation and the experiment grid run on data v2 (67 tests pass on the training server).

## Data

The recordings (capture of 2022-01-26, 8 participants, synchronised Qualisys motion capture + inertial sensors + pocket phone, frame-by-frame labels) belong to the eMOB project. **They are not part of this repository and are not distributed.** The code expects them at the path given by `EMOB_RAW_2022` (see `emob/config.py`).

## Repository layout

| Folder | Content |
|---|---|
| `emob/` | Python package used for the experiments: data pipeline (`emob/data/`), models (`emob/models/`), baselines, training, evaluation |
| `estimate/` | Scripts that estimate parameters from the data and freeze them in `params/` |
| `params/` | Frozen parameters (JSON): sensor offsets, label offsets, long label runs |
| `tests/` | `pytest` tests |
| `diagnostics/` | Read-only data checks that produced the evidence for every data-handling decision (see `diagnostics/README.md`) |
| `review_app/` | Local web app to review labels against video (FastAPI + React) |

## Principles

1. **Estimate → freeze → apply.** Every parameter taken from the data (sensor time offsets, label time offsets, treatment of anomalous label runs) is computed by an estimation script and frozen in a JSON file under `params/`, with its evidence. The build reads these files and contains no hand-typed values.
2. **Raw data is read-only.** No interpolation of missing samples. No labels are created. Labels are only time-aligned, masked (excluded from loss and metrics) or read from existing label columns.
3. **Every step has a numeric gate** checked before the next step runs.

## Environment

Python 3.11. Main environment:

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements-lock.txt
```

PyTorch is needed for training (installed on the training server; tests that need it are skipped otherwise).

## Commands

```powershell
# tests
.venv\Scripts\python.exe -m pytest tests

# (only to re-estimate the frozen parameters; the results are already in params/)
.venv\Scripts\python.exe -m estimate.sensor_offsets     # E1 device time offsets vs Qualisys
.venv\Scripts\python.exe -m estimate.label_offsets      # E2 per-participant label shift (decided 2026-11-13)
.venv\Scripts\python.exe -m estimate.long_runs          # E3 verify the six long label runs

# build the data (about 10 s) and run the gate checks
.venv\Scripts\python.exe -m emob.data.build             # -> <project>/data/processed/v2
.venv\Scripts\python.exe -m emob.data.check
```

## Pipeline v2 (data)

| Step | Module | What it does |
|---|---|---|
| S1 | `emob/data/build.py` `read` | Read a CSV; check the 50-Hz grid, the column layout and the label values |
| S2 | `emob/data/sync.py` | Shift each device by its offset against Qualisys; unverified parts become unavailable |
| S4 | `emob/data/anomalies.py` | Mask the unverifiable parts of six long label runs |
| S3 | `emob/data/alignment.py` | Shift the labels of four participants (user2, 5, 6, 8) |
| S5 | `emob/data/labels.py` | Targets: L2 (20 classes, `activite`), L1 (7), S (6 postures), W (walking), T (posture transition) |
| S6–S7 | `emob/data/build.py` | 0.1-s step grid; `userX.npz`, `manifest.json` (git commit, hashes, statistics), `data_card.md` |
| G | `emob/data/check.py` | Rules A1–A7 on the targets, residual sensor lag, reproducibility |

## Training and evaluation

| Module | What it does |
|---|---|
| `emob/data/dataset.py`, `features.py`, `normalize.py`, `splits.py` | Fold assembly: 13 orientation-invariant channels + 4 device flags, z-score on training participants, label fractions, modality dropout, sensor configurations ALL / PHONE / HB |
| `emob/models/` | TCN (main) and DeepConvLSTM (control) backbones; heads L2, L1, S, W, T |
| `emob/rules_spec.py`, `emob/rules_loss.py` | Rules A1–A7: constants and differentiable losses (product fuzzy logic; A1 semantic loss; A7 Gödel min) |
| `emob/train.py` | One run: B1, B2, NS1, NS2, NS3 × TCN / LSTM, one fold, seed, label fraction, lambda set |
| `emob/baselines/rf.py` | B0: window features + Random Forest, one forest per sensor configuration |
| `emob/metrics.py`, `emob/evaluate.py` | Frame and segment metrics; rule violations A1–A7 (same definitions as the losses) |
| `emob/run_grid.py` | Phases 1 (B0, B1, B2), 2a (NS1 lambda grid), 2b (NS2 grid), 3 (NS3); skips finished runs |
| `emob/summarize.py` | Lambda selection on validation; tables; paired comparisons (exact Wilcoxon, Holm) |

```bash
python -m emob.run_grid --phase 1 --workers 4 --out ~/emob/results
python -m emob.summarize --results ~/emob/results --out ~/emob/results/summary.json
```

`userX.npz` keys:
- `x50` (N × 15, 50 Hz, NaN where unavailable) and `avail50` (N × 5): blocks phone ACC, phone GYR, hand, belt, leg ACC;
- per 0.1-s step: `l2`, `l1`, `s`, `s_annotated`, `w`, `t` (−1 = no label); `annotated`; `masked_anomaly`; `fam` (the four label columns, coded with `family_vocab` of the manifest); `l2_raw` (`activite` as in the CSV).
