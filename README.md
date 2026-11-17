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

> Status (2026-11-17): data pipeline v2 under construction. Training code exists from a pilot (v1) and will be updated.

## Data

The recordings (capture of 2022-01-26, 8 participants, synchronised Qualisys motion capture + inertial sensors + pocket phone, frame-by-frame labels) belong to the eMOB project. **They are not part of this repository and are not distributed.** The code expects them at the path given by `EMOB_RAW_2022` (see `emob/config.py`).

## Repository layout

| Folder | Content |
|---|---|
| `emob/` | Python package used for the experiments: data pipeline (`emob/data/`), models (`emob/models/`), baselines, training, evaluation |
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
```

The data-build and training commands will be documented here once pipeline v2 is complete.
