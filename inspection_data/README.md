# inspection_data — the single source of numbers about the data

Every number about the eMOB 2022 data that is cited in a report (proposal, slides, notes) must come
from the output of this package, `reports/inspection/inspection_numbers.md` (and `.json`), and be
cited there by its ID. Numbers are never recomputed with an ad-hoc script and copied by hand.

```powershell
cd code
.\.venv\Scripts\python.exe -m inspection_data.run_all     # about 20 s
```

## What it guarantees

- **One definition per quantity.** Raw CSVs are read with the reader of the build
  (`emob.data.build.read`), so "usable sensor" has one meaning; rule agreement uses
  `emob/data/rule_stats.py`, shared with the gate check `emob.data.check`.
- **Named data version and unit for every number.** *raw* = CSV as delivered; *v2* = prepared data;
  *params* = frozen estimates in `params/*.json`; *code* = model definitions. Unit *row* = 1/50 s,
  *step* = 0.1 s. Different versions or units give different counts of the same thing (for example
  activity segments shorter than 1 s: 41.8 % on raw rows, 39.0 % on raw steps, 38.9 % on v2 steps);
  both are listed, never mixed.
- **v2 is checked before use.** v2 is rebuilt in memory from raw + params and its content hash must
  equal the files on disk; otherwise the run stops.
- **Fingerprints.** The output lists the SHA-256 of every raw file, every params file and the v2
  content hashes, plus the git commit.

## Steps

| Script | Content |
|---|---|
| `01_inventory.py` | recordings, labelled region, usable devices per participant (raw and v2) |
| `02_labels.py` | per-class seconds and participants, segments in three versions, transition durations, track overlap, unannotated time |
| `03_preparation.py` | applied sensor and label shifts, the six abnormal annotations, masked seconds |
| `04_rules.py` | agreement of A1–A7, facts used to justify them, rejected candidate rules |
| `05_targets.py` | coverage of the five heads, posture episodes, class-weight ratios |
| `06_models.py` | parameter counts and receptive field (analytic; checked against the models in `tests/test_inspection.py`) |

## Not yet included

Analyses that need the motion-capture markers beyond head height (trunk angle, marker speed in
unannotated gaps, label-offset agreement across tracks, residual sensor lag, missing transitions)
are still in `code/diagnostics/` and `code/estimate/` with their evidence files in `reports/data/`.
They are listed as such in the reports until they are moved here.
