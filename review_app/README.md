# eMOB Data Review

Local web app to inspect eMOB recordings: video ↔ CSV signals ↔ annotation tracks, with explicit, saved, user-controlled alignment.
It runs on `127.0.0.1` only. Source files are opened **read-only**, and all review outputs go to `output_dir` (default `D:\Projects\EmoB\reports\review_app`).

Authoritative scope: `D:\Projects\EmoB\eMOB_Project_Brief.md`.

## Run (Windows PowerShell)

Prerequisites already installed on this machine: Python 3.13 (`py` launcher) and Node.js LTS 24.

```powershell
cd D:\Projects\EmoB\code\review_app

# one-time: backend environment (separate from code\.venv)
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# one-time: config (edit paths if needed)
Copy-Item config.example.json config.json

# one-time (and after frontend changes): build the UI
cd frontend
npm install
npm run build
cd ..

# start (serves the UI and API at http://127.0.0.1:8765/)
cd backend
..\.venv\Scripts\python.exe -m emob_review
```

Open http://127.0.0.1:8765/ in Edge or Chrome. Stop the server with Ctrl+C.

Frontend development with hot reload: keep the backend running, then run `cd frontend; npm run dev` and open http://127.0.0.1:5173/.

### Tests

```powershell
cd D:\Projects\EmoB\code\review_app\backend
..\.venv\Scripts\python.exe -m pytest -q          # synthetic + real-data (real-data tests skip if files are absent)
cd ..\frontend
npm test                                           # alignment / timestamp / profile / external-label functions
```

## Using it

1. **Select a recording** in the top bar. Pairings are *proposed* by filename rules in `config.json` and stay `proposed` until you click **Confirm pairing**. Use **Open other CSV/video…** for any file inside the data roots. **Scan availability flags** parses every CSV once and marks `labels empty`, `no IMU data` or `GYR empty`.
2. **Video** (left): play/pause, speed, seek, ±1 s, and approximate frame step (`,` `.`). The clock is the browser media time, read via `requestVideoFrameCallback` when available and `currentTime` otherwise; the readout shows which one is in use.
3. **Plots** (right): one plot per selected channel group (x/y/z) at full resolution with no downsampling.
   - Gaps (missing / `NaN` / non-numeric) are drawn as gaps on a red band. Rows where every axis is exactly 0 get a grey band. They are candidates for review, not assumed missing.
   - Mouse wheel zooms, Shift+wheel pans, dragging selects a range, and clicking seeks the video.
4. **Label tracks**: one track per label column, using the raw values. `(empty)` is hatched and kept distinct from `inactif`. There are also two computed tracks: **overlap** (≥2 family columns hold a value other than `""`/`inactif`) and **all empty**. Click a segment to seek; hover for rows and duration.
5. **Alignment** (left, below the readout):
   - The model is `t_csv = a·t_video + b`, with `t_csv = (timestamp − reference)/1000` s.
   - It starts at a=1, b=0 with status **Unverified**. Positive b means video time 0 maps to a later CSV time.
   - Set b with the numeric field, the slider or the nudge buttons (sample / frame / 0.1 s / 1 s).
   - **Anchors**: press *Set video point* on the event frame, then *Pick CSV point in plot* and click the same event in the signal. Choose the kind (signal-based or annotation-based) and press *Add anchor*.
   - *Apply 1 anchor* keeps a and solves b. *Apply 2-anchor fit* solves a and b; it rejects invalid ordering and warns when a ≠ 1.
   - Residuals are shown for every anchor. Only you can set **Confirmed**.
   - Times outside the shared coverage show `No CSV data` or refuse to seek ("outside the video"); they are never pinned to an endpoint. The overview strip shows CSV coverage (blue) against mapped video coverage (orange).
6. **Labels now + verdicts** (left, under the readout): the value of every label column at the current moment, updated
   live while the video plays. For each segment you can give a verdict: **✓ agree**, **✗ disagree** (you say what you
   actually see, e.g. `Debout_Assis`), or **? unsure**, with an optional comment. Keys `1` `2` `3` apply to the focused column
   (click a row to focus it). `▶` replays the segment from its start; `⇥` jumps to the next unreviewed segment.
   *Pause at end of focused segment* stops playback when a segment ends so you can judge it.
   Verdicts are stored in `output_dir\verdicts\<recording>.json` (re-judging keeps the previous verdict in `history`), are drawn
   as green/red/amber strips under the segments in the label tracks, and are listed in the **Label review** tab (progress per
   column, filter by verdict, CSV download). Each verdict stores the alignment a, b it was made under; verdicts made under a
   different alignment are flagged. The CSV labels are never modified.
7. **Save profile** writes `output_dir\profiles\<recording>.json`. It holds the pairing status, timestamp column/unit/reference with provenance, the channel mapping (with a *verified* checkbox), label roles, FPS and its provenance, a, b, anchors, status and history. **Download / Import profile** round-trips this JSON; an import warns if the recording ID or CSV SHA-256 differs.
8. **Tabs**
   - **Channels**: availability matrix per group.
   - **Mapping & time**: import options; timestamp column, unit and reference; editable channel mapping; label-column roles.
   - **Labels**: unique values with rows, duration and segment counts; overlap and unlabeled intervals.
   - **Quality**: timestamp regularity, effective rate, zero / missing / constant runs with thresholds.
   - **Raw values**: the original cell text at the current sample.
   - **Notes**: notes linked to a time range, saved to `output_dir\notes`.
   - **Export**: a selected range goes to a new folder under `output_dir\exports` containing `rows.csv` (two derived columns, then the original header and lines copied verbatim), `label_segments.json`, `notes.json`, `channel_mapping.json`, `alignment.json` and `manifest.json`.
   - **External labels**: an optional separate label file. You state intervals or points, the unit and the clock; points are never forward-filled.
   - **Recording info**: CSV SHA-256 and MP4 container metadata.

## What the app does not do (by design)

- It never modifies, interpolates, smooths, resamples or re-zeros source data, and never writes inside a data root (enforced in `config.py`).
- It never infers alignment from durations, filenames, FPS or row counts, never uses `Debut` automatically, and adds no hidden sensor shift.
- It never converts blank labels into a rest class, and does no annotation editing, automatic relabeling, training or concept thresholds.
- Units are shown as unknown. Timestamp unit `ms` is pre-selected for `ts_SQ` only because the contributor stated it; the provenance text says so.

## Known limitations

- **Frame stepping is approximate.** It seeks to `(n + 0.5)/fps` at the container's nominal FPS. The 2021 videos have an edit list (media starts at 0.0667 s) and the 2022 videos have a zero-duration first sample and B-frames, so the presented frame may differ by one. Not verified frame-exactly.
- 2021 nominal FPS is 10 000 000 / 333 333 = 30.00003 (as stated in the container); 2022 is 90 000 / 1 800 = 50.
- Video decoding depends on the browser. All eMOB videos are H.264 High 8-bit, which Edge/Chrome play natively. ffmpeg is **not** installed; the app only displays a suggested conversion command that writes a new file.
- MP4 metadata comes from a minimal box parser (no ffprobe), so only MP4/MOV-style containers are described.
- Opening a recording parses the whole CSV in Python (about 3–7 s for 30–40 MB); the result is cached for up to 3 recordings.
- Time lookup assumes ascending timestamps; non-monotonic files are flagged but not reordered.
- No synthetic demo mode in the UI (the real data paths exist). Tests generate synthetic files.
