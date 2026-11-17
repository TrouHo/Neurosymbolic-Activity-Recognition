// Thin client for the local FastAPI backend.
import type { Profile } from "./lib/profile";
import type { TsUnit } from "./lib/time";
import type { Verdict, VerdictInput } from "./lib/verdicts";

export interface ImportOpts { encoding: string; delimiter: string; decimal: string }

async function j<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { msg = (await res.json()).detail ?? msg; } catch { /* keep status text */ }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

const q = (o: Record<string, string | number | undefined>) =>
  new URLSearchParams(Object.entries(o).filter(([, v]) => v !== undefined && v !== "").map(([k, v]) => [k, String(v)])).toString();

export interface Summary {
  rows: number;
  label_columns_entirely_empty: string[];
  groups: { source: string; placement: string; modality: string; no_nonzero_data: boolean }[];
}

export interface RecordingItem {
  id: string; csv_path: string; video_path: string | null; csv_size: number | null;
  pairing_rule: string | null; origin: string; summary: Summary | null;
}

export interface ColumnInfo { index: number; name: string; kind: string; counts: Record<string, number> }
export interface GroupFlags {
  rows: number; rows_with_nonzero_number: number; rows_all_axes_zero: number;
  rows_with_missing_or_nonnumeric: number; no_nonzero_data: boolean;
}
export interface GroupInfo {
  id: string; source: string; placement: string; modality: string; header: string; ts_col: number | null;
  axes: Record<string, number>; provenance: string; flags: GroupFlags;
}
export interface Info {
  recording: RecordingItem;
  csv: { path: string; size: number; sha256: string; encoding: string; delimiter: string; decimal: string; rows: number;
    columns: number; bad_rows: number[]; n_bad_rows: number; lines_match_records: boolean; row_index_convention: string };
  columns: ColumnInfo[];
  class_names: string[];
  groups: GroupInfo[];
  proposals: {
    timestamp: { column: number; unit: TsUnit | null; provenance: string } | null;
    labels: { column: number; name: string; role: "family" | "derived"; provenance: string }[];
  };
}

export interface TsReport {
  column: number; n: number; first: number | null; last: number | null; n_invalid: number; invalid_rows: number[];
  n_fractional: number; n_duplicates: number; duplicate_rows: number[]; n_nonmonotonic: number;
  nonmonotonic_rows: number[]; median_step: number | null; step_histogram: [number, number][];
  gaps: { row: number; from: number; to: number; step: number }[]; effective_rate_hz: number | null;
}

export interface Segment {
  start_row: number; end_row: number; value: string; start_ts: number; end_ts: number;
  end_extrapolated: boolean; split_at_anomaly: boolean;
}
export interface Interval { start_row: number; end_row: number; start_ts: number; end_ts: number; end_extrapolated: boolean }
export interface LabelColumn {
  column: number; name: string; role: "family" | "derived"; segments: Segment[];
  summary: { value: string; rows: number; segments: number; duration_ms: number }[];
}
export interface LabelOverview {
  columns: Record<string, LabelColumn>; overlap: Interval[]; unlabeled: Interval[];
  definitions: Record<string, string>; family_columns: number[]; derived_columns: number[];
}

export interface RunList { count: number; rows: number; runs: { start_row: number; end_row: number; start_ts: number; last_ts: number }[] }
export interface Quality {
  timestamp: TsReport;
  groups: { id: string; source: string; placement: string; modality: string; header: string; flags: GroupFlags;
    runs: { all_axes_zero: RunList; missing_or_nonnumeric: RunList }; constant_runs: Record<string, RunList> }[];
  thresholds: { min_run_samples: number; constant_min_samples: number };
  notes: string[];
}

export interface VideoTrack {
  timescale?: number; duration_s?: number; codec?: string; coded_wh?: number[]; display_wh?: number[];
  sample_count?: number; nominal_fps?: number; fps_provenance?: string; zero_duration_samples?: number;
  has_composition_offsets?: boolean; edit_list?: { media_time_s?: number; media_time_track_units: number }[];
}
export interface VideoMeta { ok: boolean; error?: string; path: string; movie_duration_s?: number; video_track: VideoTrack | null; has_audio?: boolean }

export interface Note {
  id: string; created: string; recording_id: string; text: string;
  row_start: number; row_end: number; t_csv_start: number; t_csv_end: number;
  ts_start: number; ts_end: number; t_video_start: number | null; t_video_end: number | null;
  csv_sha256: string; alignment_snapshot: unknown;
}

export const api = {
  config: () => fetch("/api/config").then(j<{ data_roots: string[]; scan_dirs: string[]; output_dir: string }>),
  recordings: () => fetch("/api/recordings").then(j<RecordingItem[]>),
  summarize: (rid: string) => fetch(`/api/recordings/${rid}/summary`, { method: "POST" }).then(j<Summary>),
  addCustom: (csv_path: string, video_path: string | null) =>
    fetch("/api/recordings/custom", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ csv_path, video_path }) }).then(j<RecordingItem>),
  browse: (path?: string) => fetch(`/api/browse?${q({ path })}`).then(j<{ path: string | null; parent?: string | null;
    entries: { name: string; path: string; dir: boolean; size?: number | null }[] }>),
  info: (rid: string, o: ImportOpts) => fetch(`/api/rec/${rid}/info?${q({ ...o })}`).then(j<Info>),
  timestamps: (rid: string, col: number, unit: TsUnit, o: ImportOpts) =>
    fetch(`/api/rec/${rid}/timestamps?${q({ col, unit, ...o })}`).then(j<{ report: TsReport; first_raw: string; values: (number | null)[] }>),
  series: (rid: string, cols: number[], o: ImportOpts) =>
    fetch(`/api/rec/${rid}/series?${q({ cols: cols.join(","), ...o })}`).then(j<Record<string, (number | null)[]>>),
  row: (rid: string, row: number, o: ImportOpts) =>
    fetch(`/api/rec/${rid}/row/${row}?${q({ ...o })}`).then(j<{ row: number; file_line: number; cells: { index: number; name: string; raw: string; class: string }[] }>),
  labels: (rid: string, ts_col: number, family: number[], derived: number[], o: ImportOpts) =>
    fetch(`/api/rec/${rid}/labels?${q({ ts_col, family: family.join(","), derived: derived.join(","), ...o })}`).then(j<LabelOverview>),
  quality: (rid: string, ts_col: number, unit: TsUnit, min_len: number, const_min_len: number, o: ImportOpts) =>
    fetch(`/api/rec/${rid}/quality?${q({ ts_col, unit, min_len, const_min_len, ...o })}`).then(j<Quality>),
  videoMeta: (rid: string) => fetch(`/api/rec/${rid}/video_meta`).then(j<VideoMeta>),
  videoUrl: (rid: string) => `/api/rec/${rid}/video`,
  getProfile: (rid: string) => fetch(`/api/rec/${rid}/profile`).then(j<{ profile: Profile | null; path: string }>),
  putProfile: (rid: string, p: Profile) => fetch(`/api/rec/${rid}/profile`, { method: "PUT",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(p) }).then(j<{ profile: Profile; path: string }>),
  notes: (rid: string) => fetch(`/api/rec/${rid}/notes`).then(j<Note[]>),
  addNote: (rid: string, n: Omit<Note, "id" | "created" | "recording_id">) => fetch(`/api/rec/${rid}/notes`, { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(n) }).then(j<Note>),
  deleteNote: (rid: string, nid: string) => fetch(`/api/rec/${rid}/notes/${nid}`, { method: "DELETE" }).then(j<{ ok: boolean }>),
  verdicts: (rid: string) => fetch(`/api/rec/${rid}/verdicts`).then(j<Verdict[]>),
  putVerdict: (rid: string, v: VerdictInput) => fetch(`/api/rec/${rid}/verdicts`, { method: "PUT",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(v) }).then(j<Verdict>),
  deleteVerdict: (rid: string, vid: string) => fetch(`/api/rec/${rid}/verdicts/${vid}`, { method: "DELETE" }).then(j<{ ok: boolean }>),
  exportRows: (rid: string, body: unknown, o: ImportOpts) => fetch(`/api/rec/${rid}/export?${q({ ...o })}`, { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(j<{ folder: string; files: string[] }>),
  extPreview: (path: string) => fetch("/api/external_labels/preview", { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path }) })
    .then(j<{ path: string; encoding: string; delimiter: string; header: string[]; rows: string[][]; n_rows: number }>),
  extLoad: (body: unknown) => fetch("/api/external_labels/load", { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
    .then(j<{ kind: "intervals" | "points"; header: string[]; items: { row: number; start_raw: string | null; end_raw: string | null;
      label: string | null; family: string | null; start: number | null; end: number | null }[] }>),
};
