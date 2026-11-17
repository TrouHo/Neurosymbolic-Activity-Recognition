// Reviewer verdicts on label segments. A segment = one label column + [start_row, end_row) + raw value.
import type { Segment } from "../api";

export type VerdictKind = "agree" | "disagree" | "unsure";

export interface Verdict {
  id: string;
  created: string;
  updated: string;
  recording_id: string;
  column: number;
  column_name: string;
  start_row: number;
  end_row: number;
  label_value: string;       // raw label string in the CSV ("" = empty)
  verdict: VerdictKind;
  observed: string;          // what the reviewer sees in the video (a proposal, never written to the CSV)
  comment: string;
  ts_start: number;
  ts_end: number;
  t_csv_start: number;
  t_csv_end: number;
  t_video_start: number | null;
  t_video_end: number | null;
  csv_sha256: string;
  alignment_snapshot: { a: number; b: number; status: string; method: string };
  history?: unknown[];
}

export type VerdictInput = Omit<Verdict, "id" | "created" | "updated" | "recording_id" | "history">;

export const VERDICT_COLORS: Record<VerdictKind, string> = { agree: "#2e9e44", disagree: "#d62728", unsure: "#e6a700" };
export const VERDICT_LABELS: Record<VerdictKind, string> = { agree: "✓ agree", disagree: "✗ disagree", unsure: "? unsure" };

export const segKey = (column: number, s: Pick<Segment, "start_row" | "end_row" | "value">) =>
  `${column}:${s.start_row}:${s.end_row}:${s.value}`;
export const verdictKey = (v: Pick<Verdict, "column" | "start_row" | "end_row" | "label_value">) =>
  `${v.column}:${v.start_row}:${v.end_row}:${v.label_value}`;

/** Segment containing `row` (segments sorted by start_row, contiguous). */
export function segmentAtRow(segs: Segment[], row: number): Segment | null {
  let lo = 0, hi = segs.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const s = segs[mid];
    if (row < s.start_row) hi = mid - 1;
    else if (row >= s.end_row) lo = mid + 1;
    else return s;
  }
  return null;
}

/** First segment starting after `row` whose value is not skipped and (optionally) not yet reviewed. */
export function nextSegment(segs: Segment[], row: number, column: number, skip: Set<string>,
  reviewed?: Map<string, Verdict>): Segment | null {
  for (const s of segs) {
    if (s.start_row <= row) continue;
    if (skip.has(s.value)) continue;
    if (reviewed && reviewed.has(segKey(column, s))) continue;
    return s;
  }
  return null;
}

export interface Progress { total: number; reviewed: number; agree: number; disagree: number; unsure: number }

export function progress(segs: Segment[], column: number, skip: Set<string>, verdicts: Map<string, Verdict>): Progress {
  const p: Progress = { total: 0, reviewed: 0, agree: 0, disagree: 0, unsure: 0 };
  for (const s of segs) {
    if (skip.has(s.value)) continue;
    p.total++;
    const v = verdicts.get(segKey(column, s));
    if (v) { p.reviewed++; p[v.verdict]++; }
  }
  return p;
}

/** True when the verdict was made under a different alignment than the current one. */
export function madeUnderOtherAlignment(v: Verdict, a: number, b: number): boolean {
  return Math.abs(v.alignment_snapshot.a - a) > 1e-9 || Math.abs(v.alignment_snapshot.b - b) > 1e-6;
}
