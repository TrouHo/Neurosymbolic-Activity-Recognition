// Pure time / alignment functions. No hidden shifts, no clamping to endpoints.
//
//   t_csv   = (timestamp - reference) * unitToMs / 1000           (seconds, CSV-relative)
//   t_csv   = a * t_video + b                                       (alignment model)
//   t_video = (t_csv - b) / a
//
// Positive b means video time 0 maps to a LATER CSV time (the CSV started earlier than the video).

export type TsUnit = "ms" | "s" | "us" | "ns";
export const UNIT_TO_MS: Record<TsUnit, number> = { ms: 1, s: 1000, us: 1e-3, ns: 1e-6 };

export interface Model {
  a: number;
  b: number;
}

export function tCsvFromTimestamp(ts: number, reference: number, unit: TsUnit): number {
  return ((ts - reference) * UNIT_TO_MS[unit]) / 1000;
}

export function timestampFromTCsv(tCsv: number, reference: number, unit: TsUnit): number {
  return reference + (tCsv * 1000) / UNIT_TO_MS[unit];
}

export function csvFromVideo(tVideo: number, m: Model): number {
  return m.a * tVideo + m.b;
}

export function videoFromCsv(tCsv: number, m: Model): number {
  if (!(m.a > 0) || !Number.isFinite(m.a)) throw new Error(`Invalid scale a=${m.a}; a must be > 0`);
  return (tCsv - m.b) / m.a;
}

export interface Anchor {
  id: string;
  tVideo: number;
  tCsv: number;
  kind: "signal" | "annotation" | "other";
  description: string;
  created: string;
}

/** One anchor: keep a, solve b so the anchor maps exactly. */
export function oneAnchor(anchor: Pick<Anchor, "tVideo" | "tCsv">, a: number): Model {
  if (!(a > 0)) throw new Error("a must be > 0");
  return { a, b: anchor.tCsv - a * anchor.tVideo };
}

export interface FitResult {
  model: Model | null;
  warnings: string[];
  errors: string[];
}

/** Two anchors: solve a and b exactly. Rejects invalid ordering; warns on unusual scale. */
export function twoAnchor(p: Pick<Anchor, "tVideo" | "tCsv">, q: Pick<Anchor, "tVideo" | "tCsv">,
  scaleWarnTolerance = 0.01): FitResult {
  const warnings: string[] = [];
  const errors: string[] = [];
  const [p1, p2] = p.tVideo <= q.tVideo ? [p, q] : [q, p];
  const dv = p2.tVideo - p1.tVideo;
  const dc = p2.tCsv - p1.tCsv;
  if (dv === 0) errors.push("Both anchors have the same video time.");
  else if (dc === 0) errors.push("Both anchors have the same CSV time.");
  else if (dc < 0) errors.push("Invalid ordering: the later video anchor maps to an earlier CSV time (a would be negative).");
  if (errors.length) return { model: null, warnings, errors };
  const a = dc / dv;
  const b = p1.tCsv - a * p1.tVideo;
  if (Math.abs(a - 1) > scaleWarnTolerance)
    warnings.push(`Scale a = ${a.toFixed(6)} differs from 1 by more than ${scaleWarnTolerance}; check anchors or unit.`);
  if (dv < 10) warnings.push(`Anchors are only ${dv.toFixed(2)} s apart; the drift estimate is sensitive to anchor error.`);
  return { model: { a, b }, warnings, errors };
}

export function residuals(anchors: Pick<Anchor, "tVideo" | "tCsv">[], m: Model): number[] {
  return anchors.map((an) => csvFromVideo(an.tVideo, m) - an.tCsv);
}

export type Coverage = "inside" | "before" | "after" | "invalid";

/** Coverage of a CSV-relative time against samples t[0..n-1]; sample i covers [t_i, t_{i+1}),
 * the last sample covers [t_last, t_last + step). */
export function csvCoverage(tCsv: number, first: number, last: number, step: number): Coverage {
  if (!Number.isFinite(tCsv)) return "invalid";
  if (tCsv < first) return "before";
  if (tCsv >= last + step) return "after";
  return "inside";
}

export function videoCoverage(tVideo: number, duration: number): Coverage {
  if (!Number.isFinite(tVideo)) return "invalid";
  if (tVideo < 0) return "before";
  if (tVideo > duration) return "after";
  return "inside";
}

/** Index of the sample whose interval [t_i, t_{i+1}) contains t (floor), or -1 if t < t[0].
 * t must be ascending (NaN entries are not supported here; callers check monotonicity). */
export function sampleIndexAt(t: ArrayLike<number>, x: number): number {
  let lo = 0;
  let hi = t.length - 1;
  if (t.length === 0 || x < t[0]) return -1;
  if (x >= t[hi]) return hi;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (t[mid] <= x) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/** Nearest sample by absolute time difference. */
export function nearestIndex(t: ArrayLike<number>, x: number): number {
  const i = sampleIndexAt(t, x);
  if (i < 0) return t.length ? 0 : -1;
  if (i + 1 < t.length && Math.abs(t[i + 1] - x) < Math.abs(x - t[i])) return i + 1;
  return i;
}

// ---------------------------------------------------------------- video frames (approximate)

/** Approximate frame index for constant-FPS video: frame n covers [n/fps, (n+1)/fps). */
export function approxFrameIndex(tVideo: number, fps: number): number {
  return Math.floor(tVideo * fps + 1e-6);
}

/** Seek target for frame n: the middle of its nominal interval (avoids landing on a boundary). */
export function frameSeekTime(n: number, fps: number): number {
  return (n + 0.5) / fps;
}

export function fmtSeconds(t: number, digits = 3): string {
  if (!Number.isFinite(t)) return "—";
  const sign = t < 0 ? "-" : "";
  const a = Math.abs(t);
  const m = Math.floor(a / 60);
  const s = a - m * 60;
  return `${sign}${m}:${s.toFixed(digits).padStart(digits + 3, "0")}`;
}

/** Integer-looking timestamps are printed without exponent or rounding. */
export function fmtTimestamp(ts: number): string {
  if (!Number.isFinite(ts)) return "—";
  return Number.isInteger(ts) ? ts.toFixed(0) : String(ts);
}
