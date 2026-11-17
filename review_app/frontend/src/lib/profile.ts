// Per-recording review profile (saved as JSON in the output dir; re-importable).
import type { Anchor, TsUnit } from "./time";

export type AlignStatus = "unverified" | "provisional" | "confirmed";
export type LabelRole = "family" | "derived" | "ignore";

export interface GroupMapping {
  group_id: string;
  header: string;          // original header text the group was parsed from
  source: string;
  placement: string;
  modality: string;
  axes: Record<string, number>;
  edited: boolean;         // true if the user changed the parsed values
}

export interface ExternalLabelConfig {
  path: string;
  start_col: number;
  end_col: number | null;     // null = point events (never forward-filled)
  label_col: number;
  family_col: number | null;
  time_unit: TsUnit;
  time_reference: "absolute" | "csv_relative" | "video";
}

export interface HistoryEntry {
  at: string;
  action: string;
  a: number;
  b: number;
  detail?: string;
}

export interface Profile {
  schema_version: 1;
  recording_id: string;
  csv: { path: string; sha256: string; size: number };
  video: { path: string | null };
  import_options: { encoding: string; delimiter: string; decimal: string };
  pairing: { status: "proposed" | "confirmed" | "none"; rule: string | null; confirmed_at: string | null };
  timestamp: {
    column: number;
    column_name: string;
    unit: TsUnit;
    unit_provenance: string;
    reference_value: number;
    reference_raw: string;
    reference_provenance: string;
  };
  channel_mapping: { groups: GroupMapping[]; verified: boolean; verified_at: string | null };
  selected_groups: string[];
  label_mapping: { column: number; name: string; role: LabelRole; provenance: string }[];
  video_timing: { fps: number | null; fps_provenance: string; frame_step: "approximate" };
  alignment: {
    model: "t_csv = a * t_video + b";
    a: number;
    b: number;
    status: AlignStatus;
    method: "default" | "manual" | "one_anchor" | "two_anchor" | "imported";
    anchors: Anchor[];
    history: HistoryEntry[];
  };
  external_labels: ExternalLabelConfig[];
  display_transforms: never[];   // none implemented; kept explicit so exports state it
  saved_at?: string;
}

export interface InfoLike {
  recording: { id: string; video_path: string | null; pairing_rule: string | null };
  csv: { path: string; sha256: string; size: number; encoding: string; delimiter: string; decimal: string };
  columns: { index: number; name: string }[];
  groups: { id: string; header: string; source: string; placement: string; modality: string; axes: Record<string, number>;
    flags: { no_nonzero_data: boolean } }[];
  proposals: {
    timestamp: { column: number; unit: TsUnit | null; provenance: string } | null;
    labels: { column: number; name: string; role: LabelRole; provenance: string }[];
  };
}

export const nowIso = () => new Date().toISOString();

export function defaultSelectedGroups(info: InfoLike): string[] {
  return info.groups
    .filter((g) => ["ACC", "GYR"].includes(g.modality.toUpperCase()) && !g.flags.no_nonzero_data)
    .map((g) => g.id);
}

export function defaultProfile(info: InfoLike, firstTs: number, firstTsRaw: string, fps: number | null,
  fpsProvenance: string): Profile {
  const tp = info.proposals.timestamp;
  return {
    schema_version: 1,
    recording_id: info.recording.id,
    csv: { path: info.csv.path, sha256: info.csv.sha256, size: info.csv.size },
    video: { path: info.recording.video_path },
    import_options: { encoding: "", delimiter: "", decimal: "." },
    pairing: { status: info.recording.video_path ? "proposed" : "none", rule: info.recording.pairing_rule, confirmed_at: null },
    timestamp: {
      column: tp?.column ?? 0,
      column_name: info.columns[tp?.column ?? 0]?.name ?? "",
      unit: tp?.unit ?? "ms",
      unit_provenance: tp?.unit ? tp.provenance : "No unit proposal; 'ms' preselected as a placeholder. Select the correct unit.",
      reference_value: firstTs,
      reference_raw: firstTsRaw,
      reference_provenance: "Default: first row of the timestamp column.",
    },
    channel_mapping: {
      groups: info.groups.map((g) => ({ group_id: g.id, header: g.header, source: g.source, placement: g.placement,
        modality: g.modality, axes: g.axes, edited: false })),
      verified: false,
      verified_at: null,
    },
    selected_groups: defaultSelectedGroups(info),
    label_mapping: info.proposals.labels.map((l) => ({ ...l })),
    video_timing: { fps, fps_provenance: fpsProvenance, frame_step: "approximate" },
    alignment: { model: "t_csv = a * t_video + b", a: 1, b: 0, status: "unverified", method: "default", anchors: [],
      history: [{ at: nowIso(), action: "default identity model (unverified)", a: 1, b: 0 }] },
    external_labels: [],
    display_transforms: [],
  };
}

export interface ImportCheck {
  ok: boolean;
  errors: string[];
  warnings: string[];
}

/** Validate an imported profile against the currently open recording. */
export function checkImport(p: unknown, current: { recording_id: string; sha256: string }): ImportCheck {
  const errors: string[] = [];
  const warnings: string[] = [];
  const o = p as Partial<Profile>;
  if (!o || typeof o !== "object") return { ok: false, errors: ["Not a JSON object"], warnings };
  if (o.schema_version !== 1) errors.push(`Unsupported schema_version: ${String(o.schema_version)}`);
  const al = o.alignment;
  if (!al || typeof al.a !== "number" || typeof al.b !== "number" || !(al.a > 0)) errors.push("alignment.a / alignment.b missing or invalid");
  if (!o.timestamp || typeof o.timestamp.column !== "number" || typeof o.timestamp.reference_value !== "number")
    errors.push("timestamp.column / timestamp.reference_value missing");
  if (o.recording_id && o.recording_id !== current.recording_id)
    warnings.push(`Profile was saved for recording '${o.recording_id}', not '${current.recording_id}'.`);
  if (o.csv?.sha256 && o.csv.sha256 !== current.sha256)
    warnings.push("CSV SHA-256 differs from the profile: the source file changed or this is a different file.");
  return { ok: errors.length === 0, errors, warnings };
}
