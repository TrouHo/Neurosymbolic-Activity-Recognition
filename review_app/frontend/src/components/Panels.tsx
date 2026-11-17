import { useEffect, useState } from "react";
import { api, type ImportOpts, type Info, type LabelOverview, type Note, type Quality, type TsReport, type VideoMeta } from "../api";
import { displayValue, labelColor } from "../lib/colors";
import type { ExternalLabelConfig, LabelRole, Profile } from "../lib/profile";
import { nowIso } from "../lib/profile";
import { fmtSeconds, fmtTimestamp, type TsUnit } from "../lib/time";

const pct = (a: number, b: number) => (b ? `${((100 * a) / b).toFixed(1)}%` : "—");

// ---------------------------------------------------------------- Channels
export function ChannelsPanel({ info, profile, onToggle }: { info: Info; profile: Profile; onToggle: (gid: string, on: boolean) => void }) {
  const bySource = new Map<string, Info["groups"]>();
  for (const g of info.groups) {
    const key = g.modality === "XYZ" ? `${g.source} (XYZ position triplets — inspection only)` : g.source;
    bySource.set(key, [...(bySource.get(key) ?? []), g]);
  }
  const mapped = new Map(profile.channel_mapping.groups.map((g) => [g.group_id, g]));
  return (
    <div>
      <p className="muted small">Groups are parsed from the CSV header text. Plots show original names; units are unknown unless verified.
        Default selection = ACC/GYR groups containing at least one non-zero numeric value. MAG is excluded from the agreed model input but can be inspected.
        {!profile.channel_mapping.verified && <b> Channel mapping not yet verified by you (see Mapping tab).</b>}</p>
      {[...bySource.entries()].map(([src, gs]) => (
        <div key={src} className="chan-src">
          <div className="chan-src-name">{src}</div>
          <table className="tbl small">
            <thead><tr><th></th><th>placement</th><th>modality</th><th>columns</th><th>rows with non-zero number</th><th>rows all-axes zero</th><th>rows missing / NaN / text</th><th></th></tr></thead>
            <tbody>
              {gs.map((g) => {
                const m = mapped.get(g.id);
                const f = g.flags;
                return (
                  <tr key={g.id} className={f.no_nonzero_data ? "dim" : ""}>
                    <td><input type="checkbox" checked={profile.selected_groups.includes(g.id)} onChange={(e) => onToggle(g.id, e.target.checked)} /></td>
                    <td>{m?.placement ?? g.placement}{m?.edited && <span className="badge info">edited</span>}</td>
                    <td>{m?.modality ?? g.modality}</td>
                    <td className="mono">{Object.entries(g.axes).map(([a, c]) => `${a}:${c}`).join(" ")}</td>
                    <td>{pct(f.rows_with_nonzero_number, f.rows)}</td>
                    <td>{pct(f.rows_all_axes_zero, f.rows)}</td>
                    <td>{pct(f.rows_with_missing_or_nonnumeric, f.rows)}</td>
                    <td>{f.no_nonzero_data && <span className="badge bad">no non-zero data</span>}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------- Mapping & time
export function MappingPanel({ info, profile, tsReport, firstRaw, onChange }: {
  info: Info; profile: Profile; tsReport: TsReport | null; firstRaw: string; onChange: (p: Profile) => void;
}) {
  const numericCols = info.columns.filter((c) => c.kind === "numeric");
  const ts = profile.timestamp;
  const setGroup = (gid: string, k: "source" | "placement" | "modality", v: string) =>
    onChange({ ...profile, channel_mapping: { ...profile.channel_mapping, verified: false, verified_at: null,
      groups: profile.channel_mapping.groups.map((g) => g.group_id === gid ? { ...g, [k]: v, edited: true } : g) } });
  return (
    <div className="mapping">
      <h4>Import options</h4>
      <div className="row wrap small">
        <span>Detected: encoding <b>{info.csv.encoding}</b>, delimiter <b>{JSON.stringify(info.csv.delimiter)}</b>, decimal <b>{info.csv.decimal}</b>.</span>
        <label>encoding <select value={profile.import_options.encoding} onChange={(e) => onChange({ ...profile, import_options: { ...profile.import_options, encoding: e.target.value } })}>
          <option value="">auto</option><option>utf-8-sig</option><option>cp1252</option><option>latin-1</option></select></label>
        <label>delimiter <select value={profile.import_options.delimiter} onChange={(e) => onChange({ ...profile, import_options: { ...profile.import_options, delimiter: e.target.value } })}>
          <option value="">auto</option><option value=",">,</option><option value=";">;</option><option value={"\t"}>tab</option></select></label>
        <label>decimal <select value={profile.import_options.decimal} onChange={(e) => onChange({ ...profile, import_options: { ...profile.import_options, decimal: e.target.value } })}>
          <option value=".">.</option><option value=",">,</option></select></label>
        <span className="muted">Changing these reloads the CSV (source is never modified).</span>
      </div>

      <h4>Master timestamp</h4>
      <div className="row wrap">
        <label>column <select value={ts.column} onChange={(e) => { const c = Number(e.target.value); onChange({ ...profile, timestamp: { ...ts, column: c, column_name: info.columns[c].name, reference_provenance: "Default: first row of the timestamp column." } }); }}>
          {numericCols.map((c) => <option key={c.index} value={c.index}>{c.index}: {c.name}</option>)}
        </select></label>
        <label>unit <select value={ts.unit} onChange={(e) => onChange({ ...profile, timestamp: { ...ts, unit: e.target.value as TsUnit, unit_provenance: "Selected by user" } })}>
          {["ms", "s", "us", "ns"].map((u) => <option key={u}>{u}</option>)}</select></label>
        <label>reference <input type="number" value={ts.reference_value} onChange={(e) => onChange({ ...profile, timestamp: { ...ts, reference_value: Number(e.target.value), reference_raw: e.target.value, reference_provenance: "Edited by user" } })} /></label>
      </div>
      <div className="muted small">Unit provenance: {ts.unit_provenance}<br />Reference provenance: {ts.reference_provenance} (first raw value: <span className="mono">{firstRaw}</span>)</div>
      {tsReport && <TimestampSummary r={tsReport} unit={ts.unit} />}

      <h4>Channel mapping</h4>
      <label className="row"><input type="checkbox" checked={profile.channel_mapping.verified}
        onChange={(e) => onChange({ ...profile, channel_mapping: { ...profile.channel_mapping, verified: e.target.checked, verified_at: e.target.checked ? nowIso() : null } })} />
        I checked this mapping against the file header (parsed values are a proposal until checked).</label>
      <table className="tbl small">
        <thead><tr><th>parsed from header</th><th>source</th><th>placement</th><th>modality</th><th>columns</th></tr></thead>
        <tbody>
          {profile.channel_mapping.groups.map((g) => (
            <tr key={g.group_id}>
              <td className="mono">{g.header}</td>
              {(["source", "placement", "modality"] as const).map((k) => (
                <td key={k}><input value={g[k]} onChange={(e) => setGroup(g.group_id, k, e.target.value)} /></td>
              ))}
              <td className="mono">{Object.entries(g.axes).map(([a, c]) => `${a}:${info.columns[c].name}@${c}`).join("  ")}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h4>Label columns</h4>
      <table className="tbl small">
        <thead><tr><th>column</th><th>role</th><th>proposal provenance</th></tr></thead>
        <tbody>
          {profile.label_mapping.map((l) => (
            <tr key={l.column}>
              <td>{l.column}: {l.name}</td>
              <td><select value={l.role} onChange={(e) => onChange({ ...profile, label_mapping: profile.label_mapping.map((x) => x.column === l.column ? { ...x, role: e.target.value as LabelRole, provenance: "Selected by user" } : x) })}>
                <option value="family">family track</option><option value="derived">derived track</option><option value="ignore">ignore</option></select></td>
              <td className="muted">{l.provenance}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function TimestampSummary({ r, unit }: { r: TsReport; unit: string }) {
  const bad = r.n_invalid || r.n_duplicates || r.n_nonmonotonic || r.gaps.length || r.n_fractional;
  return (
    <div className={`callout ${bad ? "warn" : "info"} small`}>
      rows {r.n}; first <span className="mono">{fmtTimestamp(r.first ?? NaN)}</span>, last <span className="mono">{fmtTimestamp(r.last ?? NaN)}</span> ({unit});
      median step {r.median_step}; effective rate {r.effective_rate_hz ? r.effective_rate_hz.toFixed(4) : "—"} Hz (if unit is {unit}).<br />
      step histogram: {r.step_histogram.map(([s, n]) => `${s}×${n}`).join(", ")}<br />
      invalid {r.n_invalid}; duplicates {r.n_duplicates}; non-monotonic {r.n_nonmonotonic}; gaps (&gt;1.5× median) {r.gaps.length}; fractional {r.n_fractional}.
      {bad ? " These are reported, not corrected." : ""}
    </div>
  );
}

// ---------------------------------------------------------------- Labels
export function LabelsPanel({ labels, toT, onSeekT }: { labels: LabelOverview | null; toT: (ts: number) => number; onSeekT: (t: number) => void }) {
  if (!labels) return <div className="muted">No label columns selected.</div>;
  const dur = (ivs: { start_ts: number; end_ts: number }[]) => ivs.reduce((s, i) => s + toT(i.end_ts) - toT(i.start_ts), 0);
  return (
    <div>
      <div className="callout info small">
        <div>Overlap: {labels.definitions.overlap} — {labels.overlap.length} intervals, {dur(labels.overlap).toFixed(1)} s.</div>
        <div>Unlabeled: {labels.definitions.unlabeled} — {labels.unlabeled.length} intervals, {dur(labels.unlabeled).toFixed(1)} s.</div>
        <div>{labels.definitions.sample_interval}</div>
      </div>
      <div className="label-cols">
        {Object.values(labels.columns).map((c) => (
          <div key={c.column} className="label-col">
            <h4>{c.name} <span className="badge info">{c.role}</span></h4>
            <table className="tbl small">
              <thead><tr><th>value (raw)</th><th>rows</th><th>duration s</th><th>segments</th></tr></thead>
              <tbody>
                {c.summary.map((s) => (
                  <tr key={s.value}>
                    <td><span className="swatch" style={{ background: labelColor(s.value) ?? "transparent" }} />{displayValue(s.value)}</td>
                    <td>{s.rows}</td><td>{(s.duration_ms / 1000).toFixed(1)}</td><td>{s.segments}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>
      <h4>Unlabeled intervals (first 200)</h4>
      <IntervalList ivs={labels.unlabeled.slice(0, 200)} toT={toT} onSeekT={onSeekT} />
      <h4>Overlap intervals (first 200)</h4>
      <IntervalList ivs={labels.overlap.slice(0, 200)} toT={toT} onSeekT={onSeekT} />
    </div>
  );
}

function IntervalList({ ivs, toT, onSeekT }: { ivs: { start_row: number; end_row: number; start_ts: number; end_ts: number }[]; toT: (ts: number) => number; onSeekT: (t: number) => void }) {
  return (
    <div className="chips">
      {ivs.map((i) => (
        <button key={i.start_row} className="chip" onClick={() => onSeekT(toT(i.start_ts))} title={`rows ${i.start_row}–${i.end_row - 1}`}>
          {fmtSeconds(toT(i.start_ts), 2)} · {(toT(i.end_ts) - toT(i.start_ts)).toFixed(2)} s
        </button>
      ))}
      {ivs.length === 0 && <span className="muted">none</span>}
    </div>
  );
}

// ---------------------------------------------------------------- Quality
export function QualityPanel({ rid, profile, opts, toT, onZoom }: {
  rid: string; profile: Profile; opts: ImportOpts; toT: (ts: number) => number; onZoom: (t0: number, t1: number) => void;
}) {
  const [minLen, setMinLen] = useState(10);
  const [constLen, setConstLen] = useState(25);
  const [q, setQ] = useState<Quality | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const run = () => {
    setBusy(true); setErr(null);
    api.quality(rid, profile.timestamp.column, profile.timestamp.unit, minLen, constLen, opts)
      .then(setQ).catch((e) => setErr(String(e))).finally(() => setBusy(false));
  };
  useEffect(() => { setQ(null); }, [rid]);
  return (
    <div>
      <div className="row wrap">
        <label>min run length (samples) <input type="number" min={1} value={minLen} onChange={(e) => setMinLen(Math.max(1, Number(e.target.value)))} /></label>
        <label>constant-value run min (samples) <input type="number" min={2} value={constLen} onChange={(e) => setConstLen(Math.max(2, Number(e.target.value)))} /></label>
        <button onClick={run} disabled={busy}>{busy ? "Computing…" : q ? "Recompute" : "Compute quality report"}</button>
      </div>
      {err && <div className="callout bad">{err}</div>}
      {q && (
        <>
          <TimestampSummary r={q.timestamp} unit={profile.timestamp.unit} />
          <div className="muted small">{q.notes.join(" ")}</div>
          <table className="tbl small">
            <thead><tr><th>group</th><th>all-axes-zero runs ≥{q.thresholds.min_run_samples}</th><th>missing/NaN/text runs ≥{q.thresholds.min_run_samples}</th><th>constant non-zero runs ≥{q.thresholds.constant_min_samples} (x/y/z)</th><th>first runs (click to zoom)</th></tr></thead>
            <tbody>
              {q.groups.map((g) => (
                <tr key={g.id} className={g.flags.no_nonzero_data ? "dim" : ""}>
                  <td>{g.source} · {g.placement} · {g.modality}</td>
                  <td>{g.runs.all_axes_zero.count} ({g.runs.all_axes_zero.rows} rows)</td>
                  <td>{g.runs.missing_or_nonnumeric.count} ({g.runs.missing_or_nonnumeric.rows} rows)</td>
                  <td>{Object.entries(g.constant_runs).map(([a, r]) => `${a}:${r.count}`).join(" ")}</td>
                  <td className="chips">
                    {[...g.runs.all_axes_zero.runs.slice(0, 4).map((r) => ({ ...r, k: "zero" })), ...g.runs.missing_or_nonnumeric.runs.slice(0, 4).map((r) => ({ ...r, k: "missing" }))].map((r) => (
                      <button key={`${r.k}${r.start_row}`} className={`chip ${r.k}`} onClick={() => onZoom(toT(r.start_ts) - 1, toT(r.last_ts) + 1)}>
                        {r.k} {fmtSeconds(toT(r.start_ts), 1)} ({r.end_row - r.start_row})
                      </button>
                    ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- Raw values at the current sample
export function RawRowPanel({ rid, row, opts, auto }: { rid: string; row: number | null; opts: ImportOpts; auto: boolean }) {
  const [data, setData] = useState<Awaited<ReturnType<typeof api.row>> | null>(null);
  const [filter, setFilter] = useState("");
  useEffect(() => {
    if (row == null || !auto) return;
    const h = setTimeout(() => api.row(rid, row, opts).then(setData).catch(() => setData(null)), 150);
    return () => clearTimeout(h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rid, row, auto]);
  return (
    <div>
      <div className="row wrap">
        <span>Raw cell text of the current sample (updates while paused). Row {data?.row ?? "—"} (file line {data?.file_line ?? "—"}).</span>
        <button disabled={row == null} onClick={() => row != null && api.row(rid, row, opts).then(setData)}>Refresh</button>
        <input placeholder="filter columns" value={filter} onChange={(e) => setFilter(e.target.value)} />
      </div>
      {data && (
        <div className="raw-grid">
          {data.cells.filter((c) => !filter || c.name.toLowerCase().includes(filter.toLowerCase())).map((c) => (
            <div key={c.index} className={`raw-cell ${c.class}`} title={c.class}>
              <span className="muted">{c.index}:{c.name}</span> <span className="mono">{c.raw === "" ? "(empty)" : c.raw}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- Notes
export function NotesPanel({ notes, onAdd, onDelete, onSeekT, rangeText, canAdd }: {
  notes: Note[]; onAdd: (text: string) => void; onDelete: (id: string) => void; onSeekT: (t: number) => void; rangeText: string; canAdd: boolean;
}) {
  const [text, setText] = useState("");
  const download = () => {
    const cols: (keyof Note)[] = ["id", "created", "recording_id", "row_start", "row_end", "ts_start", "ts_end", "t_csv_start", "t_csv_end", "t_video_start", "t_video_end", "csv_sha256", "text"];
    const esc = (v: unknown) => `"${String(v ?? "").replace(/"/g, '""')}"`;
    const csv = [cols.join(","), ...notes.map((n) => cols.map((c) => esc(n[c])).join(","))].join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = `${notes[0]?.recording_id ?? "notes"}_review_notes.csv`;
    a.click();
  };
  return (
    <div>
      <div className="row wrap">
        <textarea rows={2} placeholder="Review note (observation, question, suspected issue)…" value={text} onChange={(e) => setText(e.target.value)} />
        <div>
          <div className="small muted">Linked to: {rangeText}</div>
          <button disabled={!text.trim() || !canAdd} onClick={() => { onAdd(text.trim()); setText(""); }}>Add note</button>
          <button disabled={!notes.length} onClick={download}>Download notes CSV</button>
        </div>
      </div>
      <table className="tbl small">
        <thead><tr><th>created</th><th>t_csv</th><th>rows</th><th>t_video (at save)</th><th>note</th><th></th></tr></thead>
        <tbody>
          {notes.map((n) => (
            <tr key={n.id}>
              <td className="mono">{n.created.slice(0, 19)}</td>
              <td><button className="link" onClick={() => onSeekT(n.t_csv_start)}>{fmtSeconds(n.t_csv_start, 2)}{n.t_csv_end !== n.t_csv_start && ` – ${fmtSeconds(n.t_csv_end, 2)}`}</button></td>
              <td>{n.row_start}–{n.row_end - 1}</td>
              <td>{n.t_video_start == null ? "—" : fmtSeconds(n.t_video_start, 2)}</td>
              <td>{n.text}</td>
              <td><button className="link" onClick={() => window.confirm("Delete this note?") && onDelete(n.id)}>×</button></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------- External labels
export function ExternalLabelsPanel({ profile, onChange }: { profile: Profile; onChange: (p: Profile) => void }) {
  const [path, setPath] = useState("");
  const [prev, setPrev] = useState<Awaited<ReturnType<typeof api.extPreview>> | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [cfg, setCfg] = useState<Omit<ExternalLabelConfig, "path">>({ start_col: 0, end_col: null, label_col: 1, family_col: null, time_unit: "ms", time_reference: "absolute" });
  const colSel = (k: "start_col" | "end_col" | "label_col" | "family_col", optional: boolean) => (
    <select value={cfg[k] ?? ""} onChange={(e) => setCfg({ ...cfg, [k]: e.target.value === "" ? null : Number(e.target.value) })}>
      {optional && <option value="">(none)</option>}
      {prev?.header.map((h, i) => <option key={i} value={i}>{i}: {h}</option>)}
    </select>
  );
  return (
    <div>
      <p className="muted small">Optional separate annotation file. You must state whether it stores intervals (start+end) or point events (start only),
        the time unit, and which clock it uses. Point events are drawn as ticks and never forward-filled. The file is read-only.</p>
      <div className="row wrap">
        <input className="wide" placeholder="full path inside a data root, e.g. D:\Projects\EmoB\data\raw\...\labels.csv" value={path} onChange={(e) => setPath(e.target.value)} />
        <button onClick={() => api.extPreview(path).then((p) => { setPrev(p); setErr(null); }).catch((e) => setErr(String(e)))}>Preview</button>
      </div>
      {err && <div className="callout bad">{err}</div>}
      {prev && (
        <>
          <div className="small">encoding {prev.encoding}, delimiter {JSON.stringify(prev.delimiter)}, {prev.n_rows} rows</div>
          <div className="table-scroll"><table className="tbl small"><thead><tr>{prev.header.map((h, i) => <th key={i}>{i}: {h}</th>)}</tr></thead>
            <tbody>{prev.rows.slice(0, 8).map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>)}</tbody></table></div>
          <div className="row wrap">
            <label>start {colSel("start_col", false)}</label>
            <label>end {colSel("end_col", true)}</label>
            <label>label {colSel("label_col", false)}</label>
            <label>family {colSel("family_col", true)}</label>
            <label>unit <select value={cfg.time_unit} onChange={(e) => setCfg({ ...cfg, time_unit: e.target.value as TsUnit })}>{["ms", "s", "us", "ns"].map((u) => <option key={u}>{u}</option>)}</select></label>
            <label>clock <select value={cfg.time_reference} onChange={(e) => setCfg({ ...cfg, time_reference: e.target.value as ExternalLabelConfig["time_reference"] })}>
              <option value="absolute">same clock as CSV master timestamp</option>
              <option value="csv_relative">relative to CSV reference timestamp</option>
              <option value="video">video media time (moves with alignment)</option>
            </select></label>
            <button onClick={() => onChange({ ...profile, external_labels: [...profile.external_labels, { path: prev.path, ...cfg }] })}>Add as track(s)</button>
          </div>
        </>
      )}
      {profile.external_labels.length > 0 && (
        <table className="tbl small"><thead><tr><th>file</th><th>kind</th><th>unit</th><th>clock</th><th></th></tr></thead>
          <tbody>{profile.external_labels.map((x, i) => (
            <tr key={i}><td className="mono">{x.path}</td><td>{x.end_col == null ? "points" : "intervals"}</td><td>{x.time_unit}</td><td>{x.time_reference}</td>
              <td><button className="link" onClick={() => onChange({ ...profile, external_labels: profile.external_labels.filter((_, j) => j !== i) })}>remove</button></td></tr>
          ))}</tbody></table>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- Recording info
export function InfoPanel({ info, meta, profilePath }: { info: Info; meta: VideoMeta | null; profilePath: string }) {
  const vt = meta?.video_track;
  return (
    <div className="info-grid small">
      <div>
        <h4>CSV</h4>
        <div className="mono">{info.csv.path}</div>
        <div>{info.csv.rows} rows × {info.csv.columns} columns; {info.csv.size.toLocaleString()} bytes</div>
        <div>SHA-256 <span className="mono">{info.csv.sha256}</span></div>
        <div>rows with wrong field count: {info.csv.n_bad_rows}; physical lines = records: {String(info.csv.lines_match_records)}</div>
        <div>{info.csv.row_index_convention}</div>
      </div>
      <div>
        <h4>Video (container metadata)</h4>
        {meta ? (meta.ok ? (
          <>
            <div className="mono">{meta.path}</div>
            <div>codec {vt?.codec}, coded {vt?.coded_wh?.join("×")}, duration {meta.movie_duration_s?.toFixed(3)} s, samples {vt?.sample_count}</div>
            <div>nominal FPS {vt?.nominal_fps?.toFixed(6) ?? "not constant / unknown"} — {vt?.fps_provenance ?? "no constant sample delta"}</div>
            <div>B-frames/composition offsets: {String(!!vt?.has_composition_offsets)}; edit list: {vt?.edit_list?.map((e) => e.media_time_s != null ? `media starts at ${e.media_time_s.toFixed(4)} s` : `media_time ${e.media_time_track_units}`).join(", ") ?? "none"}; audio: {String(meta.has_audio)}</div>
          </>
        ) : <div className="callout bad">{meta.error}</div>) : <div className="muted">No video.</div>}
      </div>
      <div>
        <h4>Review profile</h4>
        <div className="mono">{profilePath}</div>
      </div>
    </div>
  );
}
