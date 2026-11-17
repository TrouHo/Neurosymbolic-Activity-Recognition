import { useEffect, useState } from "react";
import type { LabelColumn, Segment } from "../api";
import { displayValue, labelColor } from "../lib/colors";
import { fmtSeconds } from "../lib/time";
import {
  madeUnderOtherAlignment, progress, VERDICT_COLORS, VERDICT_LABELS, type Verdict, type VerdictKind,
} from "../lib/verdicts";

export interface NowRow {
  col: LabelColumn;
  seg: Segment | null;
  verdict: Verdict | null;
}

interface NowProps {
  rows: NowRow[];
  csvRow: number | null;
  noDataText: string;
  alignStatus: string;
  focusCol: number | null;
  onFocus: (c: number) => void;
  toT: (ts: number) => number;
  skipEmpty: boolean;
  onSkipEmpty: (b: boolean) => void;
  pauseAtEnd: boolean;
  onPauseAtEnd: (b: boolean) => void;
  onPlaySegment: (s: Segment) => void;
  onNext: (c: LabelColumn) => void;
  onPause: () => void;
  onVerdict: (c: LabelColumn, s: Segment, kind: VerdictKind, observed: string, comment: string) => void;
}

type FormState = { col: LabelColumn; seg: Segment; kind: VerdictKind; observed: string; comment: string };

/** The labels of every column at the current video moment, with verdict buttons. */
export function LabelsNowPanel(p: NowProps) {
  const [form, setForm] = useState<FormState | null>(null);

  const open = (col: LabelColumn, seg: Segment, kind: VerdictKind, existing: Verdict | null) => {
    p.onPause();
    setForm({ col, seg, kind, observed: existing?.observed ?? "", comment: existing?.comment ?? "" });
  };
  const quick = (col: LabelColumn, seg: Segment, kind: VerdictKind, existing: Verdict | null) => {
    if (kind === "agree") p.onVerdict(col, seg, "agree", "", existing?.comment ?? "");
    else open(col, seg, kind, existing);
  };

  // Keys 1/2/3 = agree / disagree / unsure for the focused column's current segment.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(tag) || form) return;
      const kind = ({ "1": "agree", "2": "disagree", "3": "unsure" } as Record<string, VerdictKind>)[e.key];
      const r = p.rows.find((x) => x.col.column === p.focusCol);
      if (!kind || !r?.seg) return;
      e.preventDefault();
      quick(r.col, r.seg, kind, r.verdict);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  return (
    <div className="labels-now">
      <div className="row space">
        <b>Labels now {p.csvRow != null ? <span className="muted small">(CSV row {p.csvRow})</span> : null}</b>
        <span className="small">
          <label><input type="checkbox" checked={p.pauseAtEnd} onChange={(e) => p.onPauseAtEnd(e.target.checked)} /> pause at end of focused segment</label>
          {" "}
          <label><input type="checkbox" checked={p.skipEmpty} onChange={(e) => p.onSkipEmpty(e.target.checked)} /> “next” skips (empty) &amp; inactif</label>
        </span>
      </div>
      {p.alignStatus !== "confirmed" && (
        <div className="callout warn small">Alignment is <b>{p.alignStatus}</b>. If the video↔CSV offset is wrong, correct labels look wrong.
          Align on a clear movement first; every verdict stores the a, b it was made under.</div>
      )}
      {p.csvRow == null ? <div className="callout bad small">{p.noDataText}</div> : (
        <table className="now-tbl">
          <tbody>
            {p.rows.map(({ col, seg, verdict }) => {
              const focused = p.focusCol === col.column;
              return (
                <tr key={col.column} className={`${focused ? "focused" : ""} ${col.role === "derived" ? "derived" : ""}`}
                  onClick={() => p.onFocus(col.column)} title="Click to focus this column (keys 1/2/3, pause-at-end, next)">
                  <td className="nm">{focused ? "▸ " : ""}{col.name.replace(/^etiq /, "")}{col.role === "derived" && <span className="muted"> (derived)</span>}</td>
                  <td>
                    {seg && <><span className="swatch" style={{ background: labelColor(seg.value) ?? "transparent" }} />
                      <b className={seg.value === "" ? "muted" : ""}>{displayValue(seg.value)}</b></>}
                  </td>
                  <td className="mono small">{seg && `${fmtSeconds(p.toT(seg.start_ts), 2)} – ${fmtSeconds(p.toT(seg.end_ts), 2)} (${((p.toT(seg.end_ts) - p.toT(seg.start_ts))).toFixed(1)} s)`}</td>
                  <td className="verdict-cell" onClick={(e) => e.stopPropagation()}>
                    {seg && (["agree", "disagree", "unsure"] as VerdictKind[]).map((k) => (
                      <button key={k} className={`vbtn ${verdict?.verdict === k ? `on ${k}` : ""}`} title={VERDICT_LABELS[k]}
                        onClick={() => quick(col, seg, k, verdict)}>{VERDICT_LABELS[k].split(" ")[0]}</button>
                    ))}
                    {seg && <button className="vbtn" title="Replay this segment from its start" onClick={() => { p.onFocus(col.column); p.onPlaySegment(seg); }}>▶</button>}
                    <button className="vbtn" title="Jump to the next unreviewed segment of this column" onClick={() => { p.onFocus(col.column); p.onNext(col); }}>⇥</button>
                  </td>
                  <td className="small">{verdict?.verdict === "disagree" && verdict.observed && <span>saw: <b>{verdict.observed}</b></span>}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      {form && (
        <div className="verdict-form">
          <div><b style={{ color: VERDICT_COLORS[form.kind] }}>{VERDICT_LABELS[form.kind]}</b> — {form.col.name}: <b>{displayValue(form.seg.value)}</b>{" "}
            <span className="muted small">rows {form.seg.start_row}–{form.seg.end_row - 1}</span></div>
          <div className="row wrap">
            <label>What do you see in the video?
              <input list={`vals-${form.col.column}`} value={form.observed} placeholder="e.g. Debout_Assis, or free text"
                onChange={(e) => setForm({ ...form, observed: e.target.value })} autoFocus />
              <datalist id={`vals-${form.col.column}`}>{form.col.summary.map((s) => <option key={s.value} value={s.value} />)}</datalist>
            </label>
          </div>
          <textarea rows={2} placeholder="comment (e.g. label starts ~1 s early; participant sits, not stands)" value={form.comment}
            onChange={(e) => setForm({ ...form, comment: e.target.value })} />
          <div className="row">
            <button className="primary" onClick={() => { p.onVerdict(form.col, form.seg, form.kind, form.observed.trim(), form.comment.trim()); setForm(null); }}>Save verdict</button>
            <button onClick={() => setForm(null)}>Cancel</button>
            <span className="muted small">Saved separately; the CSV label is never changed.</span>
          </div>
        </div>
      )}
      <div className="muted small">Keys: 1 agree · 2 disagree · 3 unsure (focused column) · Space play/pause · , . frame.</div>
    </div>
  );
}

// ---------------------------------------------------------------- tab: progress and list of verdicts
export function ReviewTab({ columns, verdicts, verdictMap, skip, a, b, onSeekT, onDelete, rid }: {
  columns: LabelColumn[]; verdicts: Verdict[]; verdictMap: Map<string, Verdict>; skip: Set<string>;
  a: number; b: number; onSeekT: (t: number) => void; onDelete: (id: string) => void; rid: string;
}) {
  const [filter, setFilter] = useState<"all" | VerdictKind>("disagree");
  const shown = verdicts.filter((v) => filter === "all" || v.verdict === filter).sort((x, y) => x.t_csv_start - y.t_csv_start);
  const stale = verdicts.filter((v) => madeUnderOtherAlignment(v, a, b)).length;
  const download = () => {
    const cols = ["id", "updated", "column_name", "label_value", "verdict", "observed", "comment", "start_row", "end_row",
      "ts_start", "ts_end", "t_csv_start", "t_csv_end", "t_video_start", "t_video_end"] as const;
    const esc = (x: unknown) => `"${String(x ?? "").replace(/"/g, '""')}"`;
    const lines = [[...cols, "align_a", "align_b", "align_status", "made_under_current_alignment"].join(",")];
    for (const v of verdicts) lines.push([...cols.map((c) => esc(v[c])), v.alignment_snapshot.a, v.alignment_snapshot.b,
      esc(v.alignment_snapshot.status), String(!madeUnderOtherAlignment(v, a, b))].join(","));
    const el = document.createElement("a");
    el.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
    el.download = `${rid}_label_verdicts.csv`;
    el.click();
  };
  return (
    <div>
      <p className="small muted">A verdict is your judgement of one label segment against the video. Verdicts are stored in
        reports\review_app\verdicts\ and never modify the CSV. “observed” is your proposal of the correct label.</p>
      <table className="tbl small">
        <thead><tr><th>column</th><th>segments{skip.size ? " (excl. (empty)/inactif)" : ""}</th><th>reviewed</th><th>agree</th><th>disagree</th><th>unsure</th></tr></thead>
        <tbody>
          {columns.map((c) => {
            const pr = progress(c.segments, c.column, skip, verdictMap);
            return (
              <tr key={c.column}>
                <td>{c.name}{c.role === "derived" && <span className="muted"> (derived)</span>}</td>
                <td>{pr.total}</td>
                <td>{pr.reviewed} ({pr.total ? ((100 * pr.reviewed) / pr.total).toFixed(0) : 0}%)</td>
                <td style={{ color: VERDICT_COLORS.agree }}>{pr.agree}</td>
                <td style={{ color: VERDICT_COLORS.disagree }}>{pr.disagree}</td>
                <td style={{ color: VERDICT_COLORS.unsure }}>{pr.unsure}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {stale > 0 && <div className="callout warn small">{stale} verdict(s) were made under a different alignment (a, b) than the current one. Re-check them.</div>}
      <div className="row wrap">
        <label>show <select value={filter} onChange={(e) => setFilter(e.target.value as typeof filter)}>
          <option value="disagree">disagree</option><option value="unsure">unsure</option><option value="agree">agree</option><option value="all">all</option>
        </select></label>
        <button disabled={!verdicts.length} onClick={download}>Download all verdicts CSV</button>
      </div>
      <table className="tbl small">
        <thead><tr><th>t_csv</th><th>column</th><th>CSV label</th><th>verdict</th><th>observed</th><th>comment</th><th>alignment</th><th></th></tr></thead>
        <tbody>
          {shown.map((v) => (
            <tr key={v.id}>
              <td><button className="link" onClick={() => onSeekT(v.t_csv_start)}>{fmtSeconds(v.t_csv_start, 2)} – {fmtSeconds(v.t_csv_end, 2)}</button></td>
              <td>{v.column_name}</td>
              <td>{displayValue(v.label_value)}</td>
              <td style={{ color: VERDICT_COLORS[v.verdict] }}>{v.verdict}</td>
              <td>{v.observed}</td>
              <td>{v.comment}</td>
              <td className={madeUnderOtherAlignment(v, a, b) ? "bad-text" : "muted"}>b={v.alignment_snapshot.b.toFixed(3)} ({v.alignment_snapshot.status})</td>
              <td><button className="link" onClick={() => window.confirm("Delete this verdict?") && onDelete(v.id)}>×</button></td>
            </tr>
          ))}
          {!shown.length && <tr><td colSpan={8} className="muted">none</td></tr>}
        </tbody>
      </table>
    </div>
  );
}
