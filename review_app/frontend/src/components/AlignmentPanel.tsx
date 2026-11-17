import { useState } from "react";
import type { AlignStatus, Profile } from "../lib/profile";
import { nowIso } from "../lib/profile";
import { fmtSeconds, oneAnchor, residuals, twoAnchor, type Anchor, type Model } from "../lib/time";

interface Props {
  profile: Profile;
  videoT: number;
  sampleStep: number;            // seconds
  span: number;                  // seconds, for the offset slider range
  pickActive: boolean;
  draftCsvT: number | null;      // set by clicking a plot while picking, or typed here
  onDraftCsv: (t: number | null) => void;
  onStartPick: () => void;
  onCancelPick: () => void;
  onChange: (alignment: Profile["alignment"]) => void;
}

export default function AlignmentPanel({ profile, videoT, sampleStep, span, pickActive, draftCsvT, onDraftCsv, onStartPick, onCancelPick, onChange }: Props) {
  const al = profile.alignment;
  const fps = profile.video_timing.fps;
  const [draftV, setDraftV] = useState<number | null>(null);
  const [kind, setKind] = useState<Anchor["kind"]>("signal");
  const [desc, setDesc] = useState("");
  const [sel, setSel] = useState<string[]>([]);
  const csvT = draftCsvT;

  const setModel = (m: Model, method: Profile["alignment"]["method"], detail?: string) => {
    let status = al.status;
    let extra = "";
    if (status === "confirmed") { status = "provisional"; extra = " (status reset from confirmed to provisional)"; }
    if (status === "unverified" && method !== "default") status = "provisional";
    onChange({ ...al, a: m.a, b: m.b, method, status,
      history: [...al.history, { at: nowIso(), action: `${method}${extra}`, a: m.a, b: m.b, detail }] });
  };
  const setStatus = (s: AlignStatus) => {
    if (s === "confirmed" && !window.confirm("Mark this alignment as CONFIRMED by you? It will be recorded in the profile history.")) return;
    onChange({ ...al, status: s, history: [...al.history, { at: nowIso(), action: `status -> ${s}`, a: al.a, b: al.b }] });
  };
  const nudge = (d: number) => setModel({ a: al.a, b: +(al.b + d).toFixed(9) }, "manual", `nudge ${d >= 0 ? "+" : ""}${d} s`);

  const addAnchor = () => {
    if (draftV == null || csvT == null) return;
    const an: Anchor = { id: `A${Date.now()}`, tVideo: draftV, tCsv: csvT, kind, description: desc, created: nowIso() };
    onChange({ ...al, anchors: [...al.anchors, an],
      history: [...al.history, { at: nowIso(), action: "anchor added", a: al.a, b: al.b, detail: `${an.id} ${kind}: v=${draftV} c=${csvT}` }] });
    setDraftV(null); onDraftCsv(null); setDesc(""); onCancelPick();
  };
  const removeAnchor = (id: string) => {
    onChange({ ...al, anchors: al.anchors.filter((a) => a.id !== id),
      history: [...al.history, { at: nowIso(), action: "anchor removed", a: al.a, b: al.b, detail: id }] });
    setSel(sel.filter((s) => s !== id));
  };

  const chosen = al.anchors.filter((a) => sel.includes(a.id));
  const fit2 = chosen.length === 2 ? twoAnchor(chosen[0], chosen[1]) : null;
  const res = residuals(al.anchors, al);

  return (
    <div className="align-panel">
      <div className="row space">
        <b>Alignment</b>
        <select className={`status ${al.status}`} value={al.status} onChange={(e) => setStatus(e.target.value as AlignStatus)}>
          <option value="unverified">Unverified</option>
          <option value="provisional">Provisional (manual)</option>
          <option value="confirmed">Confirmed (by me)</option>
        </select>
      </div>
      <div className="formula">t_csv = a · t_video + b &nbsp; <span className="muted">t_csv = (timestamp − reference) / 1000 s</span></div>
      <div className="muted small">Positive b: video time 0 maps to a later CSV time (CSV recording began earlier than the video). Method: {al.method}.</div>
      <div className="grid2">
        <label>b (s)
          <input type="number" step={0.001} value={al.b} onChange={(e) => setModel({ a: al.a, b: Number(e.target.value) }, "manual")} />
        </label>
        <label>a (scale)
          <input type="number" step={0.0001} value={al.a} onChange={(e) => { const a = Number(e.target.value); if (a > 0) setModel({ a, b: al.b }, "manual"); }} />
        </label>
      </div>
      <input type="range" className="wide" min={-span} max={span} step={0.01} value={al.b}
        onChange={(e) => setModel({ a: al.a, b: Number(e.target.value) }, "manual")} />
      <div className="row wrap">
        {[-1, -0.1].map((d) => <button key={d} onClick={() => nudge(d)}>{d} s</button>)}
        {fps && <button onClick={() => nudge(-1 / fps)}>−1 frame</button>}
        <button onClick={() => nudge(-sampleStep)}>−1 sample</button>
        <button onClick={() => nudge(sampleStep)}>+1 sample</button>
        {fps && <button onClick={() => nudge(1 / fps)}>+1 frame</button>}
        {[0.1, 1].map((d) => <button key={d} onClick={() => nudge(d)}>+{d} s</button>)}
      </div>

      <details open>
        <summary>Anchors ({al.anchors.length})</summary>
        <div className="anchor-draft">
          <div className="row wrap">
            <button onClick={() => setDraftV(videoT)}>Set video point = {fmtSeconds(videoT)}</button>
            <span className="mono">{draftV != null ? `v=${draftV.toFixed(3)} s` : "v: —"}</span>
          </div>
          <div className="row wrap">
            {pickActive
              ? <button className="active" onClick={onCancelPick}>Click the event in a plot… (cancel)</button>
              : <button onClick={onStartPick}>Pick CSV point in plot</button>}
            <input type="number" step={0.001} placeholder="or type t_csv" value={csvT ?? ""} onChange={(e) => onDraftCsv(e.target.value === "" ? null : Number(e.target.value))} />
          </div>
          <div className="row wrap">
            <select value={kind} onChange={(e) => setKind(e.target.value as Anchor["kind"])}>
              <option value="signal">signal-based (visible in signal + video)</option>
              <option value="annotation">annotation-based (e.g. Debut)</option>
              <option value="other">other</option>
            </select>
            <input placeholder="description (e.g. first lunge, lowest point)" value={desc} onChange={(e) => setDesc(e.target.value)} />
            <button disabled={draftV == null || csvT == null} onClick={addAnchor}>Add anchor</button>
          </div>
        </div>
        {al.anchors.length > 0 && (
          <table className="tbl small">
            <thead><tr><th></th><th>id</th><th>kind</th><th>t_video</th><th>t_csv</th><th>residual</th><th>description</th><th></th></tr></thead>
            <tbody>
              {al.anchors.map((an, i) => (
                <tr key={an.id}>
                  <td><input type="checkbox" checked={sel.includes(an.id)} onChange={(e) => setSel(e.target.checked ? [...sel, an.id] : sel.filter((s) => s !== an.id))} /></td>
                  <td>{an.id.slice(-5)}</td><td>{an.kind}</td>
                  <td className="mono">{an.tVideo.toFixed(3)}</td><td className="mono">{an.tCsv.toFixed(3)}</td>
                  <td className="mono" title="(a·t_video + b) − t_csv under the current model">{(res[i] * 1000).toFixed(1)} ms</td>
                  <td>{an.description}</td>
                  <td><button className="link" onClick={() => removeAnchor(an.id)}>×</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <div className="row wrap">
          <button disabled={chosen.length !== 1} onClick={() => { const m = oneAnchor(chosen[0], al.a); setModel(m, "one_anchor", `anchor ${chosen[0].id}, a kept = ${al.a}`); }}>
            Apply 1 anchor (keep a)
          </button>
          <button disabled={!fit2?.model} onClick={() => fit2?.model && setModel(fit2.model, "two_anchor", `anchors ${chosen.map((c) => c.id).join(", ")}`)}>
            Apply 2-anchor fit (a, b)
          </button>
        </div>
        {fit2 && (
          <div className={`callout ${fit2.errors.length ? "bad" : fit2.warnings.length ? "warn" : "info"}`}>
            {fit2.model && <div className="mono">preview: a = {fit2.model.a.toFixed(7)}, b = {fit2.model.b.toFixed(4)} s</div>}
            {[...fit2.errors, ...fit2.warnings].map((m) => <div key={m}>{m}</div>)}
          </div>
        )}
        <div className="muted small">Anchors are your observations, not ground truth. `Debut` is never used automatically.</div>
      </details>
      <details>
        <summary>History ({al.history.length})</summary>
        <ul className="history">
          {[...al.history].reverse().map((h, i) => <li key={i}><span className="mono">{h.at.slice(0, 19)}</span> {h.action} — a={h.a.toFixed(6)}, b={h.b.toFixed(4)} {h.detail && <span className="muted">({h.detail})</span>}</li>)}
        </ul>
      </details>
    </div>
  );
}
