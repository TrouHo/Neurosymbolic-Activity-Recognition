import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type ImportOpts, type Info, type LabelOverview, type Note, type RecordingItem, type TsReport, type VideoMeta } from "./api";
import AlignmentPanel from "./components/AlignmentPanel";
import FileBrowser from "./components/FileBrowser";
import { LabelsNowPanel, ReviewTab, type NowRow } from "./components/LabelReview";
import LabelTracks, { type Track } from "./components/LabelTracks";
import Overview from "./components/Overview";
import {
  ChannelsPanel, ExternalLabelsPanel, InfoPanel, LabelsPanel, MappingPanel, NotesPanel, QualityPanel, RawRowPanel,
} from "./components/Panels";
import SignalPlot, { type Band, type View } from "./components/SignalPlot";
import VideoPanel, { type ClockSource } from "./components/VideoPanel";
import { AXIS_COLORS } from "./lib/colors";
import { extToTCsv } from "./lib/extlabels";
import { checkImport, defaultProfile, nowIso, type Profile } from "./lib/profile";
import {
  csvCoverage, csvFromVideo, fmtSeconds, fmtTimestamp, nearestIndex, sampleIndexAt, tCsvFromTimestamp, UNIT_TO_MS,
  videoCoverage, videoFromCsv,
} from "./lib/time";
import { nextSegment, segKey, segmentAtRow, VERDICT_COLORS, verdictKey, type Verdict, type VerdictKind } from "./lib/verdicts";
import type { LabelColumn, Segment } from "./api";

type TsData = { report: TsReport; firstRaw: string; raw: Float64Array; col: number; unit: string; optsKey: string };
const TABS = ["review", "channels", "mapping", "labels", "quality", "raw", "notes", "export", "external", "info"] as const;
type Tab = (typeof TABS)[number];
const TAB_NAMES: Record<Tab, string> = { review: "Label review", channels: "Channels", mapping: "Mapping & time", labels: "Labels", quality: "Quality",
  raw: "Raw values", notes: "Notes", export: "Export", external: "External labels", info: "Recording info" };

const optsOf = (p: Profile | null): ImportOpts => p?.import_options ?? { encoding: "", delimiter: "", decimal: "." };
const keyOf = (o: ImportOpts) => `${o.encoding}|${o.delimiter}|${o.decimal}`;

export default function App() {
  const [outputDir, setOutputDir] = useState("");
  const [recs, setRecs] = useState<RecordingItem[]>([]);
  const [rid, setRid] = useState("");
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<Info | null>(null);
  const [vmeta, setVmeta] = useState<VideoMeta | null>(null);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [savedJson, setSavedJson] = useState("");
  const [profilePath, setProfilePath] = useState("");
  const [profileWarn, setProfileWarn] = useState<string[]>([]);
  const [ts, setTs] = useState<TsData | null>(null);
  const [series, setSeries] = useState<Record<number, (number | null)[]>>({});
  const [labels, setLabels] = useState<LabelOverview | null>(null);
  const [extRaw, setExtRaw] = useState<Record<string, Awaited<ReturnType<typeof api.extLoad>>>>({});
  const [notes, setNotes] = useState<Note[]>([]);
  const [view, setViewRaw] = useState<View>({ x0: 0, x1: 60 });
  const [selection, setSelection] = useState<Band | null>(null);
  const [videoT, setVideoT] = useState(0);
  const [clock, setClock] = useState<ClockSource | null>(null);
  const [duration, setDuration] = useState(0);
  const [picking, setPicking] = useState(false);
  const [draftCsv, setDraftCsv] = useState<number | null>(null);
  const [follow, setFollow] = useState(true);
  const [tab, setTab] = useState<Tab>("review");
  const [toast, setToast] = useState<string | null>(null);
  const [browse, setBrowse] = useState<null | { step: "csv" | "video"; csv?: string }>(null);
  const [scanBusy, setScanBusy] = useState(false);
  const [exportResult, setExportResult] = useState<string | null>(null);
  const [verdicts, setVerdicts] = useState<Verdict[]>([]);
  const [focusCol, setFocusCol] = useState<number | null>(null);
  const [skipEmpty, setSkipEmpty] = useState(true);
  const [pauseAtEnd, setPauseAtEnd] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  const say = (m: string) => { setToast(m); window.setTimeout(() => setToast((t) => (t === m ? null : t)), 6000); };

  useEffect(() => {
    api.config().then((c) => setOutputDir(c.output_dir)).catch((e) => setError(`Backend not reachable: ${e}`));
    api.recordings().then(setRecs).catch(() => undefined);
  }, []);

  // ------------------------------------------------------------ open a recording
  const openRecording = useCallback(async (id: string) => {
    setError(null); setInfo(null); setProfile(null); setTs(null); setSeries({}); setLabels(null); setVmeta(null);
    setSelection(null); setNotes([]); setDraftCsv(null); setPicking(false); setExportResult(null); setProfileWarn([]);
    setVideoT(0); setDuration(0); setVerdicts([]); setFocusCol(null);
    if (!id) return;
    try {
      setLoading("Reading saved review profile…");
      const saved = await api.getProfile(id);
      const opts = optsOf(saved.profile);
      setLoading("Parsing CSV (read-only)…");
      const inf = await api.info(id, opts);
      let meta: VideoMeta | null = null;
      if (inf.recording.video_path) meta = await api.videoMeta(id).catch((e) => ({ ok: false, error: String(e), path: inf.recording.video_path!, video_track: null }));
      const vt = meta?.video_track;
      const fps = vt?.nominal_fps ?? null;
      const fpsProv = vt?.fps_provenance ?? "No constant sample delta in the container; FPS unknown (frame stepping disabled).";
      const col = saved.profile?.timestamp.column ?? inf.proposals.timestamp?.column ?? 0;
      const unit = saved.profile?.timestamp.unit ?? inf.proposals.timestamp?.unit ?? "ms";
      setLoading("Reading timestamps…");
      const tsr = await api.timestamps(id, col, unit, opts);
      const raw = Float64Array.from(tsr.values, (v) => (v == null ? NaN : v));
      let prof: Profile;
      if (saved.profile) {
        prof = saved.profile;
        const chk = checkImport(prof, { recording_id: id, sha256: inf.csv.sha256 });
        setProfileWarn([`Loaded saved profile (${saved.profile.saved_at ?? "?"}).`, ...chk.warnings, ...chk.errors]);
      } else {
        prof = defaultProfile(inf, raw[0], tsr.first_raw, fps, fpsProv);
      }
      setInfo(inf); setVmeta(meta); setProfile(prof); setProfilePath(saved.path);
      setSavedJson(saved.profile ? JSON.stringify(saved.profile) : "");
      setTs({ report: tsr.report, firstRaw: tsr.first_raw, raw, col, unit, optsKey: keyOf(opts) });
      const t0 = tCsvFromTimestamp(raw[0], prof.timestamp.reference_value, prof.timestamp.unit);
      setViewRaw({ x0: t0, x1: t0 + Math.min(120, (raw.length * (tsr.report.median_step ?? 20) * UNIT_TO_MS[prof.timestamp.unit]) / 1000) });
      api.notes(id).then(setNotes).catch(() => undefined);
      api.verdicts(id).then(setVerdicts).catch(() => undefined);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(null);
    }
  }, []);

  useEffect(() => { openRecording(rid); }, [rid, openRecording]);

  // Reload timestamps / info if the user changes the timestamp column, unit or import options.
  useEffect(() => {
    if (!profile || !ts || !rid) return;
    const o = optsOf(profile);
    if (ts.col === profile.timestamp.column && ts.unit === profile.timestamp.unit && ts.optsKey === keyOf(o)) return;
    (async () => {
      try {
        setLoading("Reloading CSV with new settings…");
        if (ts.optsKey !== keyOf(o)) { setInfo(await api.info(rid, o)); setSeries({}); }
        const tsr = await api.timestamps(rid, profile.timestamp.column, profile.timestamp.unit, o);
        const raw = Float64Array.from(tsr.values, (v) => (v == null ? NaN : v));
        if (ts.col !== profile.timestamp.column)
          setProfile((p) => p && { ...p, timestamp: { ...p.timestamp, reference_value: raw[0], reference_raw: tsr.first_raw } });
        setTs({ report: tsr.report, firstRaw: tsr.first_raw, raw, col: profile.timestamp.column, unit: profile.timestamp.unit, optsKey: keyOf(o) });
      } catch (e) { setError(String(e)); } finally { setLoading(null); }
    })();
  }, [profile, ts, rid]);

  // ------------------------------------------------------------ derived time arrays
  const unit = profile?.timestamp.unit ?? "ms";
  const ref = profile?.timestamp.reference_value ?? 0;
  const t = useMemo(() => (ts ? Float64Array.from(ts.raw, (v) => tCsvFromTimestamp(v, ref, unit)) : null), [ts, ref, unit]);
  const toT = useCallback((v: number) => tCsvFromTimestamp(v, ref, unit), [ref, unit]);
  const stepS = ((ts?.report.median_step ?? 20) * UNIT_TO_MS[unit]) / 1000;
  const n = t?.length ?? 0;
  const csvFirst = n ? t![0] : 0;
  const csvLast = n ? t![n - 1] : 0;
  const model = useMemo(() => ({ a: profile?.alignment.a ?? 1, b: profile?.alignment.b ?? 0 }), [profile?.alignment.a, profile?.alignment.b]);
  const tc = csvFromVideo(videoT, model);
  const cov = csvCoverage(tc, csvFirst, csvLast, stepS);
  const row = t && cov === "inside" ? nearestIndex(t, tc) : null;
  const monotonic = !ts || (ts.report.n_nonmonotonic === 0 && ts.report.n_invalid === 0);
  const vidBand: Band | null = duration ? { t0: csvFromVideo(0, model), t1: csvFromVideo(duration, model), color: "" } : null;
  const extent: View = useMemo(() => {
    const lo = Math.min(csvFirst, vidBand ? vidBand.t0 : csvFirst);
    const hi = Math.max(csvLast + stepS, vidBand ? vidBand.t1 : csvLast);
    const pad = (hi - lo) * 0.01 + 1;
    return { x0: lo - pad, x1: hi + pad };
  }, [csvFirst, csvLast, stepS, vidBand?.t0, vidBand?.t1]); // eslint-disable-line react-hooks/exhaustive-deps

  const setView = useCallback((v: View) => {
    const span = Math.min(Math.max(0.2, v.x1 - v.x0), extent.x1 - extent.x0);
    let x0 = Math.max(extent.x0, Math.min(v.x0, extent.x1 - span));
    if (!Number.isFinite(x0)) x0 = extent.x0;
    setViewRaw({ x0, x1: x0 + span });
  }, [extent]);

  // keep the cursor in view while playing
  useEffect(() => {
    const v = videoRef.current;
    if (!follow || !v || v.paused || !Number.isFinite(tc)) return;
    if (tc < view.x0 || tc > view.x1) {
      const span = view.x1 - view.x0;
      setView({ x0: tc - span * 0.1, x1: tc + span * 0.9 });
    }
  }, [tc, follow, view, setView]);

  // ------------------------------------------------------------ series for selected groups
  const selGroups = useMemo(() => (info && profile ? profile.selected_groups.map((id) => info.groups.find((g) => g.id === id)).filter(Boolean) as Info["groups"] : []),
    [info, profile?.selected_groups]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!rid || !profile) return;
    const need = selGroups.flatMap((g) => Object.values(g.axes)).filter((c) => !(c in series));
    if (!need.length) return;
    api.series(rid, need, optsOf(profile)).then((s) => setSeries((old) => {
      const out = { ...old };
      for (const [k, v] of Object.entries(s)) out[Number(k)] = v;
      return out;
    })).catch((e) => setError(String(e)));
  }, [selGroups, rid]); // eslint-disable-line react-hooks/exhaustive-deps

  const plots = useMemo(() => {
    if (!t || !info || !profile) return [];
    const mapping = new Map(profile.channel_mapping.groups.map((g) => [g.group_id, g]));
    return selGroups.filter((g) => Object.values(g.axes).every((c) => c in series)).map((g) => {
      const m = mapping.get(g.id);
      const axes = Object.entries(g.axes).sort();
      const ys = axes.map(([a, c]) => ({ label: `${a} (${info.columns[c].name}@${c})`, data: series[c], color: AXIS_COLORS[a] ?? "#555" }));
      const bands: Band[] = [];
      let z0 = -1, m0 = -1;
      for (let i = 0; i <= n; i++) {
        const allZero = i < n && ys.every((y) => y.data[i] === 0);
        const missing = i < n && ys.some((y) => y.data[i] == null);
        if (allZero && z0 < 0) z0 = i;
        if (!allZero && z0 >= 0) { bands.push({ t0: t[z0], t1: i < n ? t[i] : t[n - 1] + stepS, color: "rgba(110,110,110,0.13)" }); z0 = -1; }
        if (missing && m0 < 0) m0 = i;
        if (!missing && m0 >= 0) { bands.push({ t0: t[m0], t1: i < n ? t[i] : t[n - 1] + stepS, color: "rgba(220,30,30,0.13)" }); m0 = -1; }
      }
      const badges: { text: string; tone: "warn" | "info" | "bad" }[] = [];
      if (g.flags.no_nonzero_data) badges.push({ text: "no non-zero data", tone: "bad" });
      if (!profile.channel_mapping.verified) badges.push({ text: "mapping unverified", tone: "warn" });
      if (m?.edited) badges.push({ text: "mapping edited", tone: "info" });
      if (g.modality === "XYZ") badges.push({ text: "Qualisys position — inspection only", tone: "info" });
      return { id: g.id, title: `${m?.source ?? g.source} · ${m?.placement ?? g.placement} · ${m?.modality ?? g.modality}   [${g.header}]`, ys, bands, badges };
    });
  }, [t, info, profile?.channel_mapping, selGroups, series, n, stepS]); // eslint-disable-line react-hooks/exhaustive-deps

  // ------------------------------------------------------------ labels
  const labelKey = profile ? JSON.stringify([profile.timestamp.column, profile.label_mapping.map((l) => [l.column, l.role]), optsOf(profile)]) : "";
  useEffect(() => {
    if (!rid || !profile || !ts || ts.col !== profile.timestamp.column) return;
    const fam = profile.label_mapping.filter((l) => l.role === "family").map((l) => l.column);
    const der = profile.label_mapping.filter((l) => l.role === "derived").map((l) => l.column);
    if (!fam.length && !der.length) { setLabels(null); return; }
    api.labels(rid, profile.timestamp.column, fam, der, optsOf(profile)).then(setLabels).catch((e) => setError(String(e)));
  }, [labelKey, rid, ts?.col]); // eslint-disable-line react-hooks/exhaustive-deps

  // external label files (raw rows cached; mapped to time below)
  useEffect(() => {
    profile?.external_labels.forEach((x) => {
      const k = JSON.stringify(x);
      if (extRaw[k]) return;
      api.extLoad(x).then((r) => setExtRaw((o) => ({ ...o, [k]: r }))).catch((e) => setError(`External labels: ${e}`));
    });
  }, [profile?.external_labels]); // eslint-disable-line react-hooks/exhaustive-deps

  // ------------------------------------------------------------ label review (verdicts)
  const labelCols: LabelColumn[] = useMemo(() => (labels && profile
    ? profile.label_mapping.filter((l) => l.role !== "ignore").map((l) => labels.columns[String(l.column)]).filter(Boolean)
    : []), [labels, profile?.label_mapping]); // eslint-disable-line react-hooks/exhaustive-deps
  const verdictMap = useMemo(() => new Map(verdicts.map((v) => [verdictKey(v), v])), [verdicts]);
  const skipSet = useMemo(() => new Set(skipEmpty ? ["", "inactif"] : []), [skipEmpty]);
  useEffect(() => {
    if (focusCol == null && labelCols.length) setFocusCol(labelCols.find((c) => c.role === "family" && c.name !== "etiq standard")?.column ?? labelCols[0].column);
  }, [labelCols, focusCol]);

  const tracks: Track[] = useMemo(() => {
    const out: Track[] = [];
    if (labels && profile) {
      for (const c of labelCols) {
        out.push({ key: `c${c.column}`, name: c.name.replace(/^etiq /, ""), note: c.role === "derived" ? "derived" : undefined,
          items: c.segments.map((s) => {
            const v = verdictMap.get(segKey(c.column, s));
            return { t0: toT(s.start_ts), t1: toT(s.end_ts), value: s.value, mark: v ? VERDICT_COLORS[v.verdict] : undefined,
              info: `rows ${s.start_row}–${s.end_row - 1}${s.end_extrapolated ? " · end extrapolated" : ""}${s.split_at_anomaly ? " · split at timestamp anomaly" : ""}`
                + (v ? `\nverdict: ${v.verdict}${v.observed ? ` (saw: ${v.observed})` : ""}${v.comment ? ` — ${v.comment}` : ""}` : "") };
          }) });
      }
      out.push({ key: "overlap", name: "overlap", note: "computed", fixedColor: "#d62728",
        items: labels.overlap.map((o) => ({ t0: toT(o.start_ts), t1: toT(o.end_ts), value: "overlap", info: `rows ${o.start_row}–${o.end_row - 1}` })) });
      out.push({ key: "unlabeled", name: "all empty", note: "computed", fixedColor: "#9aa0a6",
        items: labels.unlabeled.map((o) => ({ t0: toT(o.start_ts), t1: toT(o.end_ts), value: "all families empty", info: `rows ${o.start_row}–${o.end_row - 1}` })) });
    }
    profile?.external_labels.forEach((x, i) => {
      const r = extRaw[JSON.stringify(x)];
      if (!r) return;
      const toTc = (v: number) => extToTCsv(v, x.time_unit, x.time_reference, ref, unit, model);
      const items = r.items.filter((it) => it.start != null).map((it) => ({
        t0: toTc(it.start!), t1: it.end != null ? toTc(it.end) : null, value: it.label ?? "",
        info: `file row ${it.row}; raw start ${it.start_raw}${it.end_raw != null ? `, end ${it.end_raw}` : ""}${it.family ? `; family ${it.family}` : ""}`,
      })).sort((a, b) => a.t0 - b.t0);
      out.push({ key: `ext${i}`, name: `ext ${i + 1}`, note: r.kind, items });
    });
    return out;
  }, [labels, labelCols, verdictMap, profile?.external_labels, extRaw, toT, ref, unit, model]); // eslint-disable-line react-hooks/exhaustive-deps

  // ------------------------------------------------------------ actions
  const seekCsv = useCallback((tCsv: number) => {
    if (picking) { setDraftCsv(tCsv); setPicking(false); say(`Anchor CSV point set to t_csv = ${tCsv.toFixed(3)} s`); return; }
    const v = videoRef.current;
    if (!v) { say("No video loaded."); return; }
    const tv = videoFromCsv(tCsv, model);
    const vc = videoCoverage(tv, duration || v.duration);
    if (vc !== "inside") { say(`t_csv ${fmtSeconds(tCsv)} maps to video time ${fmtSeconds(tv)} — outside the video (0 – ${fmtSeconds(duration)}). Not seeking.`); return; }
    v.currentTime = tv;
  }, [picking, model, duration]);

  const nowRows: NowRow[] = useMemo(() => labelCols.map((col) => {
    const seg = row != null ? segmentAtRow(col.segments, row) : null;
    return { col, seg, verdict: seg ? verdictMap.get(segKey(col.column, seg)) ?? null : null };
  }), [labelCols, row, verdictMap]);

  // Optional: pause when playback leaves a (non-skipped) segment of the focused column.
  const focusSeg = nowRows.find((r) => r.col.column === focusCol)?.seg ?? null;
  const prevFocusSeg = useRef<Segment | null>(null);
  useEffect(() => {
    const prev = prevFocusSeg.current;
    prevFocusSeg.current = focusSeg;
    const v = videoRef.current;
    if (!pauseAtEnd || !v || v.paused || !prev || !focusSeg || prev === focusSeg) return;
    if (focusSeg.start_row === prev.end_row && !skipSet.has(prev.value)) {
      v.pause();
      say(`Paused at the end of “${prev.value}”. Give a verdict (1/2/3) or press Space to continue.`);
    }
  }, [focusSeg, pauseAtEnd, skipSet]);

  const segStartT = (s: Segment) => toT(s.start_ts) + stepS / 2; // middle of the first sample
  const playSegment = (s: Segment) => {
    const v = videoRef.current;
    if (!v) return;
    const tv = videoFromCsv(segStartT(s), model);
    if (videoCoverage(tv, duration || v.duration) !== "inside") { say("This segment is outside the video under the current alignment."); return; }
    prevFocusSeg.current = null; // do not pause on the jump itself
    v.currentTime = tv;
    v.play();
  };
  const nextFor = (c: LabelColumn) => {
    const s = nextSegment(c.segments, row ?? -1, c.column, skipSet, verdictMap);
    if (!s) { say(`No further unreviewed segment in ${c.name}${skipEmpty ? " (skipping (empty) and inactif)" : ""}.`); return; }
    prevFocusSeg.current = null;
    seekCsv(segStartT(s));
  };
  const saveVerdict = async (c: LabelColumn, s: Segment, kind: VerdictKind, observed: string, comment: string) => {
    if (!info || !ts || !profile) return;
    const tv = (x: number) => { try { return videoFromCsv(x, model); } catch { return null; } };
    try {
      const saved = await api.putVerdict(rid, {
        column: c.column, column_name: c.name, start_row: s.start_row, end_row: s.end_row, label_value: s.value,
        verdict: kind, observed, comment, ts_start: s.start_ts, ts_end: s.end_ts, t_csv_start: toT(s.start_ts), t_csv_end: toT(s.end_ts),
        t_video_start: tv(toT(s.start_ts)), t_video_end: tv(toT(s.end_ts)), csv_sha256: info.csv.sha256,
        alignment_snapshot: { a: model.a, b: model.b, status: profile.alignment.status, method: profile.alignment.method },
      });
      setVerdicts((vs) => [...vs.filter((x) => x.id !== saved.id), saved]);
      say(`${kind} saved for ${c.name}: ${s.value || "(empty)"}`);
    } catch (e) { setError(`Saving verdict failed: ${e}`); }
  };

  const updateProfile = (p: Profile) => setProfile(p);
  const dirty = profile ? JSON.stringify(profile) !== savedJson : false;
  const saveProfile = async () => {
    if (!profile) return;
    const r = await api.putProfile(rid, profile);
    setProfile(r.profile); setSavedJson(JSON.stringify(r.profile)); setProfilePath(r.path);
    say(`Profile saved: ${r.path}`);
  };
  const importProfile = async (f: File) => {
    try {
      const obj = JSON.parse(await f.text());
      const chk = checkImport(obj, { recording_id: rid, sha256: info!.csv.sha256 });
      if (!chk.ok) { window.alert(`Cannot import:\n${chk.errors.join("\n")}`); return; }
      if (chk.warnings.length && !window.confirm(`Import anyway?\n${chk.warnings.join("\n")}`)) return;
      const p = obj as Profile;
      setProfile({ ...p, recording_id: rid, alignment: { ...p.alignment,
        history: [...p.alignment.history, { at: nowIso(), action: `imported from file ${f.name}`, a: p.alignment.a, b: p.alignment.b }] } });
      say("Profile imported (not yet saved).");
    } catch (e) { window.alert(`Invalid JSON: ${e}`); }
  };
  const downloadProfile = () => {
    if (!profile) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(profile, null, 2)], { type: "application/json" }));
    a.download = `${rid}_review_profile.json`;
    a.click();
  };

  const rangeRows = (): [number, number] | null => {
    if (!t) return null;
    if (selection) {
      const r0 = Math.max(0, sampleIndexAt(t, selection.t0));
      const r1 = Math.min(n, sampleIndexAt(t, selection.t1) + 1);
      return r1 > r0 ? [r0, r1] : null;
    }
    return row != null ? [row, row + 1] : null;
  };
  const addNote = async (text: string) => {
    const rr = rangeRows();
    if (!rr || !t || !ts || !profile || !info) return;
    const [r0, r1] = rr;
    const tv = (x: number) => { try { return videoFromCsv(x, model); } catch { return null; } };
    const note = await api.addNote(rid, {
      text, row_start: r0, row_end: r1, ts_start: ts.raw[r0], ts_end: ts.raw[r1 - 1], t_csv_start: t[r0], t_csv_end: t[r1 - 1],
      t_video_start: tv(t[r0]), t_video_end: tv(t[r1 - 1]), csv_sha256: info.csv.sha256,
      alignment_snapshot: { a: model.a, b: model.b, status: profile.alignment.status, method: profile.alignment.method,
        anchors: profile.alignment.anchors.length, reference_value: ref, unit },
    });
    setNotes((ns) => [...ns, note]);
  };
  const doExport = async () => {
    const rr = rangeRows();
    if (!rr || !profile || !labels) { if (!rr) say("Select a time range first (drag in a plot)."); }
    if (!rr || !profile) return;
    const [r0, r1] = rr;
    const segs = labels ? Object.values(labels.columns).flatMap((c) => c.segments.filter((s) => s.end_row > r0 && s.start_row < r1)
      .map((s) => ({ column: c.name, role: c.role, ...s }))) : [];
    const res = await api.exportRows(rid, {
      row_start: r0, row_end: r1, timestamp: profile.timestamp,
      alignment: { ...profile.alignment, video_timing: profile.video_timing, pairing: profile.pairing, video_path: profile.video.path },
      channel_mapping: profile.channel_mapping, label_segments: segs,
      notes: notes.filter((x) => x.row_end > r0 && x.row_start < r1),
      verdicts: verdicts.filter((x) => x.end_row > r0 && x.start_row < r1),
    }, optsOf(profile));
    setExportResult(`${res.folder}\n${res.files.join(", ")}`);
  };

  const scanAll = async () => {
    setScanBusy(true);
    for (const r of recs) {
      if (r.summary) continue;
      try { const s = await api.summarize(r.id); setRecs((rs) => rs.map((x) => (x.id === r.id ? { ...x, summary: s } : x))); } catch { /* shown per row */ }
    }
    setScanBusy(false);
  };

  const recFlags = (r: RecordingItem) => {
    const f: string[] = [];
    if (!r.video_path) f.push("no video");
    if (r.summary) {
      if (r.summary.label_columns_entirely_empty.length) f.push("labels empty");
      const imu = r.summary.groups.filter((g) => ["ACC", "GYR"].includes(g.modality.toUpperCase()));
      if (imu.length && imu.every((g) => g.no_nonzero_data)) f.push("no IMU data");
      else if (imu.some((g) => g.modality.toUpperCase() === "GYR") && imu.filter((g) => g.modality.toUpperCase() === "GYR").every((g) => g.no_nonzero_data)) f.push("GYR empty");
    }
    return f;
  };

  // ------------------------------------------------------------ render
  const vt = vmeta?.video_track;
  const al = profile?.alignment;
  const anchorsT = al?.anchors.map((a) => a.tCsv) ?? [];
  const markers = [
    ...(al?.anchors.map((a) => ({ t: a.tCsv, color: "#7b2cbf", title: `anchor ${a.id} (${a.kind}) ${a.description}` })) ?? []),
    ...(draftCsv != null ? [{ t: draftCsv, color: "#00a3a3", title: "draft anchor CSV point" }] : []),
  ];
  const selBand = selection ? [{ ...selection, color: "rgba(255,200,0,0.22)" }] : [];

  return (
    <div className="app">
      <header>
        <b className="brand">eMOB Data Review</b>
        <select value={rid} onChange={(e) => setRid(e.target.value)}>
          <option value="">— select recording —</option>
          {recs.map((r) => {
            const f = recFlags(r);
            return <option key={r.id} value={r.id}>{r.id}{f.length ? `  [${f.join(", ")}]` : ""}</option>;
          })}
        </select>
        <button onClick={scanAll} disabled={scanBusy} title="Parse every CSV once to flag empty labels / empty IMU groups (cached in the output dir)">
          {scanBusy ? "Scanning…" : "Scan availability flags"}</button>
        <button onClick={() => setBrowse({ step: "csv" })}>Open other CSV/video…</button>
        <span className="spacer" />
        {profile && (
          <>
            <span className={`chip-status ${profile.pairing.status}`}>Pairing: {profile.pairing.status}{profile.pairing.rule ? ` (${profile.pairing.rule})` : ""}</span>
            {profile.pairing.status === "proposed" && (
              <button onClick={() => window.confirm(`Confirm that\n${info?.csv.path}\nand\n${profile.video.path}\nare the same recording?`) &&
                updateProfile({ ...profile, pairing: { ...profile.pairing, status: "confirmed", confirmed_at: nowIso() } })}>Confirm pairing</button>
            )}
            <span className={`chip-status ${profile.alignment.status}`}>Alignment: {profile.alignment.status}</span>
            <button className={dirty ? "primary" : ""} onClick={saveProfile} disabled={!dirty}>{dirty ? "Save profile*" : "Saved"}</button>
            <button onClick={downloadProfile}>Download profile</button>
            <label className="file-btn">Import profile<input type="file" accept=".json" onChange={(e) => e.target.files?.[0] && importProfile(e.target.files[0])} /></label>
          </>
        )}
      </header>

      {error && <div className="callout bad">{error} <button className="link" onClick={() => setError(null)}>dismiss</button></div>}
      {loading && <div className="callout info">{loading}</div>}
      {profileWarn.length > 0 && <div className="callout info small">{profileWarn.map((w) => <div key={w}>{w}</div>)}</div>}
      {!monotonic && <div className="callout bad">Timestamps contain invalid or non-monotonic values. Time lookup assumes ascending timestamps; see Quality.</div>}
      {!rid && <div className="empty-state">Select a recording. Source files are opened read-only; review outputs go to <span className="mono">{outputDir}</span>.</div>}

      {profile && info && t && (
        <>
          <div className="main">
            <div className="left">
              <VideoPanel videoRef={videoRef} src={info.recording.video_path ? api.videoUrl(rid) : null} meta={vmeta}
                fps={profile.video_timing.fps} outputDir={outputDir}
                onTime={(x, s) => { setVideoT(x); setClock(s); }} onDuration={setDuration} />
              <div className="readout">
                <div><span>Video time</span><b className="mono">{fmtSeconds(videoT)}</b>
                  <span className="muted">≈ frame {profile.video_timing.fps ? Math.floor(videoT * profile.video_timing.fps + 1e-6) : "—"} (approx.)</span></div>
                <div><span>Mapped CSV time</span><b className="mono">{fmtSeconds(tc)}</b><span className="muted">t_csv = {model.a}·t_v + {model.b.toFixed(4)}</span></div>
                <div><span>Nearest sample</span>{row != null
                  ? <><b className="mono">row {row}</b><span className="mono">ts {fmtTimestamp(ts!.raw[row])}</span><span className="mono">Δ {((t[row] - tc) * 1000).toFixed(1)} ms</span></>
                  : <b className="bad-text">{cov === "before" ? "No CSV data (before first sample)" : cov === "after" ? "No CSV data (after last sample)" : "—"}</b>}</div>
                <div className="muted small">Clock: {clock ?? "—"}. Video FPS {vt?.nominal_fps?.toFixed(5) ?? "unknown"} ({profile.video_timing.fps_provenance}).
                  CSV step {stepS * 1000} ms ({unit}: {profile.timestamp.unit_provenance})</div>
              </div>
              <LabelsNowPanel rows={nowRows} csvRow={row} toT={toT} alignStatus={profile.alignment.status}
                noDataText={cov === "before" ? "No CSV data at this video time (before the first CSV sample)." : cov === "after" ? "No CSV data at this video time (after the last CSV sample)." : "No CSV sample at this video time."}
                focusCol={focusCol} onFocus={setFocusCol} skipEmpty={skipEmpty} onSkipEmpty={setSkipEmpty}
                pauseAtEnd={pauseAtEnd} onPauseAtEnd={setPauseAtEnd} onPlaySegment={playSegment} onNext={nextFor}
                onPause={() => videoRef.current?.pause()} onVerdict={saveVerdict} />
              <AlignmentPanel profile={profile} videoT={videoT} sampleStep={stepS}
                span={Math.max(csvLast - csvFirst, duration) || 600}
                pickActive={picking} draftCsvT={draftCsv} onDraftCsv={setDraftCsv}
                onStartPick={() => { setPicking(true); say("Click the event in any plot or label track to set the anchor's CSV time."); }}
                onCancelPick={() => setPicking(false)}
                onChange={(alignment) => updateProfile({ ...profile, alignment })} />
            </div>

            <div className={`right ${picking ? "picking" : ""}`}>
              <div className="view-bar">
                <button onClick={() => setView(extent)}>Fit all</button>
                <button onClick={() => setView({ x0: tc - 5, x1: tc + 5 })}>10 s at cursor</button>
                <button onClick={() => setView({ x0: tc - 30, x1: tc + 30 })}>60 s at cursor</button>
                <button disabled={!selection} onClick={() => selection && setView({ x0: selection.t0, x1: selection.t1 })}>Zoom to selection</button>
                <button disabled={!selection} onClick={() => setSelection(null)}>Clear selection</button>
                <label><input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> follow playback</label>
                <span className="muted small">view {fmtSeconds(view.x0, 2)} – {fmtSeconds(view.x1, 2)} · wheel = zoom, Shift+wheel = pan, drag = select, click = seek video</span>
              </div>
              <Overview extent={extent} csv={{ t0: csvFirst, t1: csvLast + stepS, color: "" }} video={vidBand} view={view} cursorT={Number.isFinite(tc) ? tc : null}
                anchors={anchorsT} selection={selection} onView={setView} />
              <div className="plots">
                {plots.length === 0 && <div className="muted">No channel groups selected or data still loading (Channels tab).</div>}
                {plots.map((p) => (
                  <SignalPlot key={p.id} title={p.title} badges={p.badges} x={t} ys={p.ys} view={view} onView={setView}
                    cursorT={Number.isFinite(tc) ? tc : null} markers={markers} bands={[...p.bands, ...selBand]}
                    onClickT={seekCsv} onSelect={(a, b) => setSelection({ t0: Math.min(a, b), t1: Math.max(a, b), color: "" })} />
                ))}
              </div>
              <div className="legend-bands small muted">
                <span className="sw zero" /> all axes exactly 0 (suspicious candidate, not assumed missing)
                <span className="sw missing" /> missing / NaN / non-numeric (drawn as gap)
                <span className="sw sel" /> selection <span className="sw cur" /> video cursor <span className="sw anc" /> anchors
              </div>
              <LabelTracks tracks={tracks} view={view} cursorT={Number.isFinite(tc) ? tc : null} selection={selection} onClickT={seekCsv} onView={setView} />
              <div className="small muted">Label tracks show raw values; “(empty)” (hatched) is kept distinct from “inactif” and never treated as rest.</div>
            </div>
          </div>

          <nav className="tabs">
            {TABS.map((k) => <button key={k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{TAB_NAMES[k]}</button>)}
          </nav>
          <section className="tab-body">
            {tab === "review" && <ReviewTab columns={labelCols} verdicts={verdicts} verdictMap={verdictMap} skip={skipSet}
              a={model.a} b={model.b} onSeekT={seekCsv} rid={rid}
              onDelete={(id) => api.deleteVerdict(rid, id).then(() => setVerdicts((vs) => vs.filter((x) => x.id !== id)))} />}
            {tab === "channels" && <ChannelsPanel info={info} profile={profile}
              onToggle={(gid, on) => updateProfile({ ...profile, selected_groups: on ? [...profile.selected_groups, gid] : profile.selected_groups.filter((g) => g !== gid) })} />}
            {tab === "mapping" && <MappingPanel info={info} profile={profile} tsReport={ts?.report ?? null} firstRaw={ts?.firstRaw ?? ""} onChange={updateProfile} />}
            {tab === "labels" && <LabelsPanel labels={labels} toT={toT} onSeekT={seekCsv} />}
            {tab === "quality" && <QualityPanel rid={rid} profile={profile} opts={optsOf(profile)} toT={toT} onZoom={(a, b) => setView({ x0: a, x1: b })} />}
            {tab === "raw" && <RawRowPanel rid={rid} row={row} opts={optsOf(profile)} auto={!!videoRef.current?.paused || !videoRef.current} />}
            {tab === "notes" && <NotesPanel notes={notes} onAdd={addNote} onDelete={(id) => api.deleteNote(rid, id).then(() => setNotes((ns) => ns.filter((x) => x.id !== id)))}
              onSeekT={seekCsv} canAdd={!!rangeRows()}
              rangeText={selection ? `selection ${fmtSeconds(selection.t0, 2)} – ${fmtSeconds(selection.t1, 2)}` : row != null ? `current sample (row ${row})` : "nothing (no CSV sample at cursor)"} />}
            {tab === "export" && (
              <div>
                <p className="small">Exports the selected rows (original header and line text copied verbatim, with two clearly named derived columns),
                  the label segments in range, notes in range, the channel mapping, and the alignment metadata — as separate files in a new folder under
                  <span className="mono"> {outputDir}\exports</span>. Nothing is overwritten.</p>
                <div className="row">
                  <span>Range: {selection ? `${fmtSeconds(selection.t0, 3)} – ${fmtSeconds(selection.t1, 3)} (rows ${rangeRows()?.join("–")})` : "no selection (drag in a plot)"}</span>
                  <button disabled={!selection} onClick={doExport}>Export selection</button>
                </div>
                {exportResult && <pre className="callout info">{exportResult}</pre>}
              </div>
            )}
            {tab === "external" && <ExternalLabelsPanel profile={profile} onChange={updateProfile} />}
            {tab === "info" && <InfoPanel info={info} meta={vmeta} profilePath={profilePath} />}
          </section>
        </>
      )}

      {browse && (
        <FileBrowser title={browse.step === "csv" ? "Choose a CSV file" : "Choose the paired video (or Close for none)"}
          accept={browse.step === "csv" ? [".csv", ".tsv", ".txt"] : [".mp4", ".m4v", ".mov", ".webm"]}
          onClose={() => {
            if (browse.step === "video" && browse.csv) {
              api.addCustom(browse.csv, null).then((r) => { setRecs((rs) => [...rs.filter((x) => x.id !== r.id), r]); setRid(r.id); }).catch((e) => setError(String(e)));
            }
            setBrowse(null);
          }}
          onPick={(p) => {
            if (browse.step === "csv") { setBrowse({ step: "video", csv: p }); return; }
            api.addCustom(browse.csv!, p).then((r) => { setRecs((rs) => [...rs.filter((x) => x.id !== r.id), r]); setRid(r.id); }).catch((e) => setError(String(e)));
            setBrowse(null);
          }} />
      )}
      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}
