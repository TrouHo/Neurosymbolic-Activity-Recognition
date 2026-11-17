import { useEffect, useRef, useState } from "react";
import { displayValue, labelColor } from "../lib/colors";
import { fmtSeconds } from "../lib/time";
import { GUTTER_LEFT, GUTTER_RIGHT, type Band, type View } from "./SignalPlot";

export interface TrackItem {
  t0: number;
  t1: number | null;      // null = point event
  value: string;
  info: string;           // tooltip detail
  mark?: string;          // reviewer verdict colour, drawn as a strip under the segment
}
export interface Track {
  key: string;
  name: string;
  note?: string;          // e.g. "derived", "external", "computed"
  items: TrackItem[];     // sorted by t0
  fixedColor?: string;
}

interface Props {
  tracks: Track[];
  view: View;
  cursorT: number | null;
  selection: Band | null;
  onClickT: (t: number) => void;
  onView: (v: View) => void;
}

const ROW_H = 22;

function findItem(items: TrackItem[], t: number, tolT: number): TrackItem | null {
  let lo = 0, hi = items.length - 1, best = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (items[mid].t0 <= t) { best = mid; lo = mid + 1; } else hi = mid - 1;
  }
  for (const i of [best, best + 1]) {
    const it = items[i];
    if (!it) continue;
    if (it.t1 == null ? Math.abs(it.t0 - t) <= tolT : it.t0 <= t && t < it.t1) return it;
  }
  return null;
}

export default function LabelTracks({ tracks, view, cursorT, selection, onClickT, onView }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(800);
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  const height = Math.max(1, tracks.length) * ROW_H + 4;

  useEffect(() => {
    const ro = new ResizeObserver(() => setWidth(wrapRef.current?.clientWidth ?? 800));
    ro.observe(wrapRef.current!);
    return () => ro.disconnect();
  }, []);

  const plotW = width - GUTTER_LEFT - GUTTER_RIGHT;
  const toX = (t: number) => GUTTER_LEFT + ((t - view.x0) / (view.x1 - view.x0)) * plotW;
  const toT = (x: number) => view.x0 + ((x - GUTTER_LEFT) / plotW) * (view.x1 - view.x0);

  useEffect(() => {
    const c = canvasRef.current!;
    const dpr = window.devicePixelRatio || 1;
    c.width = width * dpr;
    c.height = height * dpr;
    c.style.width = `${width}px`;
    c.style.height = `${height}px`;
    const ctx = c.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const hatch = (() => {
      const p = document.createElement("canvas");
      p.width = p.height = 6;
      const g = p.getContext("2d")!;
      g.strokeStyle = "#b8bec7";
      g.beginPath(); g.moveTo(0, 6); g.lineTo(6, 0); g.stroke();
      return ctx.createPattern(p, "repeat")!;
    })();
    ctx.font = "11px system-ui";
    tracks.forEach((tr, r) => {
      const y = r * ROW_H + 2;
      ctx.fillStyle = "#f6f7f9";
      ctx.fillRect(GUTTER_LEFT, y, plotW, ROW_H - 2);
      ctx.save();
      ctx.beginPath();
      ctx.rect(GUTTER_LEFT, y, plotW, ROW_H - 2);
      ctx.clip();
      for (const it of tr.items) {
        const t1 = it.t1 ?? it.t0;
        if (t1 < view.x0 || it.t0 > view.x1) continue;
        const x0 = toX(it.t0);
        if (it.t1 == null) {
          ctx.fillStyle = tr.fixedColor ?? labelColor(it.value) ?? "#555";
          ctx.fillRect(x0 - 1, y, 2, ROW_H - 2);
          continue;
        }
        const x1 = toX(it.t1);
        const color = tr.fixedColor ?? labelColor(it.value);
        ctx.fillStyle = color ?? hatch;
        ctx.fillRect(x0, y + 1, Math.max(1, x1 - x0), ROW_H - 4);
        if (it.mark) {
          ctx.fillStyle = it.mark;
          ctx.fillRect(x0, y + ROW_H - 6, Math.max(2, x1 - x0), 4);
        }
        const text = displayValue(it.value);
        const vis0 = Math.max(x0, GUTTER_LEFT) + 3;
        const vis1 = Math.min(x1, GUTTER_LEFT + plotW) - 3;
        if (vis1 - vis0 > ctx.measureText(text).width) {
          ctx.fillStyle = color ? "#111" : "#666";
          ctx.fillText(text, vis0, y + 14);
        }
      }
      ctx.restore();
      ctx.fillStyle = "#222";
      const name = tr.name.length > 13 ? tr.name.slice(0, 12) + "…" : tr.name;
      ctx.fillText(name, 2, y + 11);
      if (tr.note) { ctx.fillStyle = "#888"; ctx.font = "9px system-ui"; ctx.fillText(tr.note, 2, y + 20); ctx.font = "11px system-ui"; }
    });
    if (selection) {
      ctx.fillStyle = "rgba(255,200,0,0.18)";
      ctx.fillRect(toX(selection.t0), 0, toX(selection.t1) - toX(selection.t0), height);
    }
    if (cursorT != null && cursorT >= view.x0 && cursorT <= view.x1) {
      ctx.strokeStyle = "#e0007a";
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(toX(cursorT), 0); ctx.lineTo(toX(cursorT), height); ctx.stroke();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tracks, view, cursorT, selection, width, height]);

  const locate = (e: React.MouseEvent) => {
    const r = canvasRef.current!.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    const row = Math.floor((y - 2) / ROW_H);
    return { x, y, row, t: toT(x) };
  };

  return (
    <div ref={wrapRef} className="label-tracks">
      <canvas
        ref={canvasRef}
        onMouseMove={(e) => {
          const { x, y, row, t } = locate(e);
          const tr = tracks[row];
          if (!tr || x < GUTTER_LEFT) { setTip(tr && x < GUTTER_LEFT ? { x, y, text: tr.name + (tr.note ? ` (${tr.note})` : "") } : null); return; }
          const it = findItem(tr.items, t, (4 / plotW) * (view.x1 - view.x0));
          setTip(it ? { x, y, text: `${tr.name}: ${displayValue(it.value)}\n${fmtSeconds(it.t0)}${it.t1 != null ? ` – ${fmtSeconds(it.t1)} (${(it.t1 - it.t0).toFixed(2)} s)` : " (point)"}\n${it.info}` } : null);
        }}
        onMouseLeave={() => setTip(null)}
        onClick={(e) => {
          const { x, row, t } = locate(e);
          if (x < GUTTER_LEFT) return;
          const tr = tracks[row];
          const it = tr ? findItem(tr.items, t, (4 / plotW) * (view.x1 - view.x0)) : null;
          onClickT(it ? it.t0 : t);
        }}
        onWheel={(e) => {
          const { t } = locate(e);
          const span = view.x1 - view.x0;
          const f = e.deltaY > 0 ? 1.25 : 0.8;
          const ns = Math.max(0.2, span * f);
          const k = (t - view.x0) / span;
          onView({ x0: t - k * ns, x1: t - k * ns + ns });
        }}
      />
      {tip && <div className="tooltip" style={{ left: tip.x + 12, top: tip.y + 8 }}>{tip.text}</div>}
    </div>
  );
}
