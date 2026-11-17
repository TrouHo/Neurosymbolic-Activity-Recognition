import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

export const GUTTER_LEFT = 72;   // y-axis width, shared with label tracks / overview for x alignment
export const GUTTER_RIGHT = 12;

export interface Band { t0: number; t1: number; color: string }
export interface View { x0: number; x1: number }

interface Props {
  title: string;
  badges: { text: string; tone: "warn" | "info" | "bad" }[];
  x: Float64Array;
  ys: { label: string; data: (number | null)[]; color: string }[];
  view: View;
  onView: (v: View) => void;
  cursorT: number | null;
  markers: { t: number; color: string; title: string }[];
  bands: Band[];
  onClickT: (t: number) => void;
  onSelect: (t0: number, t1: number) => void;
  height?: number;
}

/** One uPlot instance. Full-resolution data (no downsampling, gaps kept as null). */
export default function SignalPlot(p: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);
  const cursorRef = useRef<HTMLDivElement | null>(null);
  const markerLayerRef = useRef<HTMLDivElement | null>(null);
  const bandsRef = useRef<Band[]>(p.bands);
  const cb = useRef(p);
  cb.current = p;

  useEffect(() => {
    const el = wrapRef.current!;
    const opts: uPlot.Options = {
      width: Math.max(300, el.clientWidth),
      height: p.height ?? 150,
      padding: [6, GUTTER_RIGHT, 0, 0],
      legend: { show: true, live: true },
      cursor: { drag: { x: true, y: false, setScale: false }, points: { show: false } },
      scales: { x: { time: false, min: p.view.x0, max: p.view.x1 } },
      axes: [
        { size: 30, font: "11px system-ui", values: (_u, vals) => vals.map((v) => v.toFixed(Math.abs(vals[1] - vals[0]) < 1 ? 2 : 0)) },
        { size: GUTTER_LEFT, font: "11px system-ui", label: "unit unknown", labelSize: 14, labelFont: "10px system-ui" },
      ],
      series: [
        { label: "t_csv (s)", value: (_u, v) => (v == null ? "—" : v.toFixed(3)) },
        // Not hovering -> idx is null -> "—". Hovering a missing value -> "gap".
        ...p.ys.map((s) => ({ label: s.label, stroke: s.color, width: 1, spanGaps: false,
          value: (_u: uPlot, v: number | null, _si: number, idx: number | null) => (idx == null ? "—" : v == null ? "gap" : String(v)) })),
      ],
      hooks: {
        drawClear: [(u) => {
          const ctx = u.ctx;
          const { top, height } = u.bbox;
          for (const b of bandsRef.current) {
            const x0 = u.valToPos(b.t0, "x", true);
            const x1 = u.valToPos(b.t1, "x", true);
            if (x1 < u.bbox.left || x0 > u.bbox.left + u.bbox.width) continue;
            ctx.fillStyle = b.color;
            ctx.fillRect(Math.max(x0, u.bbox.left), top, Math.max(1, Math.min(x1, u.bbox.left + u.bbox.width) - Math.max(x0, u.bbox.left)), height);
          }
        }],
        setSelect: [(u) => {
          if (u.select.width > 3) {
            const t0 = u.posToVal(u.select.left, "x");
            const t1 = u.posToVal(u.select.left + u.select.width, "x");
            cb.current.onSelect(t0, t1);
          }
          u.setSelect({ left: 0, top: 0, width: 0, height: 0 }, false);
        }],
      },
    };
    const data = [p.x, ...p.ys.map((s) => s.data)] as unknown as uPlot.AlignedData;
    const u = new uPlot(opts, data, el);
    plotRef.current = u;

    const cursor = document.createElement("div");
    cursor.className = "plot-cursor";
    u.over.appendChild(cursor);
    cursorRef.current = cursor;
    const markers = document.createElement("div");
    markers.className = "plot-markers";
    u.over.appendChild(markers);
    markerLayerRef.current = markers;

    let downX = 0;
    const onDown = (e: MouseEvent) => { downX = e.clientX; };
    const onUp = (e: MouseEvent) => {
      if (Math.abs(e.clientX - downX) < 3) {
        const r = u.over.getBoundingClientRect();
        cb.current.onClickT(u.posToVal(e.clientX - r.left, "x"));
      }
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = u.over.getBoundingClientRect();
      const { x0, x1 } = cb.current.view;
      const span = x1 - x0;
      if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
        const d = Math.sign(e.deltaX || e.deltaY) * span * 0.1;
        cb.current.onView({ x0: x0 + d, x1: x1 + d });
      } else {
        const at = u.posToVal(e.clientX - r.left, "x");
        const f = e.deltaY > 0 ? 1.25 : 0.8;
        const ns = Math.max(0.2, span * f);
        const k = (at - x0) / span;
        cb.current.onView({ x0: at - k * ns, x1: at - k * ns + ns });
      }
    };
    u.over.addEventListener("mousedown", onDown);
    u.over.addEventListener("mouseup", onUp);
    u.over.addEventListener("wheel", onWheel, { passive: false });
    const ro = new ResizeObserver(() => u.setSize({ width: Math.max(300, el.clientWidth), height: p.height ?? 150 }));
    ro.observe(el);
    return () => { ro.disconnect(); u.destroy(); plotRef.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [p.x, p.ys]);

  useEffect(() => {
    const u = plotRef.current;
    if (!u) return;
    if (u.scales.x.min !== p.view.x0 || u.scales.x.max !== p.view.x1) u.setScale("x", { min: p.view.x0, max: p.view.x1 });
  }, [p.view]);

  useEffect(() => {
    bandsRef.current = p.bands;
    plotRef.current?.redraw(false, false);
  }, [p.bands]);

  useEffect(() => {
    const u = plotRef.current;
    const c = cursorRef.current;
    if (!u || !c) return;
    if (p.cursorT == null || p.cursorT < p.view.x0 || p.cursorT > p.view.x1) { c.style.display = "none"; return; }
    c.style.display = "block";
    c.style.left = `${u.valToPos(p.cursorT, "x")}px`;
  }, [p.cursorT, p.view]);

  useEffect(() => {
    const u = plotRef.current;
    const layer = markerLayerRef.current;
    if (!u || !layer) return;
    layer.innerHTML = "";
    for (const m of p.markers) {
      if (m.t < p.view.x0 || m.t > p.view.x1) continue;
      const d = document.createElement("div");
      d.className = "plot-marker";
      d.style.left = `${u.valToPos(m.t, "x")}px`;
      d.style.borderColor = m.color;
      d.title = m.title;
      layer.appendChild(d);
    }
  }, [p.markers, p.view]);

  return (
    <div className="signal-plot">
      <div className="plot-title">
        <span>{p.title}</span>
        {p.badges.map((b) => <span key={b.text} className={`badge ${b.tone}`}>{b.text}</span>)}
      </div>
      <div ref={wrapRef} className="plot-wrap" />
    </div>
  );
}
