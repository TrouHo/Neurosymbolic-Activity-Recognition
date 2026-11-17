import { useEffect, useRef, useState } from "react";
import { fmtSeconds } from "../lib/time";
import { GUTTER_LEFT, GUTTER_RIGHT, type Band, type View } from "./SignalPlot";

interface Props {
  extent: View;                         // full navigable range (CSV-relative seconds)
  csv: Band;                            // CSV coverage
  video: Band | null;                   // video coverage mapped through the current model
  view: View;
  cursorT: number | null;
  anchors: number[];
  selection: Band | null;
  onView: (v: View) => void;
}

/** Whole-recording strip: CSV coverage vs mapped video coverage, current viewport, cursor. */
export default function Overview(p: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(800);
  const H = 44;

  useEffect(() => {
    const ro = new ResizeObserver(() => setW(wrap.current?.clientWidth ?? 800));
    ro.observe(wrap.current!);
    return () => ro.disconnect();
  }, []);

  const pw = w - GUTTER_LEFT - GUTTER_RIGHT;
  const span = p.extent.x1 - p.extent.x0;
  const X = (t: number) => GUTTER_LEFT + ((t - p.extent.x0) / span) * pw;

  useEffect(() => {
    const c = ref.current!;
    const dpr = window.devicePixelRatio || 1;
    c.width = w * dpr; c.height = H * dpr; c.style.width = `${w}px`; c.style.height = `${H}px`;
    const ctx = c.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, H);
    ctx.font = "10px system-ui";
    ctx.fillStyle = "#333";
    ctx.fillText("CSV", 4, 13);
    ctx.fillText("Video (mapped)", 4, 29);
    ctx.fillStyle = "#3b6fb6";
    ctx.fillRect(X(p.csv.t0), 5, X(p.csv.t1) - X(p.csv.t0), 10);
    if (p.video) {
      ctx.fillStyle = "#e08a1e";
      ctx.fillRect(X(p.video.t0), 21, X(p.video.t1) - X(p.video.t0), 10);
    }
    if (p.selection) {
      ctx.fillStyle = "rgba(255,200,0,0.35)";
      ctx.fillRect(X(p.selection.t0), 0, Math.max(1, X(p.selection.t1) - X(p.selection.t0)), H);
    }
    ctx.strokeStyle = "#111";
    ctx.lineWidth = 1;
    ctx.strokeRect(X(p.view.x0), 1.5, Math.max(2, X(p.view.x1) - X(p.view.x0)), 33);
    for (const a of p.anchors) {
      ctx.fillStyle = "#7b2cbf";
      ctx.beginPath(); ctx.moveTo(X(a), 35); ctx.lineTo(X(a) - 4, 43); ctx.lineTo(X(a) + 4, 43); ctx.fill();
    }
    if (p.cursorT != null) {
      ctx.strokeStyle = "#e0007a"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(X(p.cursorT), 0); ctx.lineTo(X(p.cursorT), 35); ctx.stroke();
    }
    ctx.fillStyle = "#555";
    ctx.fillText(fmtSeconds(p.extent.x0, 1), GUTTER_LEFT, 43);
    const endLbl = fmtSeconds(p.extent.x1, 1);
    ctx.fillText(endLbl, w - GUTTER_RIGHT - ctx.measureText(endLbl).width, 43);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [p, w]);

  return (
    <div ref={wrap} className="overview" title="Click to center the view here. Blue: CSV rows. Orange: video duration mapped with the current a, b.">
      <canvas ref={ref} onClick={(e) => {
        const r = ref.current!.getBoundingClientRect();
        const t = p.extent.x0 + ((e.clientX - r.left - GUTTER_LEFT) / pw) * span;
        const half = (p.view.x1 - p.view.x0) / 2;
        p.onView({ x0: t - half, x1: t + half });
      }} />
    </div>
  );
}
