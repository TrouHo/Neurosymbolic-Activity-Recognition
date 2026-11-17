import { useEffect, useState, type RefObject } from "react";
import type { VideoMeta } from "../api";
import { approxFrameIndex, fmtSeconds, frameSeekTime } from "../lib/time";

export type ClockSource = "requestVideoFrameCallback (mediaTime)" | "currentTime (timeupdate / animation frame)";

interface Props {
  videoRef: RefObject<HTMLVideoElement>;
  src: string | null;
  meta: VideoMeta | null;
  fps: number | null;
  outputDir: string;
  onTime: (t: number, source: ClockSource) => void;
  onDuration: (d: number) => void;
}

type VFC = (now: number, md: { mediaTime: number }) => void;
type VideoWithVFC = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: VFC) => number;
  cancelVideoFrameCallback?: (h: number) => void;
};

export default function VideoPanel({ videoRef, src, meta, fps, outputDir, onTime, onDuration }: Props) {
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const [error, setError] = useState<string | null>(null);
  const [t, setT] = useState(0);
  const [dur, setDur] = useState(0);

  useEffect(() => {
    const v = videoRef.current as VideoWithVFC | null;
    if (!v) return;
    setError(null);
    let vfcHandle = 0;
    let raf = 0;
    const hasVFC = typeof v.requestVideoFrameCallback === "function";
    const report = (time: number, source: ClockSource) => { setT(time); onTime(time, source); };
    const vfc: VFC = (_now, md) => {
      report(md.mediaTime, "requestVideoFrameCallback (mediaTime)");
      vfcHandle = v.requestVideoFrameCallback!(vfc);
    };
    const rafLoop = () => {
      report(v.currentTime, "currentTime (timeupdate / animation frame)");
      if (!v.paused) raf = requestAnimationFrame(rafLoop);
    };
    const onPlay = () => { setPlaying(true); if (!hasVFC) raf = requestAnimationFrame(rafLoop); };
    const onPause = () => { setPlaying(false); report(v.currentTime, hasVFC ? "requestVideoFrameCallback (mediaTime)" : "currentTime (timeupdate / animation frame)"); };
    // After a seek, rVFC fires when the new frame is presented; also report currentTime immediately.
    const onSeeked = () => { if (!hasVFC || v.paused) report(v.currentTime, hasVFC ? "requestVideoFrameCallback (mediaTime)" : "currentTime (timeupdate / animation frame)"); };
    const onMeta = () => { setDur(v.duration); onDuration(v.duration); };
    const onErr = () => {
      const e = v.error;
      setError(e ? `Browser could not decode/play this video (MediaError code ${e.code}${e.message ? `: ${e.message}` : ""}).` : "Video error.");
    };
    if (hasVFC) vfcHandle = v.requestVideoFrameCallback!(vfc);
    v.addEventListener("play", onPlay);
    v.addEventListener("pause", onPause);
    v.addEventListener("seeked", onSeeked);
    v.addEventListener("loadedmetadata", onMeta);
    v.addEventListener("error", onErr);
    return () => {
      if (hasVFC && vfcHandle) v.cancelVideoFrameCallback?.(vfcHandle);
      cancelAnimationFrame(raf);
      v.removeEventListener("play", onPlay);
      v.removeEventListener("pause", onPause);
      v.removeEventListener("seeked", onSeeked);
      v.removeEventListener("loadedmetadata", onMeta);
      v.removeEventListener("error", onErr);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src]);

  const v = () => videoRef.current!;
  const step = (d: number) => {
    if (!fps) return;
    v().pause();
    const n = approxFrameIndex(v().currentTime, fps) + d;
    v().currentTime = Math.max(0, frameSeekTime(n, fps));
  };
  const jump = (s: number) => { v().currentTime = Math.max(0, Math.min(dur || Infinity, v().currentTime + s)); };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(tag) || !videoRef.current) return;
      if (e.key === " ") { e.preventDefault(); v().paused ? v().play() : v().pause(); }
      else if (e.key === ",") step(-1);
      else if (e.key === ".") step(1);
      else if (e.key === "ArrowLeft") jump(e.shiftKey ? -5 : -1);
      else if (e.key === "ArrowRight") jump(e.shiftKey ? 5 : 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const vt = meta?.video_track;
  const srcName = meta?.path?.split(/[\\/]/).pop() ?? "video.mp4";

  return (
    <div className="video-panel">
      {src ? (
        <video ref={videoRef} src={src} preload="auto" playsInline controls={false} className="video" />
      ) : <div className="video placeholder">No video paired with this recording.</div>}
      {error && (
        <div className="callout bad">
          <b>{error}</b>
          <div>Container metadata: codec {vt?.codec ?? "?"}, {vt?.coded_wh?.join("×") ?? "?"}. The original file is never modified.</div>
          <div>Optional local conversion (not run by this app; ffmpeg is not installed here). It writes a NEW file:</div>
          <code>ffmpeg -i "{meta?.path}" -c:v libx264 -pix_fmt yuv420p -an "{outputDir}\converted\{srcName.replace(/\.[^.]+$/, "")}_h264.mp4"</code>
        </div>
      )}
      <div className="transport">
        <button onClick={() => (v().paused ? v().play() : v().pause())} disabled={!src}>{playing ? "Pause" : "Play"}</button>
        <button onClick={() => step(-1)} disabled={!fps} title="Approximate frame step (key ,)">◀ frame</button>
        <button onClick={() => step(1)} disabled={!fps} title="Approximate frame step (key .)">frame ▶</button>
        <button onClick={() => jump(-1)} disabled={!src}>−1 s</button>
        <button onClick={() => jump(1)} disabled={!src}>+1 s</button>
        <label>Speed&nbsp;
          <select value={rate} onChange={(e) => { const r = Number(e.target.value); setRate(r); v().playbackRate = r; }}>
            {[0.1, 0.25, 0.5, 1, 1.5, 2].map((r) => <option key={r} value={r}>{r}×</option>)}
          </select>
        </label>
        <span className="mono">{fmtSeconds(t)} / {fmtSeconds(dur)}</span>
      </div>
      <input type="range" className="seek" min={0} max={dur || 0} step={0.001} value={Math.min(t, dur || 0)}
        onChange={(e) => { v().currentTime = Number(e.target.value); }} disabled={!src} />
      <div className="hint">Frame stepping is approximate: it seeks to the middle of frame n at the container's nominal FPS
        ({fps ? fps.toFixed(5) : "unknown"}); B-frames, edit lists and zero-duration first samples may shift the presented frame.
        Keys: Space play/pause, , . frame, ←/→ 1 s (Shift 5 s).</div>
    </div>
  );
}
