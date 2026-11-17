import { useEffect, useState } from "react";
import { api } from "../api";

/** Minimal browser over the backend's allowed roots (no arbitrary filesystem access). */
export default function FileBrowser({ title, accept, onPick, onClose }: {
  title: string; accept: string[]; onPick: (path: string) => void; onClose: () => void;
}) {
  const [cur, setCur] = useState<Awaited<ReturnType<typeof api.browse>> | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const go = (p?: string) => api.browse(p).then((r) => { setCur(r); setErr(null); }).catch((e) => setErr(String(e)));
  useEffect(() => { go(); }, []);
  return (
    <div className="modal-back" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="row space"><b>{title}</b><button onClick={onClose}>Close</button></div>
        <div className="mono small">{cur?.path ?? "(allowed roots)"}</div>
        {err && <div className="callout bad">{err}</div>}
        <ul className="file-list">
          {cur?.path && <li><button className="link" onClick={() => go(cur.parent ?? undefined)}>⬑ up</button></li>}
          {cur?.entries.map((e) => {
            const ok = e.dir || accept.some((x) => e.name.toLowerCase().endsWith(x));
            return (
              <li key={e.path} className={ok ? "" : "dim"}>
                {e.dir ? <button className="link" onClick={() => go(e.path)}>📁 {e.name}</button>
                  : ok ? <button className="link" onClick={() => onPick(e.path)}>{e.name}</button> : <span>{e.name}</span>}
                {!e.dir && e.size != null && <span className="muted small"> {(e.size / 1e6).toFixed(1)} MB</span>}
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
