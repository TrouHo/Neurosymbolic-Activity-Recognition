import { describe, expect, it } from "vitest";
import { checkImport, defaultProfile, type InfoLike } from "./profile";

const info: InfoLike = {
  recording: { id: "cap__user3", video_path: "D:/v.mp4", pairing_rule: "userN" },
  csv: { path: "D:/u3.csv", sha256: "abc", size: 10, encoding: "utf-8-sig", delimiter: ",", decimal: "." },
  columns: [{ index: 0, name: "ts_SQ" }],
  groups: [
    { id: "g0", header: "ts_SB - Main droite - ACC", source: "SB", placement: "Main droite", modality: "ACC", axes: { x: 1 }, flags: { no_nonzero_data: false } },
    { id: "g1", header: "ts_SB - Main droite - GYR", source: "SB", placement: "Main droite", modality: "GYR", axes: { x: 2 }, flags: { no_nonzero_data: true } },
    { id: "g2", header: "sq_D_main", source: "sq", placement: "D_main", modality: "XYZ", axes: { x: 3 }, flags: { no_nonzero_data: false } },
  ],
  proposals: { timestamp: { column: 0, unit: "ms", provenance: "contributor" }, labels: [] },
};

describe("profile", () => {
  const p = defaultProfile(info, 1643189820000, "1643189820000.0", 50, "MP4");

  it("starts unverified with identity model and proposed pairing", () => {
    expect(p.alignment).toMatchObject({ a: 1, b: 0, status: "unverified", method: "default" });
    expect(p.pairing.status).toBe("proposed");
    expect(p.channel_mapping.verified).toBe(false);
  });

  it("preselects ACC/GYR groups that contain data; never empty GYR or Qualisys", () => {
    expect(p.selected_groups).toEqual(["g0"]);
  });

  it("JSON round-trip preserves everything", () => {
    const withAnchors = { ...p, alignment: { ...p.alignment, a: 1.0002, b: -3.14,
      anchors: [{ id: "a1", tVideo: 4, tCsv: 0.86, kind: "signal" as const, description: "lunge", created: "x" }] } };
    const back = JSON.parse(JSON.stringify(withAnchors));
    expect(back).toEqual(withAnchors);
    expect(checkImport(back, { recording_id: "cap__user3", sha256: "abc" })).toEqual({ ok: true, errors: [], warnings: [] });
  });

  it("import warns on a different recording / file and rejects invalid models", () => {
    const r = checkImport({ ...p, recording_id: "other", csv: { ...p.csv, sha256: "zzz" } }, { recording_id: "cap__user3", sha256: "abc" });
    expect(r.ok).toBe(true);
    expect(r.warnings.length).toBe(2);
    const bad = checkImport({ ...p, alignment: { ...p.alignment, a: 0 } }, { recording_id: "cap__user3", sha256: "abc" });
    expect(bad.ok).toBe(false);
  });
});
