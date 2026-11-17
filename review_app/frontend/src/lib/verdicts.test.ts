import { describe, expect, it } from "vitest";
import type { Segment } from "../api";
import { madeUnderOtherAlignment, nextSegment, progress, segKey, segmentAtRow, verdictKey, type Verdict } from "./verdicts";

const seg = (a: number, b: number, value: string): Segment =>
  ({ start_row: a, end_row: b, value, start_ts: a * 20, end_ts: b * 20, end_extrapolated: false, split_at_anomaly: false });
const segs = [seg(0, 10, ""), seg(10, 20, "inactif"), seg(20, 35, "Lever_2_Bras"), seg(35, 40, "inactif"), seg(40, 60, "Baisser_2_Bras"), seg(60, 70, "")];
const v = (s: Segment, verdict: Verdict["verdict"], b = 0): Verdict => ({
  id: "x", created: "", updated: "", recording_id: "r", column: 13, column_name: "etiq mvt_bras", start_row: s.start_row,
  end_row: s.end_row, label_value: s.value, verdict, observed: "", comment: "", ts_start: 0, ts_end: 0, t_csv_start: 0,
  t_csv_end: 0, t_video_start: null, t_video_end: null, csv_sha256: "", alignment_snapshot: { a: 1, b, status: "provisional", method: "manual" },
});

describe("verdict helpers", () => {
  it("finds the segment at a row, including boundaries", () => {
    expect(segmentAtRow(segs, 0)?.value).toBe("");
    expect(segmentAtRow(segs, 20)?.value).toBe("Lever_2_Bras");
    expect(segmentAtRow(segs, 34)?.value).toBe("Lever_2_Bras");
    expect(segmentAtRow(segs, 35)?.value).toBe("inactif");
    expect(segmentAtRow(segs, 70)).toBeNull();
  });
  it("segment key matches verdict key", () => {
    expect(segKey(13, segs[2])).toBe(verdictKey(v(segs[2], "agree")));
  });
  it("jumps to the next non-skipped, unreviewed segment", () => {
    const skip = new Set(["", "inactif"]);
    expect(nextSegment(segs, 0, 13, skip)?.value).toBe("Lever_2_Bras");
    const reviewed = new Map([[segKey(13, segs[2]), v(segs[2], "agree")]]);
    expect(nextSegment(segs, 0, 13, skip, reviewed)?.value).toBe("Baisser_2_Bras");
    expect(nextSegment(segs, 45, 13, skip)).toBeNull();
    expect(nextSegment(segs, 0, 13, new Set())?.value).toBe("inactif"); // empty/inactif reviewable when not skipped
  });
  it("counts progress over non-skipped segments", () => {
    const m = new Map([[segKey(13, segs[2]), v(segs[2], "agree")], [segKey(13, segs[4]), v(segs[4], "disagree")]]);
    expect(progress(segs, 13, new Set(["", "inactif"]), m)).toEqual({ total: 2, reviewed: 2, agree: 1, disagree: 1, unsure: 0 });
    expect(progress(segs, 13, new Set(), m).total).toBe(6);
  });
  it("flags verdicts made under another alignment", () => {
    expect(madeUnderOtherAlignment(v(segs[2], "agree", 8.5), 1, 8.5)).toBe(false);
    expect(madeUnderOtherAlignment(v(segs[2], "agree", 8.5), 1, 8.6)).toBe(true);
  });
});
