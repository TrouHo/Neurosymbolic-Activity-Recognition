import { describe, expect, it } from "vitest";
import {
  approxFrameIndex, csvCoverage, csvFromVideo, frameSeekTime, nearestIndex, oneAnchor, residuals,
  sampleIndexAt, tCsvFromTimestamp, timestampFromTCsv, twoAnchor, videoCoverage, videoFromCsv,
} from "./time";

// Realistic Unix-ms values (user3, 2022 capture).
const REF = 1643189820000;
const step = 0.02;
const tGrid = Float64Array.from({ length: 45001 }, (_, i) => tCsvFromTimestamp(REF + 20 * i, REF, "ms"));

describe("timestamps", () => {
  it("converts Unix ms exactly at realistic magnitudes", () => {
    expect(tCsvFromTimestamp(1643189820020, REF, "ms")).toBeCloseTo(0.02, 12);
    expect(tCsvFromTimestamp(1643190720000, REF, "ms")).toBe(900);
    expect(timestampFromTCsv(108.38, REF, "ms")).toBe(1643189928380);
    expect(tCsvFromTimestamp(1643189821, 1643189820, "s")).toBe(1);
  });
});

describe("alignment model", () => {
  it("defaults to identity", () => {
    expect(csvFromVideo(12.5, { a: 1, b: 0 })).toBe(12.5);
  });
  it("positive b: video t=0 maps to a later CSV time", () => {
    const m = { a: 1, b: 108.3 };
    expect(csvFromVideo(0, m)).toBe(108.3);
    expect(videoFromCsv(108.3, m)).toBe(0);
    expect(videoFromCsv(110.3, m)).toBeCloseTo(2, 12);
  });
  it("negative b round-trips", () => {
    const m = { a: 1, b: -3.25 };
    for (const tv of [0, 1.234, 590.14]) expect(videoFromCsv(csvFromVideo(tv, m), m)).toBeCloseTo(tv, 9);
  });
  it("rejects non-positive scale on inverse", () => {
    expect(() => videoFromCsv(1, { a: 0, b: 0 })).toThrow();
    expect(() => videoFromCsv(1, { a: -1, b: 0 })).toThrow();
  });
  it("one anchor keeps a and solves b", () => {
    const m = oneAnchor({ tVideo: 4.0, tCsv: 112.42 }, 1);
    expect(m.a).toBe(1);
    expect(m.b).toBeCloseTo(108.42, 12);
  });
  it("two anchors recover drift exactly", () => {
    const truth = { a: 1.0005, b: 108.3 };
    const p = { tVideo: 5, tCsv: csvFromVideo(5, truth) };
    const q = { tVideo: 580, tCsv: csvFromVideo(580, truth) };
    const r = twoAnchor(q, p); // order given reversed on purpose
    expect(r.errors).toEqual([]);
    expect(r.model!.a).toBeCloseTo(truth.a, 12);
    expect(r.model!.b).toBeCloseTo(truth.b, 9);
    expect(residuals([p, q], r.model!).every((x) => Math.abs(x) < 1e-9)).toBe(true);
  });
  it("two anchors: invalid ordering and degenerate cases are errors", () => {
    expect(twoAnchor({ tVideo: 1, tCsv: 10 }, { tVideo: 5, tCsv: 9 }).errors.length).toBe(1);
    expect(twoAnchor({ tVideo: 1, tCsv: 10 }, { tVideo: 1, tCsv: 12 }).errors.length).toBe(1);
    expect(twoAnchor({ tVideo: 1, tCsv: 10 }, { tVideo: 5, tCsv: 10 }).errors.length).toBe(1);
  });
  it("two anchors: warns on large scale deviation", () => {
    const r = twoAnchor({ tVideo: 0, tCsv: 0 }, { tVideo: 100, tCsv: 105 });
    expect(r.model!.a).toBeCloseTo(1.05);
    expect(r.warnings.some((w) => w.includes("Scale"))).toBe(true);
  });
});

describe("coverage and sample lookup", () => {
  it("does not pin out-of-range times to an endpoint", () => {
    expect(csvCoverage(-0.001, 0, 900, step)).toBe("before");
    expect(csvCoverage(900.019, 0, 900, step)).toBe("inside");
    expect(csvCoverage(900.02, 0, 900, step)).toBe("after");
    expect(videoCoverage(-0.1, 590.14)).toBe("before");
    expect(videoCoverage(600, 590.14)).toBe("after");
    expect(csvCoverage(NaN, 0, 900, step)).toBe("invalid");
    expect(sampleIndexAt(tGrid, -0.5)).toBe(-1);
  });
  it("finds the containing and nearest sample", () => {
    expect(sampleIndexAt(tGrid, 108.38)).toBe(5419);
    expect(sampleIndexAt(tGrid, 108.399)).toBe(5419);
    expect(nearestIndex(tGrid, 108.389)).toBe(5419);
    expect(nearestIndex(tGrid, 108.391)).toBe(5420);
    expect(sampleIndexAt(tGrid, 1e9)).toBe(45000);
  });
  it("clicking a known timestamp seeks to the expected video time (inverse mapping)", () => {
    const m = { a: 1, b: 108.3 };
    const tc = tCsvFromTimestamp(1643189928380, REF, "ms"); // row 5419
    expect(videoFromCsv(tc, m)).toBeCloseTo(0.08, 9);
    const m2 = { a: 1, b: -2 };
    expect(videoFromCsv(tc, m2)).toBeCloseTo(110.38, 9);
  });
});

describe("approximate frames", () => {
  it("frame index and seek target are consistent", () => {
    for (const fps of [30, 50]) {
      for (const n of [0, 1, 99, 12635]) {
        expect(approxFrameIndex(frameSeekTime(n, fps), fps)).toBe(n);
        expect(approxFrameIndex(n / fps, fps)).toBe(n);
      }
    }
  });
});
