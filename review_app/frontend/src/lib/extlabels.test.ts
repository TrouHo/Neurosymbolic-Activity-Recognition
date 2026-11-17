import { describe, expect, it } from "vitest";
import { extToTCsv } from "./extlabels";

const REF = 1643189820000; // CSV reference, Unix ms

describe("external label clock mapping", () => {
  it("absolute Unix ms on the CSV clock", () => {
    expect(extToTCsv(1643189928380, "ms", "absolute", REF, "ms", { a: 1, b: 0 })).toBeCloseTo(108.38, 9);
  });
  it("absolute Unix seconds against a ms CSV clock", () => {
    expect(extToTCsv(1643189928.38, "s", "absolute", REF, "ms", { a: 1, b: 0 })).toBeCloseTo(108.38, 3);
  });
  it("CSV-relative seconds ignore the alignment model", () => {
    expect(extToTCsv(12.5, "s", "csv_relative", REF, "ms", { a: 1, b: 99 })).toBe(12.5);
  });
  it("video-time labels follow the alignment model", () => {
    expect(extToTCsv(4000, "ms", "video", REF, "ms", { a: 1, b: 108.3 })).toBeCloseTo(112.3, 9);
    expect(extToTCsv(4, "s", "video", REF, "ms", { a: 1.001, b: -2 })).toBeCloseTo(2.004, 9);
  });
});
