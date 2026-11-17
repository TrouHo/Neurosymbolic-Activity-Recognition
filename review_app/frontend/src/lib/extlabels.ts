// Mapping of external label-file times onto the CSV-relative timeline, with an explicit unit and reference.
import { csvFromVideo, tCsvFromTimestamp, UNIT_TO_MS, type Model, type TsUnit } from "./time";

export type ExtReference = "absolute" | "csv_relative" | "video";

/**
 * absolute     : value is on the same clock as the CSV master timestamp, in `unit`
 * csv_relative : value is seconds-like time since the CSV reference timestamp, in `unit`
 * video        : value is video media time, in `unit`; it moves with the alignment model
 */
export function extToTCsv(v: number, unit: TsUnit, ref: ExtReference, csvReference: number, csvUnit: TsUnit, m: Model): number {
  const ms = v * UNIT_TO_MS[unit];
  switch (ref) {
    case "absolute": {
      // convert the value into the CSV timestamp's own unit, then to CSV-relative seconds
      const inCsvUnit = ms / UNIT_TO_MS[csvUnit];
      return tCsvFromTimestamp(inCsvUnit, csvReference, csvUnit);
    }
    case "csv_relative":
      return ms / 1000;
    case "video":
      return csvFromVideo(ms / 1000, m);
  }
}

export interface ExtItem { t0: number; t1: number | null; value: string; family: string | null; row: number }
