// Stable colors for label values. "" (empty) is not colored (drawn hatched); "inactif" is neutral gray.
const PALETTE = [
  "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#76b7b2", "#edc948", "#9c755f",
  "#ff9da7", "#3b8bc2", "#8cd17d", "#d37295", "#499894", "#b6992d", "#86bcb6", "#f1ce63",
];

export function labelColor(value: string): string | null {
  if (value === "") return null;
  if (value === "inactif") return "#c9ccd1";
  let h = 2166136261;
  for (let i = 0; i < value.length; i++) h = Math.imul(h ^ value.charCodeAt(i), 16777619);
  return PALETTE[Math.abs(h) % PALETTE.length];
}

export const AXIS_COLORS: Record<string, string> = { x: "#d62728", y: "#2ca02c", z: "#1f77b4" };

export function displayValue(v: string): string {
  return v === "" ? "(empty)" : v;
}
