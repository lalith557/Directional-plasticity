// Theme-aware semantic palette. Each accessor reads from the current CSS
// variables so values track light/dark mode without a React re-render.

function cssVar(name: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

export const C = {
  get ink() { return cssVar("--ink", "#171717"); },
  get ink2() { return cssVar("--ink-2", "#686868"); },
  get ink3() { return cssVar("--ink-3", "#8a8984"); },
  get border() { return cssVar("--border", "#d9d8d3"); },
  get grid() { return cssVar("--border", "#ececE6"); },
  get blue() { return cssVar("--blue", "#315a73"); },
  get blueLight() { return cssVar("--blue", "#7fa6bd"); },
  get red() { return cssVar("--red", "#a8564f"); },
  get green() { return cssVar("--green", "#647a67"); },
  get amber() { return cssVar("--amber", "#a5824d"); },
  get navy() { return cssVar("--navy", "#263b46"); },
  get gray() { return cssVar("--ink-3", "#a8a49c"); },
};

export const METRIC_COLOR = {
  get acc() { return C.blue; },
  get decay() { return C.red; },
  get norm() { return C.navy; },
  get dormant() { return C.amber; },
} as Record<string, string>;

export const REGIME_COLOR = {
  get width16() { return C.blue; },
  get width64() { return C.blue; },
  get width128() { return C.gray; },
  get rot_mlp() { return C.red; },
  get rot_cnn() { return C.amber; },
} as Record<string, string>;

export function corrColor(r: number | null): string {
  if (r === null) return C.gray;
  if (r > 0.15) return C.blue;
  if (r < -0.15) return C.red;
  return C.gray;
}

export const FAMILY_COLOR = {
  get control() { return C.ink2; },
  get magnitude() { return C.green; },
  get directional() { return C.amber; },
  get other() { return C.gray; },
} as Record<string, string>;
