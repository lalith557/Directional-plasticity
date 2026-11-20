// Formatting helpers. "Not available" / "N/A" are distinct from 0 (section 46).

export const NA = "Not available";

export function acc(v: number | null | undefined, d = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return NA;
  return v.toFixed(d);
}

export function meanStd(m: number | null | undefined, s: number | null | undefined, d = 3): string {
  if (m === null || m === undefined || Number.isNaN(m)) return NA;
  if (s === null || s === undefined || Number.isNaN(s)) return m.toFixed(d);
  return `${m.toFixed(d)} ± ${s.toFixed(d)}`;
}

export function num(v: number | null | undefined, d = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return NA;
  return v.toFixed(d);
}

export function pct(v: number | null | undefined, d = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return NA;
  return `${(v * 100).toFixed(d)}%`;
}

export function signed(v: number | null | undefined, d = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return NA;
  return (v >= 0 ? "+" : "") + v.toFixed(d);
}

// Scientific notation for very small deviations (matched-step).
export function sci(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return NA;
  if (v === 0) return "0";
  const e = v.toExponential(1); // e.g. 2.2e-7
  return e.replace("e", "×10^").replace("+", "");
}

// lambda axis label (0, 1e-3, ...)
export function lamLabel(v: number): string {
  if (v === 0) return "0";
  const exp = Math.round(Math.log10(v));
  const mant = v / Math.pow(10, exp);
  if (Math.abs(mant - 1) < 1e-6) return `1e${exp}`;
  return `${Math.round(mant)}e${exp}`;
}

export function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function timestamp(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}
