/** Least-squares fit in log10(x) space -> two endpoints for a trend line. */
export function logTrend(points: { norm: number; decay: number }[]): { x: number; y: number }[] {
  const pts = points.filter((p) => p.norm > 0 && p.decay != null);
  if (pts.length < 2) return [];
  const xs = pts.map((p) => Math.log10(p.norm));
  const ys = pts.map((p) => p.decay);
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (xs[i] - mx) * (ys[i] - my);
    den += (xs[i] - mx) ** 2;
  }
  const m = num / den;
  const b = my - m * mx;
  const xmin = Math.min(...pts.map((p) => p.norm));
  const xmax = Math.max(...pts.map((p) => p.norm));
  return [
    { x: xmin, y: m * Math.log10(xmin) + b },
    { x: xmax, y: m * Math.log10(xmax) + b },
  ];
}
