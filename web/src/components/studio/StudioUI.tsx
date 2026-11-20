import { ReactNode } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { C } from "../../lib/colors";
import { Badge } from "../ui";
import { IS_HOSTED, REPO_URL, useBackend } from "../../lib/api";
import { NA } from "../../data/format";

/* ---------------- badges ---------------- */
const STATUS_TONE: Record<string, "gray" | "blue" | "green" | "amber" | "red"> = {
  proposed: "gray", queued: "gray", initializing: "blue", running: "blue", completed: "green",
  failed: "red", cancelled: "amber", completed_diverged: "amber", superseded: "gray", incomplete: "amber",
};
const STATUS_TEXT: Record<string, string> = {
  completed_diverged: "completed · diverged", incomplete: "incomplete (no artifact)",
};

export function StatusBadge({ status }: { status: string }) {
  return (
    <Badge tone={STATUS_TONE[status] ?? "gray"} dot>
      {STATUS_TEXT[status] ?? status}
    </Badge>
  );
}

export function SourceBadge({ source }: { source: string }) {
  if (source === "legacy_artifact") return <Badge tone="gray">existing artifact</Badge>;
  if (source === "studio_group") return <Badge tone="blue">studio group</Badge>;
  return <Badge tone="blue">studio run</Badge>;
}

const VAL_TEXT: Record<string, [string, "gray" | "blue" | "green" | "amber" | "red"]> = {
  artifact: ["artifact (not re-run)", "gray"],
  missing: ["missing artifact", "amber"],
  not_validated: ["not validated", "gray"],
  no_legacy_counterpart: ["new result (no legacy counterpart)", "blue"],
  reproduced_exact: ["reproduced · exact", "green"],
  reproduced_within_tolerance: ["reproduced · within 1e-4", "green"],
  mismatch: ["reproduction mismatch", "red"],
  not_comparable: ["not comparable", "gray"],
};

export function ValidationBadge({ v }: { v?: { status: string } | null }) {
  if (!v) return <span className="na">{NA}</span>;
  const [t, tone] = VAL_TEXT[v.status] ?? [v.status, "gray"];
  return <Badge tone={tone}>{t}</Badge>;
}

/* ---------------- backend gate ---------------- */
export function BackendGate({ children, feature }: { children: ReactNode; feature: string }) {
  const { online } = useBackend(4000);
  if (IS_HOSTED)
    return (
      <div className="error-box">
        <div className="eb-title">This hosted version is read-only — {feature} needs the local Studio backend.</div>
        <div className="small">
          The Experimental Studio launches and inspects real PyTorch training runs on a GPU, so it runs on
          your own machine. Clone the repository and follow the{" "}
          <a className="linkish" href={`${REPO_URL}#quick-start`} target="_blank" rel="noreferrer">Quick start</a>{" "}
          in the README to use it.
        </div>
        <div className="small muted mt2">
          The experiment pages, the norm–plasticity, geometry and ablation analyses, the documentation, the
          system architecture and the paper are fully available here; their values come from the released
          result files.
        </div>
      </div>
    );
  if (online === null) return <div className="studio-note">Connecting to the Studio backend…</div>;
  if (!online)
    return (
      <div className="error-box">
        <div className="eb-title">Studio backend is not running — {feature} needs it.</div>
        <div className="small">Start it from the project root:</div>
        <pre>python -m uvicorn studio.api.server:app --host 127.0.0.1 --port 8765</pre>
        <div className="small muted mt2">
          No values are simulated while the backend is offline. Pages built from released artifacts
          (Experiments, Analysis) keep working without it.
        </div>
      </div>
    );
  return <>{children}</>;
}

/* ---------------- source caption ---------------- */
export function SourceNote({ children }: { children: ReactNode }) {
  return <div className="source-note"><b>Data source</b> {children}</div>;
}

/* ---------------- progress ---------------- */
export function Progress({ value, max, label }: { value: number; max: number; label?: string }) {
  const p = max > 0 ? Math.min(100, (100 * value) / max) : 0;
  return (
    <div className="progress-wrap">
      {label && (
        <div className="row between small mb2">
          <span>{label}</span>
          <span className="mono">{value} / {max}</span>
        </div>
      )}
      <div className="progress" role="progressbar" aria-valuenow={value} aria-valuemax={max}>
        <div className="progress-fill" style={{ width: `${p}%` }} />
      </div>
    </div>
  );
}

/* ---------------- key/value ---------------- */
export function KV({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <div className="kv">
      {rows.map(([k, v]) => (
        <div key={k} style={{ display: "contents" }}>
          <div className="kv-k">{k}</div>
          <div className="kv-v">{v ?? <span className="na">{NA}</span>}</div>
        </div>
      ))}
    </div>
  );
}

/* ---------------- metric formatting ---------------- */
export function fmtMetric(v: number | null | undefined, key: string): string {
  if (v === null || v === undefined || Number.isNaN(v)) return NA;
  if (key === "matched_max_dev" || key === "cor1_relative_error")
    return v === 0 ? "0" : v.toExponential(1);
  if (key === "final_weight_norm") return v.toFixed(2);
  if (key === "final_dormant") return `${(v * 100).toFixed(1)}%`;
  if (key === "final_eff_rank") return v.toFixed(2);
  return v.toFixed(3);
}

export function MeanStd({ m, k }: { m?: { mean: number | null; std: number | null } | null; k: string }) {
  if (!m || m.mean === null || m.mean === undefined) return <span className="na">{NA}</span>;
  return (
    <span className="mono">
      {fmtMetric(m.mean, k)}
      {m.std !== null && m.std !== undefined && k !== "final_dormant" && (
        <span className="std"> ± {fmtMetric(m.std, k)}</span>
      )}
    </span>
  );
}

export const METRIC_LABEL: Record<string, string> = {
  mean_acc: "Mean acc ↑", decay: "Decay ↓", final_weight_norm: "Final ‖θ‖", final_dormant: "Dormant",
  final_eff_rank: "Eff. rank", matched_max_dev: "Match dev.", peak_acc: "Peak acc", end_acc: "Tail acc",
};

/* ---------------- per-task series ---------------- */
export interface Series {
  key: string;
  name: string;
  color: string;
  mean: (number | null)[];
  std?: (number | null)[] | null;
}

export function SeriesChart({
  series,
  height = 230,
  yLabel,
  yFmt,
  xLabel = "task",
}: {
  series: Series[];
  height?: number;
  yLabel?: string;
  yFmt?: (v: number) => string;
  xLabel?: string;
}) {
  const T = Math.max(0, ...series.map((s) => s.mean.length));
  const data = Array.from({ length: T }, (_, t) => {
    const row: any = { t: t + 1 };
    for (const s of series) {
      const m = s.mean[t];
      row[s.key] = m ?? null;
      const sd = s.std?.[t];
      row[`${s.key}__band`] = m != null && sd != null ? [m - sd, m + sd] : null;
    }
    return row;
  });
  if (!T) return <div className="studio-note">No per-task values yet.</div>;
  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 18, left: 4 }}>
          <CartesianGrid stroke={C.grid} vertical={false} />
          <XAxis dataKey="t" stroke={C.border} tick={{ fill: C.ink3, fontSize: 11 }} interval="preserveStartEnd"
                 minTickGap={24} label={{ value: xLabel, position: "insideBottom", dy: 14, fill: C.ink3, fontSize: 11 }} />
          <YAxis stroke={C.border} tick={{ fill: C.ink3, fontSize: 11 }} width={50} tickFormatter={yFmt}
                 domain={["auto", "auto"]}
                 label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", fill: C.ink3, fontSize: 11 } : undefined} />
          <Tooltip
            contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", fontSize: 12 }}
            formatter={(v: any, n: any) => (Array.isArray(v) ? [`${v[0].toFixed(3)} – ${v[1].toFixed(3)}`, `${n} ±1 sd`] :
              [typeof v === "number" ? (yFmt ? yFmt(v) : v.toFixed(4)) : v, n])}
            labelFormatter={(l) => `task ${l}`}
          />
          {series.length > 1 && <Legend verticalAlign="top" height={24} iconType="plainline" />}
          {series.map((s) =>
            s.std ? (
              <Area key={`${s.key}b`} dataKey={`${s.key}__band`} stroke="none" fill={s.color} fillOpacity={0.12}
                    isAnimationActive={false} legendType="none" name={s.name} />
            ) : null
          )}
          {series.map((s) => (
            <Line key={s.key} dataKey={s.key} name={s.name} stroke={s.color} strokeWidth={1.7} dot={T <= 12}
                  isAnimationActive={false} connectNulls={false} />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export const PALETTE = () => [C.blue, C.red, C.green, C.amber, C.navy, C.ink2, "#7b6aa8", "#3f8f8a"];
