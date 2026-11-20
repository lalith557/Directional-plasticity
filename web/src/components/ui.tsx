import { ReactNode } from "react";

/* ---------- Card ---------- */
export function Card({
  children,
  className = "",
  pad = false,
}: {
  children: ReactNode;
  className?: string;
  pad?: boolean;
}) {
  return <div className={`card ${pad ? "card-pad" : ""} ${className}`}>{children}</div>;
}

export function CardHead({
  title,
  sub,
  right,
}: {
  title: string;
  sub?: string;
  right?: ReactNode;
}) {
  return (
    <div className="card-head">
      <div>
        <h3 className="card-title">{title}</h3>
        {sub && <div className="card-sub">{sub}</div>}
      </div>
      {right}
    </div>
  );
}

/* ---------- Badge ---------- */
export function Badge({
  children,
  tone = "gray",
  dot = false,
}: {
  children: ReactNode;
  tone?: "gray" | "blue" | "green" | "amber" | "red";
  dot?: boolean;
}) {
  const dotColor: Record<string, string> = {
    green: "var(--green)",
    blue: "var(--blue)",
    amber: "var(--amber)",
    red: "var(--red)",
    gray: "var(--ink-3)",
  };
  return (
    <span className={`badge ${tone}`}>
      {dot && <span className="dot" style={{ background: dotColor[tone] }} />}
      {children}
    </span>
  );
}

/* ---------- Observation panel ---------- */
export function Observation({
  children,
  label = "Observation",
  tone = "blue",
}: {
  children: ReactNode;
  label?: string;
  tone?: "blue" | "amber" | "neutral";
}) {
  return (
    <div className={`observation ${tone === "blue" ? "" : tone}`}>
      <div className="obs-label">{label}</div>
      <p>{children}</p>
    </div>
  );
}

/* ---------- Integrity banner ---------- */
export function IntegrityBanner({
  source,
  seeds,
  loaded,
  extra,
}: {
  source: string;
  seeds?: number | string;
  loaded?: string;
  extra?: string;
}) {
  return (
    <div className="integrity">
      <span className="it">
        <b>Source</b> {source}
      </span>
      {seeds !== undefined && (
        <span className="it">
          <b>Seeds</b> {seeds}
        </span>
      )}
      {extra && (
        <span className="it">
          <b></b> {extra}
        </span>
      )}
      {loaded && (
        <span className="it">
          <b>Loaded</b> {loaded}
        </span>
      )}
    </div>
  );
}

/* ---------- Empty / Loading / Error ---------- */
export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <div className="e-title">{title}</div>
      {children}
    </div>
  );
}

export function Loading({ lines = 3, height = 18 }: { lines?: number; height?: number }) {
  return (
    <div className="grid" style={{ gap: 10 }}>
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          className="skeleton"
          style={{ height, width: `${100 - i * 8}%` }}
        />
      ))}
    </div>
  );
}

export function ErrorBox({ source, reason }: { source: string; reason: string }) {
  return (
    <div className="error-box">
      <div className="eb-title">Unable to load experiment data.</div>
      <div className="small">
        <b>Source:</b> <span className="mono">{source}</span>
      </div>
      <div className="small mt2">
        <b>Reason:</b>
      </div>
      <pre>{reason}</pre>
      <div className="small muted mt4">
        The UI does not fall back to mock data. Regenerate the dataset with{" "}
        <span className="mono">python web/scripts/build_data.py</span>.
      </div>
    </div>
  );
}

export function SectionLabel({ children }: { children: ReactNode }) {
  return <div className="section-label">{children}</div>;
}

/* ---------- StatLine ---------- */
export function StatLine({
  stats,
}: {
  stats: { label: string; value: ReactNode }[];
}) {
  return (
    <div className="statline">
      {stats.map((s, i) => (
        <div className="stat" key={i}>
          <span className="sv">{s.value}</span>
          <span className="sl">{s.label}</span>
        </div>
      ))}
    </div>
  );
}
