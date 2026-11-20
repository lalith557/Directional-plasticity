import { Fragment, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { DataGate } from "../components/DataGate";
import { PageHeader } from "../components/PageHeader";
import { Tabs } from "../components/Tabs";
import { Card, CardHead, Badge, Observation, IntegrityBanner } from "../components/ui";
import { ResultTable, Column } from "../components/ResultTable";
import { ConfigPanel } from "../components/ConfigPanel";
import { BarSci } from "../components/charts";
import { C } from "../lib/colors";
import { acc, num, pct, sci, timestamp } from "../data/format";
import type { Arm, R20 as R20Type } from "../data/types";

const TABS = ["Setup", "Results", "Runs", "Config", "Notes"];

function MatchDevCell({ arm }: { arm: Arm }) {
  if (arm.matchStep === "applicable")
    return (
      <span title="Update renormalized to the plain-SGD step norm; deviation confirms matched step.">
        {sci(arm.matchDev)}
      </span>
    );
  if (arm.matchStep === "reference")
    return (
      <span className="na" title="SGD defines the reference step norm.">
        ref
      </span>
    );
  return (
    <span
      className="na"
      title="Magnitude/reset method: update magnitude is set by the method itself, so matched-step control is not applicable."
    >
      N/A
    </span>
  );
}

export function R20() {
  const [params, setParams] = useSearchParams();
  const nav = useNavigate();
  const tabParam = params.get("tab");
  const initial = TABS.find((t) => t.toLowerCase() === (tabParam ?? "").toLowerCase()) ?? "Setup";
  const [tab, setTab] = useState(initial);
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const setTabAndUrl = (t: string) => {
    setTab(t);
    setParams(t === "Setup" ? {} : { tab: t.toLowerCase() }, { replace: true });
  };

  return (
    <DataGate>
      {(d) => {
        const r = d.r20;
        return (
          <>
            <PageHeader
              title={r.name}
              subtitle={r.subtitle}
              badge={<Badge tone="green" dot>{r.status}</Badge>}
            />
            <Tabs tabs={TABS} active={tab} onChange={setTabAndUrl} />

            {tab === "Setup" && <Setup r={r} goTab={setTabAndUrl} />}
            {tab === "Results" && (
              <Results
                r={r}
                selected={selected}
                setSelected={setSelected}
                goTab={setTabAndUrl}
                seedN={d.meta.seedsPerArm}
              />
            )}
            {tab === "Runs" && <Runs r={r} nav={nav} />}
            {tab === "Config" && <ConfigTab r={r} />}
            {tab === "Notes" && <Notes r={r} />}

            <IntegrityBanner
              source={r.source}
              seeds={d.meta.seedsPerArm}
              loaded={timestamp(d.generatedAt)}
              extra={`matched step verified to ${d.meta.matchedStepTolerance}`}
            />
          </>
        );
      }}
    </DataGate>
  );
}

/* ------------------------------- Setup ---------------------------------- */
function Setup({ r, goTab }: { r: R20Type; goTab: (t: string) => void }) {
  const cfg = r.config;
  const rows: [string, string | number][] = [
    ["Dataset", cfg.dataset],
    ["Architecture", cfg.architecture],
    ["Tasks", cfg.tasks],
    ["Steps per task", cfg.stepsPerTask],
    ["Learning rate", cfg.lr],
    ["Training subset", `${cfg.trainSubset.toLocaleString()} examples`],
    ["Batch size", cfg.batch],
    ["Classifier", cfg.classifier],
    ["Seeds", cfg.seeds],
    ["Epochs", `${cfg.epochs} (online)`],
  ];
  return (
    <>
      {/* Responsive two-column: Config left, Task Seq + Protocol right */}
      <div className="grid-2col">
        <Card>
          <CardHead title="Experiment Configuration" />
          <div className="card-pad">
            <div className="kv">
              {rows.map(([k, v]) => (
                <Fragment key={k}>
                  <div className="kv-k">{k}</div>
                  <div className="kv-v">{v}</div>
                </Fragment>
              ))}
            </div>
          </div>
        </Card>

        <div className="grid" style={{ gap: 16 }}>
          <Card>
            <CardHead title="Task Sequence" sub="Illustration" />
            <div className="card-pad">
              <div className="taskseq">
                {[1, 2, 3].map((i) => (
                  <TaskChip key={i} i={i} />
                ))}
                <span className="task-ellipsis">&middot;&middot;&middot;</span>
                <TaskChip i={cfg.tasks} />
              </div>
              <div className="small muted" style={{ marginTop: 20 }}>
                Each task applies a fixed random pixel permutation with labels unchanged; a
                single shared 10-way head isolates sequential adaptation. Per-task accuracy
                curves are not recorded in the released summaries.
              </div>
            </div>
          </Card>

          <Card>
            <CardHead title="Training Protocol" />
            <div className="card-pad">
              <ol style={{ margin: 0, paddingLeft: 18, lineHeight: 1.9, fontSize: 13 }}>
                <li>Warm-start a shared model.</li>
                <li>
                  Train sequentially on {cfg.tasks} permuted tasks ({cfg.stepsPerTask} steps
                  each).
                </li>
                <li>Evaluate per-task trainability (accuracy and plasticity decay).</li>
                <li>
                  Repeat for {cfg.seeds} random seeds; report mean &plusmn; std.
                </li>
              </ol>
            </div>
          </Card>
        </div>
      </div>

      <div className="section-label mt5">Evaluated Methods</div>
      <div className="method-grid">
        {r.arms.map((a) => (
          <div
            key={a.key}
            className="method-card clickable"
            onClick={() => goTab("Results")}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === "Enter" && goTab("Results")}
            title={`${a.name} — view in Results`}
          >
            <div className="mc-name">
              <span className="dot" style={{ background: familyColor(a.family) }} />
              {a.name}
            </div>
            <div className="mc-tag">{a.tag}</div>
          </div>
        ))}
      </div>
    </>
  );
}

function familyColor(f: string) {
  return f === "magnitude"
    ? C.green
    : f === "directional"
    ? C.amber
    : f === "control"
    ? C.ink2
    : C.gray;
}

function TaskChip({ i }: { i: number }) {
  const cells = [0, 1, 2, 3, 4, 5, 6, 7, 8];
  const on = [(i * 3) % 9, (i * 5 + 1) % 9, (i * 7 + 2) % 9, (i + 4) % 9];
  return (
    <div
      className="task-chip"
      title={`Task ${i} · fixed pixel permutation · per-task accuracy not recorded`}
    >
      <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
        {cells.map((c) => {
          const x = (c % 3) * 7 + 1;
          const y = Math.floor(c / 3) * 7 + 1;
          return (
            <rect
              key={c}
              x={x}
              y={y}
              width="6"
              height="6"
              rx="1"
              fill={on.includes(c) ? C.blue : "#e4e2db"}
            />
          );
        })}
      </svg>
      <span className="tlabel">Task {i}</span>
    </div>
  );
}

/* ------------------------------ Results --------------------------------- */
function Results({
  r,
  selected,
  setSelected,
  goTab,
  seedN,
}: {
  r: R20Type;
  selected: Set<string>;
  setSelected: (s: Set<string>) => void;
  goTab: (t: string) => void;
  seedN: number;
}) {
  const arms = r.arms;
  const toggle = (id: string) => {
    const next = new Set(selected);
    next.has(id) ? next.delete(id) : next.add(id);
    setSelected(next);
  };

  // Derive per-metric winners from source data
  const maxAcc = Math.max(...arms.map((a) => a.acc));
  const minDecay = Math.min(...arms.map((a) => a.decay));
  const minNorm = Math.min(...arms.map((a) => a.norm));
  const minDormant = Math.min(...arms.map((a) => a.dormant));

  const cols: Column<Arm>[] = [
    {
      key: "name",
      header: "Method",
      sortable: true,
      value: (a) => a.name,
      render: (a) => (
        <span style={{ display: "inline-flex", alignItems: "center", gap: 7 }}>
          <span className="dot" style={{ background: familyColor(a.family) }} />
          {a.name}
        </span>
      ),
    },
    {
      key: "acc",
      header: "Accuracy ↑",
      numeric: true,
      sortable: true,
      value: (a) => a.acc,
      render: (a) => {
        const winner = Math.abs(a.acc - maxAcc) < 1e-9;
        return (
          <>
            <span style={winner ? { color: "var(--blue)", fontWeight: 600 } : undefined}>
              {acc(a.acc)}
            </span>
            {" "}
            <span className="std">&plusmn;&thinsp;{acc(a.accStd)}</span>
          </>
        );
      },
    },
    {
      key: "decay",
      header: "Plasticity Decay ↓",
      numeric: true,
      sortable: true,
      value: (a) => a.decay,
      render: (a) => {
        const winner = Math.abs(a.decay - minDecay) < 1e-9;
        return (
          <>
            <span style={winner ? { color: "var(--blue)", fontWeight: 600 } : undefined}>
              {acc(a.decay)}
            </span>
            {" "}
            <span className="std">&plusmn;&thinsp;{acc(a.decayStd)}</span>
          </>
        );
      },
    },
    {
      key: "norm",
      header: "Final Norm ↓",
      numeric: true,
      sortable: true,
      value: (a) => a.norm,
      render: (a) => {
        const winner = Math.abs(a.norm - minNorm) < 1e-9;
        return (
          <span style={winner ? { color: "var(--blue)", fontWeight: 600 } : undefined}>
            {num(a.norm)}
          </span>
        );
      },
    },
    {
      key: "dormant",
      header: "Dormant ↓",
      numeric: true,
      sortable: true,
      value: (a) => a.dormant,
      render: (a) => {
        const winner = Math.abs(a.dormant - minDormant) < 1e-9;
        return (
          <span style={winner ? { color: "var(--blue)", fontWeight: 600 } : undefined}>
            {pct(a.dormant)}
          </span>
        );
      },
    },
    {
      key: "match",
      header: "Match Dev.",
      numeric: true,
      sortable: false,
      render: (a) => <MatchDevCell arm={a} />,
    },
  ];

  // Chart data sorted by accuracy (desc) for visual consistency
  const chartData = [...arms].sort((a, b) => b.acc - a.acc);

  return (
    <>
      <Card className="mb4">
        <CardHead
          title="Method Comparison"
          sub={`${arms.length} arms · ${seedN} seeds · mean ± std`}
          right={
            selected.size >= 2 ? (
              <span className="badge blue">{selected.size} selected</span>
            ) : (
              <span className="small muted">Select rows to compare</span>
            )
          }
        />
        <ResultTable
          columns={cols}
          rows={arms}
          getRowId={(a) => a.key}
          onRowClick={() => goTab("Runs")}
          selectable
          selected={selected}
          onToggleSelect={toggle}
          initialSort={{ key: "acc", dir: "desc" }}
        />
        <div className="card-pad small muted" style={{ borderTop: "1px solid var(--border)" }}>
          <b>Match Dev.</b> = max per-step deviation from the plain-SGD step norm. Shown only
          for the fixed-step directional interventions; magnitude / reset methods set their own
          update magnitude, so it is <span className="na">N/A</span> (not zero) for them.
        </div>
      </Card>

      {selected.size >= 2 && (
        <CompareCard arms={arms.filter((a) => selected.has(a.key))} />
      )}

      {/* Three independent chart cards — each with its own card boundary */}
      <div className="grid-111">
        <Card pad>
          <div className="card-title mb2">
            Accuracy vs Method <span className="muted">&uarr;</span>
          </div>
          <BarSci
            data={chartData}
            xKey="short"
            yKey="acc"
            stdKey="accStd"
            color={C.blue}
            yDomain={[0, 1]}
            height={200}
            valueFmt={(v) => acc(v)}
            metricLabel="Mean acc"
            seedN={seedN}
          />
        </Card>
        <Card pad>
          <div className="card-title mb2">
            Plasticity Decay vs Method <span className="muted">&darr;</span>
          </div>
          <BarSci
            data={chartData}
            xKey="short"
            yKey="decay"
            stdKey="decayStd"
            color={C.red}
            yDomain={[0, 0.24]}
            height={200}
            valueFmt={(v) => acc(v)}
            metricLabel="Decay"
            seedN={seedN}
          />
        </Card>
        <Card pad>
          <div className="card-title mb2">
            Final Norm vs Method <span className="muted">&darr;</span>
          </div>
          <BarSci
            data={chartData}
            xKey="short"
            yKey="norm"
            stdKey="normStd"
            color={C.navy}
            yDomain={[0, 28]}
            height={200}
            valueFmt={(v) => num(v)}
            metricLabel="Final norm"
            seedN={seedN}
          />
        </Card>
      </div>
    </>
  );
}

function CompareCard({ arms }: { arms: Arm[] }) {
  const base = arms[0];
  const metric = (
    label: string,
    get: (a: Arm) => number,
    fmt: (v: number) => string,
    better: "hi" | "lo"
  ) => (
    <tr>
      <td>{label}</td>
      {arms.map((a) => {
        const delta = get(a) - get(base);
        const good = better === "hi" ? delta > 0 : delta < 0;
        return (
          <td key={a.key} className="num">
            {fmt(get(a))}
            {a.key !== base.key && Math.abs(delta) > 1e-9 && (
              <span
                className="std"
                style={{ color: good ? "var(--green)" : "var(--red)" }}
              >
                {" "}
                ({delta > 0 ? "+" : ""}
                {fmt(delta)})
              </span>
            )}
          </td>
        );
      })}
    </tr>
  );
  return (
    <Card className="mb4">
      <CardHead title="Compare selected" sub={`deltas vs ${base.name}`} />
      <div className="tbl-wrap">
        <table className="tbl">
          <thead>
            <tr>
              <th>Metric</th>
              {arms.map((a) => (
                <th key={a.key} className="num">
                  {a.short}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metric("Accuracy ↑", (a) => a.acc, (v) => acc(v), "hi")}
            {metric("Plasticity decay ↓", (a) => a.decay, (v) => acc(v), "lo")}
            {metric("Final norm ↓", (a) => a.norm, (v) => num(v), "lo")}
            {metric(
              "Dormant ↓",
              (a) => a.dormant,
              (v) => (v * 100).toFixed(1) + "%",
              "lo"
            )}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

/* ------------------------------- Runs ----------------------------------- */
function Runs({ r, nav }: { r: R20Type; nav: (to: string) => void }) {
  const [method, setMethod] = useState(r.arms[0].key);
  const rows = r.perSeed[method] ?? [];
  const arm = r.arms.find((a) => a.key === method);
  const armName = arm?.name ?? method;
  return (
    <Card>
      <CardHead
        title="Per-seed runs"
        right={
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <label
              htmlFor="runs-method-select"
              className="small muted"
              style={{ fontWeight: 500 }}
            >
              Method
            </label>
            <select
              id="runs-method-select"
              className="select"
              value={method}
              onChange={(e) => setMethod(e.target.value)}
              aria-label="Select method to view per-seed runs"
            >
              {r.arms.map((a) => (
                <option key={a.key} value={a.key}>
                  {a.name}
                </option>
              ))}
            </select>
          </div>
        }
      />
      <div className="tbl-wrap">
        <table className="tbl">
          <thead>
            <tr>
              <th>Seed</th>
              <th className="num">Mean Accuracy</th>
              <th className="num">Plasticity Decay</th>
              <th className="num">Final Norm</th>
              <th className="num">Dormant</th>
              <th className="num">Eff. Rank</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td colSpan={6} className="muted small" style={{ textAlign: "center", padding: "20px 12px" }}>
                  No per-seed records available for this method.
                </td>
              </tr>
            ) : (
              rows.map((row) => (
                <tr key={row.seed}>
                  <td className="mono">{row.seed}</td>
                  <td className="num">{acc(row.acc)}</td>
                  <td className="num">{acc(row.decay)}</td>
                  <td className="num">{num(row.norm)}</td>
                  <td className="num">{pct(row.dormant)}</td>
                  <td className="num">{num(row.effRank ?? null, 2)}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
      <div className="card-pad" style={{ borderTop: "1px solid var(--border)" }}>
        <button
          className="btn ghost sm"
          onClick={() => nav(`/runs?method=${method}`)}
          aria-label={`Open ${armName} seed runs in Run Details`}
        >
          Open {armName} in Run Details &rarr;
        </button>
      </div>
    </Card>
  );
}

/* ------------------------------ Config ---------------------------------- */
function ConfigTab({ r }: { r: R20Type }) {
  const c = r.config;
  return (
    <div style={{ maxWidth: 680 }}>
      <ConfigPanel
        title={r.source}
        lines={[
          ["dataset", "permuted-mnist"],
          ["model", "mlp"],
          ["hidden_width", c.hidden],
          ["depth", c.depth],
          ["activation", "relu"],
          ["lr", c.lr],
          ["steps_per_task", c.stepsPerTask],
          ["batch_size", c.batch],
          ["tasks", c.tasks],
          ["train_subset", c.trainSubset],
          ["epochs", c.epochs],
          ["classifier", "shared_10way"],
          ["seeds", `[${r.seeds.join(", ")}]`],
        ]}
      />
    </div>
  );
}

/* ------------------------------- Notes ---------------------------------- */
function Notes({ r }: { r: R20Type }) {
  return (
    <div className="grid" style={{ gap: 16, maxWidth: 760 }}>
      <Observation label="Matched-step control" tone="blue">
        For the directional interventions, every candidate update is renormalized to the exact
        Euclidean norm of the plain-SGD step at the same parameters, so only direction varies.
        Magnitude methods (L2-init, Shrink &amp; Perturb, ANR) are permitted to change the step
        norm by design and are therefore not matched-step.
      </Observation>
      <Observation label="How to read the comparison" tone="neutral">
        Norm-reducing magnitude methods preserve plasticity best here; the orthogonal directional
        arm cannot reduce the norm and decays most, while the inward arm drives the norm down but
        collapses accuracy. Weight-norm growth is treated as a regime-specific correlate of
        plasticity decay, not a proven cause.
      </Observation>
      {r.verdict && (
        <Observation label="Released verdict" tone="amber">
          The primary comparison is released with verdict{" "}
          <span className="mono">{r.verdict}</span>: the targeted per-unit method (ANR) does
          not beat global magnitude control on accuracy in this regime (0 /{" "}
          {r.seeds.length} seeds), reported as an honest negative.
        </Observation>
      )}
    </div>
  );
}
