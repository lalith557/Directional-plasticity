import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { DataGate } from "../components/DataGate";
import { PageHeader } from "../components/PageHeader";
import { Tabs } from "../components/Tabs";
import { Card, CardHead, Observation, IntegrityBanner, StatLine } from "../components/ui";
import { LineSci } from "../components/charts";
import { GeometryDiagram } from "../components/GeometryDiagram";
import { SweepSlider } from "../components/SweepSlider";
import { ResultTable, Column } from "../components/ResultTable";
import { C } from "../lib/colors";
import { acc, num, pct, timestamp } from "../data/format";

const TABS = ["Results", "Table", "Geometry", "Runs"];
const gfmt = (v: number) => v.toFixed(1);

function SciVal({ v }: { v: number | null | undefined }) {
  if (v === null || v === undefined || Number.isNaN(v)) return <span className="na">N/A</span>;
  if (v === 0) return <span>0</span>;
  const s = v.toExponential(1);
  const [mantissa, exp] = s.split("e");
  return (
    <span className="mono">
      {mantissa}×10<sup>{exp.replace("+", "")}</sup>
    </span>
  );
}

type RadialRow = {
  gamma: number;
  acc: number; accStd: number;
  decay: number; decayStd: number;
  norm: number; normStd: number;
  dormant: number; dormantStd: number;
  radialAlign: number | null;
  matchDev: number | null;
};

function radialColumns(opts: {
  maxAcc: number; minDecay: number; minNorm: number; minDormant: number;
  selectedGamma: number;
}): Column<RadialRow>[] {
  const { maxAcc, minDecay, minNorm, minDormant, selectedGamma } = opts;
  const highlight = (hit: boolean, tone: string) =>
    hit ? { color: `var(--${tone})`, fontWeight: 600 } : undefined;
  return [
    {
      key: "gamma",
      header: "γ",
      sortable: true,
      value: (p) => p.gamma,
      render: (p) => (
        <span className={`mono ${p.gamma === selectedGamma ? "selected-cell" : ""}`}>
          {gfmt(p.gamma)}{p.gamma === 0 ? " (SGD)" : ""}
        </span>
      ),
    },
    {
      key: "acc", header: "Accuracy ↑", numeric: true, sortable: true,
      value: (p) => p.acc,
      render: (p) => (
        <>
          <span style={highlight(Math.abs(p.acc - maxAcc) < 1e-9, "blue")}>{acc(p.acc)}</span>{" "}
          <span className="std">± {acc(p.accStd)}</span>
        </>
      ),
    },
    {
      key: "decay", header: "Plasticity Decay ↓", numeric: true, sortable: true,
      value: (p) => p.decay,
      render: (p) => (
        <>
          <span style={highlight(Math.abs(p.decay - minDecay) < 1e-9, "green")}>{acc(p.decay)}</span>{" "}
          <span className="std">± {acc(p.decayStd)}</span>
        </>
      ),
    },
    {
      key: "norm", header: "Final Norm ↓", numeric: true, sortable: true,
      value: (p) => p.norm,
      render: (p) => (
        <>
          <span style={highlight(Math.abs(p.norm - minNorm) < 1e-9, "green")}>{num(p.norm, 2)}</span>{" "}
          <span className="std">± {num(p.normStd, 2)}</span>
        </>
      ),
    },
    {
      key: "dormant", header: "Dormant ↓", numeric: true, sortable: true,
      value: (p) => p.dormant,
      render: (p) => (
        <>
          <span style={highlight(Math.abs(p.dormant - minDormant) < 1e-9, "green")}>{pct(p.dormant)}</span>
          {p.dormantStd != null && <span className="std"> ± {pct(p.dormantStd)}</span>}
        </>
      ),
    },
    {
      key: "radialAlign", header: "Radial Align", numeric: true, sortable: true,
      value: (p) => p.radialAlign,
      render: (p) => <>{p.radialAlign == null ? <span className="na">N/A</span> : num(p.radialAlign, 3)}</>,
    },
    {
      key: "matchDev", header: "Match Dev.", numeric: true, sortable: true,
      value: (p) => p.matchDev,
      render: (p) => <SciVal v={p.matchDev} />,
    },
  ];
}

export function RadialSweep() {
  const [params, setParams] = useSearchParams();
  const tabParam = params.get("tab");
  const initial = TABS.find((t) => t.toLowerCase() === (tabParam ?? "").toLowerCase()) ?? "Results";
  const [tab, setTab] = useState(initial);
  const [gi, setGi] = useState(1); // default γ = 0.2 (accuracy optimum)

  const setTabAndUrl = (t: string) => {
    setTab(t);
    setParams(t === "Results" ? {} : { tab: t.toLowerCase() }, { replace: true });
  };

  return (
    <DataGate>
      {(d) => {
        const R = d.radial;
        const pts = R.points.map((p) => ({ ...p, glabel: gfmt(p.gamma) }));
        const cur = R.points[gi];

        const maxAcc = Math.max(...R.points.map((p) => p.acc));
        const minDecay = Math.min(...R.points.map((p) => p.decay));
        const minNorm = Math.min(...R.points.map((p) => p.norm));
        const minDormant = Math.min(...R.points.map((p) => p.dormant));

        const tip = (p: any) => [
          { label: "Accuracy", value: `${acc(p.acc)} ± ${acc(p.accStd)}` },
          { label: "Final norm", value: `${num(p.norm, 2)} ± ${num(p.normStd, 2)}` },
          { label: "Dormant", value: pct(p.dormant) },
          { label: "Seeds", value: R.seeds },
        ];

        const runs =
          R.perSeed[gfmt(cur.gamma)] ?? R.perSeed[String(cur.gamma)] ?? [];

        return (
          <>
            <PageHeader
              title="Radial Sweep — Update Direction Geometry"
              subtitle="Effect of radial (inward) vs tangential update allocation under matched step magnitude."
            />
            <Tabs tabs={TABS} active={tab} onChange={setTabAndUrl} />

            {tab === "Results" && (
              <>
                <div className="grid-side mb4">
                  <Card>
                    <CardHead title="Direction Decomposition" sub="schematic" />
                    <div className="card-pad" style={{ display: "grid", placeItems: "center" }}>
                      <GeometryDiagram gamma={cur.gamma} />
                    </div>
                  </Card>
                  <Card pad>
                    <div className="card-title mb2">
                      Accuracy &amp; Final Norm vs Radial allocation (γ)
                    </div>
                    <LineSci
                      data={pts}
                      xKey="glabel"
                      xLabel="radial allocation γ"
                      highlightX={gfmt(cur.gamma)}
                      lines={[
                        { key: "acc", name: "Accuracy", color: C.blue, axis: "left", stdKey: "accStd" },
                        { key: "norm", name: "Final norm", color: C.red, axis: "right" },
                      ]}
                      leftDomain={[0, 0.85]}
                      leftFmt={(v) => v.toFixed(2)}
                      rightFmt={(v) => v.toFixed(0)}
                      legend
                      height={240}
                      tooltip={tip}
                    />
                  </Card>
                </div>

                <Card pad className="mb4">
                  <div className="grid-side-right">
                    <SweepSlider
                      values={R.gammas}
                      index={gi}
                      onChange={setGi}
                      label="Radial coefficient γ"
                      format={gfmt}
                    />
                    <StatLine
                      stats={[
                        { label: "Accuracy", value: acc(cur.acc) },
                        { label: "Final norm", value: num(cur.norm, 2) },
                        { label: "Dormancy", value: pct(cur.dormant) },
                        { label: "Seeds", value: R.seeds },
                      ]}
                    />
                  </div>
                  <div className="small muted mt4">
                    γ = {gfmt(cur.gamma)} · accuracy {acc(cur.acc)} ± {acc(cur.accStd)} ·{" "}
                    matched-step deviation <SciVal v={cur.matchDev} />
                    {cur.gamma === 0 && " · γ = 0 reproduces plain SGD"}
                  </div>
                </Card>

                <Observation label="Key Observation">{R.observation}</Observation>
              </>
            )}

            {tab === "Table" && (
              <Card>
                <CardHead
                  title="Radial sweep — aggregate metrics"
                  sub={`mean ± std over ${R.seeds} seeds · matched step · click a column header to sort`}
                />
                <ResultTable
                  rows={R.points}
                  getRowId={(p) => String(p.gamma)}
                  initialSort={{ key: "gamma", dir: "asc" }}
                  columns={radialColumns({ maxAcc, minDecay, minNorm, minDormant, selectedGamma: cur.gamma })}
                />
              </Card>
            )}

            {tab === "Geometry" && (
              <div className="grid-side">
                <Card>
                  <CardHead
                    title="Direction Decomposition"
                    sub={`γ = ${gfmt(cur.gamma)}`}
                  />
                  <div className="card-pad" style={{ display: "grid", placeItems: "center" }}>
                    <GeometryDiagram gamma={cur.gamma} />
                    <div style={{ width: "100%", marginTop: 16 }}>
                      <SweepSlider
                        values={R.gammas}
                        index={gi}
                        onChange={setGi}
                        label="γ"
                        format={gfmt}
                      />
                    </div>
                  </div>
                </Card>
                <Observation label="Radial–tangential trade-off" tone="neutral">
                  A matched-length update of unit direction <span className="mono">d</span> splits
                  into a radial part <span className="mono">d∥</span> (along ±θ) and a tangential
                  part <span className="mono">d⊥</span>. Only the radial part changes the global
                  norm; the tangential part carries task descent. Allocating more of the fixed budget
                  inward (larger γ) shrinks the norm but starves descent — the norm falls
                  monotonically ({num(R.points[0].norm, 1)} →{" "}
                  {num(R.points[R.points.length - 1].norm, 1)}) while accuracy peaks at a mild bias
                  (γ = 0.2, {acc(R.points[1].acc)}) and then collapses.
                </Observation>
              </div>
            )}

            {tab === "Runs" && (
              <Card>
                <CardHead
                  title="Per-seed runs"
                  right={
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <label
                        htmlFor="runs-gamma-select"
                        className="small muted"
                        style={{ fontWeight: 500 }}
                      >
                        γ
                      </label>
                      <select
                        id="runs-gamma-select"
                        className="select"
                        value={gi}
                        onChange={(e) => setGi(Number(e.target.value))}
                        aria-label="Select radial coefficient γ to view per-seed runs"
                      >
                        {R.gammas.map((g, i) => (
                          <option key={g} value={i}>
                            γ = {gfmt(g)}
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
                      </tr>
                    </thead>
                    <tbody>
                      {runs.length === 0 ? (
                        <tr>
                          <td
                            colSpan={5}
                            className="muted small"
                            style={{ textAlign: "center", padding: "20px 12px" }}
                          >
                            No per-seed records available for γ = {gfmt(cur.gamma)}.
                          </td>
                        </tr>
                      ) : (
                        runs.map((row) => (
                          <tr key={row.seed}>
                            <td className="mono">{row.seed}</td>
                            <td className="num">{acc(row.acc)}</td>
                            <td className="num">{acc(row.decay)}</td>
                            <td className="num">{num(row.norm, 2)}</td>
                            <td className="num">{pct(row.dormant)}</td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </Card>
            )}

            <IntegrityBanner
              source={R.source}
              seeds={R.seeds}
              loaded={timestamp(d.generatedAt)}
              extra="γ=0 reproduces SGD; γ=0.6 the inward arm of R20"
            />
          </>
        );
      }}
    </DataGate>
  );
}
