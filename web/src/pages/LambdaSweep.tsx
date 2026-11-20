import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { DataGate } from "../components/DataGate";
import { PageHeader } from "../components/PageHeader";
import { Tabs } from "../components/Tabs";
import { Card, CardHead, Observation, IntegrityBanner } from "../components/ui";
import { ConfigPanel } from "../components/ConfigPanel";
import { LineSci } from "../components/charts";
import { C } from "../lib/colors";
import { acc, num, lamLabel, timestamp } from "../data/format";

const TABS = ["Results", "Table", "Runs", "Config"];

export function LambdaSweep() {
  const [params, setParams] = useSearchParams();
  const tabParam = params.get("tab");
  const initial = TABS.find((t) => t.toLowerCase() === (tabParam ?? "").toLowerCase()) ?? "Results";
  const [tab, setTab] = useState(initial);

  const setTabAndUrl = (t: string) => {
    setTab(t);
    setParams(t === "Results" ? {} : { tab: t.toLowerCase() }, { replace: true });
  };

  return (
    <DataGate>
      {(d) => {
        const L = d.lambda_;
        const data = L.points.map((p) => ({ ...p, xlabel: lamLabel(p.lam) }));

        const maxAcc = Math.max(...L.points.map((p) => p.acc));
        const minDecay = Math.min(...L.points.map((p) => p.decay));
        const minNorm = Math.min(...L.points.map((p) => p.norm));
        const minDormant = Math.min(...L.points.map((p) => p.dormant));

        return (
          <>
            <PageHeader
              title="λ Sweep — Controlled Norm Reduction"
              subtitle="Effect of the weight-decay coefficient (L2-init) on performance and final parameter norm."
            />
            <Tabs tabs={TABS} active={tab} onChange={setTabAndUrl} />

            {tab === "Results" && (
              <>
                <div className="grid-111 mb4">
                  <Card pad>
                    <div className="card-title mb2">
                      Accuracy vs λ <span className="muted">&uarr;</span>
                    </div>
                    <LineSci
                      data={data}
                      xKey="xlabel"
                      lines={[{ key: "acc", name: "Accuracy", color: C.blue, stdKey: "accStd" }]}
                      leftDomain={[0.6, 0.9]}
                      leftFmt={(v) => v.toFixed(1)}
                      xLabel="weight-decay λ"
                      height={200}
                      tooltip={(p) => [
                        { label: "Accuracy", value: `${acc(p.acc)} ± ${acc(p.accStd)}` },
                        { label: "Seeds", value: L.seeds },
                      ]}
                    />
                  </Card>

                  <Card pad>
                    <div className="card-title mb2">
                      Plasticity Decay vs λ <span className="muted">&darr;</span>
                    </div>
                    <LineSci
                      data={data}
                      xKey="xlabel"
                      lines={[{ key: "decay", name: "Decay", color: C.red, stdKey: "decayStd" }]}
                      leftDomain={[0, 0.18]}
                      leftFmt={(v) => v.toFixed(2)}
                      xLabel="weight-decay λ"
                      height={200}
                      tooltip={(p) => [
                        { label: "Decay", value: `${acc(p.decay)} ± ${acc(p.decayStd)}` },
                        { label: "Seeds", value: L.seeds },
                      ]}
                    />
                  </Card>

                  <Card pad>
                    <div className="card-title mb2">
                      Final Norm vs λ <span className="muted">&darr;</span>
                    </div>
                    <LineSci
                      data={data}
                      xKey="xlabel"
                      lines={[{ key: "norm", name: "Final norm", color: C.navy, stdKey: "normStd" }]}
                      leftDomain={[0, 24]}
                      leftFmt={(v) => v.toFixed(0)}
                      xLabel="weight-decay λ"
                      height={200}
                      tooltip={(p) => [
                        { label: "Final norm", value: `${num(p.norm)} ± ${num(p.normStd)}` },
                        { label: "Seeds", value: L.seeds },
                      ]}
                    />
                  </Card>
                </div>

                <Observation label="Observation" tone="blue">
                  Moderate regularization improves performance in this regime. Accuracy peaks at
                  λ&nbsp;=&nbsp;1e-2 among the evaluated settings ({acc(L.points[3].acc)}&nbsp;±&nbsp;
                  {acc(L.points[3].accStd)}). Stronger regularization at λ&nbsp;=&nbsp;1e-1 harms
                  accuracy ({acc(L.points[5].acc)}&nbsp;±&nbsp;{acc(L.points[5].accStd)}) despite
                  further reducing final parameter norm — indicating over-regularization disrupts learned
                  representations before norm reduction can benefit plasticity.
                </Observation>
              </>
            )}

            {tab === "Table" && (
              <Card>
                <CardHead
                  title="λ sweep — aggregate metrics"
                  sub={`mean ± std over ${L.seeds} seeds · ${L.grid ?? ""}`}
                />
                <div className="tbl-wrap">
                  <table className="tbl">
                    <thead>
                      <tr>
                        <th>λ</th>
                        <th className="num">Accuracy ↑</th>
                        <th className="num">Plasticity Decay ↓</th>
                        <th className="num">Final Norm ↓</th>
                        <th className="num">Dormant ↓</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.map((p) => {
                        const isAccWinner = Math.abs(p.acc - maxAcc) < 1e-9;
                        const isDecayWinner = Math.abs(p.decay - minDecay) < 1e-9;
                        const isNormWinner = Math.abs(p.norm - minNorm) < 1e-9;
                        const isDormantWinner = Math.abs(p.dormant - minDormant) < 1e-9;
                        return (
                          <tr key={p.lam}>
                            <td className="mono">{p.xlabel}</td>
                            <td
                              className="num"
                              style={isAccWinner ? { color: "var(--blue)", fontWeight: 600 } : undefined}
                            >
                              {acc(p.acc)}{" "}
                              <span className="std">± {acc(p.accStd)}</span>
                            </td>
                            <td
                              className="num"
                              style={isDecayWinner ? { color: "var(--green)", fontWeight: 600 } : undefined}
                            >
                              {acc(p.decay)}{" "}
                              <span className="std">± {acc(p.decayStd)}</span>
                            </td>
                            <td
                              className="num"
                              style={isNormWinner ? { color: "var(--green)", fontWeight: 600 } : undefined}
                            >
                              {num(p.norm)}{" "}
                              <span className="std">± {num(p.normStd)}</span>
                            </td>
                            <td
                              className="num"
                              style={isDormantWinner ? { color: "var(--green)", fontWeight: 600 } : undefined}
                            >
                              {(p.dormant * 100).toFixed(1)}%{" "}
                              <span className="std">± {(p.dormantStd * 100).toFixed(1)}%</span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </Card>
            )}

            {tab === "Runs" && (
              <Card pad>
                <div
                  style={{
                    textAlign: "center",
                    padding: "var(--s6) var(--s4)",
                    maxWidth: 520,
                    margin: "0 auto",
                  }}
                >
                  <div className="card-title" style={{ marginBottom: "var(--s2)", fontSize: 14 }}>
                    Per-seed runs not individually released for this sweep
                  </div>
                  <p
                    className="small muted"
                    style={{ margin: 0, lineHeight: 1.6 }}
                  >
                    The λ-sweep summary stores aggregates (mean ± std over {L.seeds} seeds) only.
                    Per-seed records are not available for this experiment. See the{" "}
                    <strong style={{ color: "var(--ink-2)" }}>Table</strong> tab for the released
                    aggregate values.
                  </p>
                </div>
              </Card>
            )}

            {tab === "Config" && (
              <div style={{ maxWidth: 680 }}>
                <ConfigPanel
                  title={L.source}
                  lines={[
                    ["method", "l2_init (regularize to init)"],
                    ["lambda_grid", `[${L.values.map(lamLabel).join(", ")}]`],
                    ["grid", L.grid ?? "predetermined geometric"],
                    ["seeds", L.seeds],
                    ["base_regime", "permuted-mnist · mlp width 16 · 100 tasks"],
                  ]}
                />
              </div>
            )}

            <IntegrityBanner source={L.source} seeds={L.seeds} loaded={timestamp(d.generatedAt)} />
          </>
        );
      }}
    </DataGate>
  );
}
