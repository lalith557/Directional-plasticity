import { useState } from "react";
import { DataGate } from "../components/DataGate";
import { PageHeader } from "../components/PageHeader";
import { Tabs } from "../components/Tabs";
import { Card, CardHead, Observation, IntegrityBanner, Empty, Badge } from "../components/ui";
import { BarSci } from "../components/charts";
import { C } from "../lib/colors";
import { acc, num, pct, timestamp } from "../data/format";

const TABS = ["Results", "Table", "Runs"];
const SHORT: Record<string, string> = {
  anr_full: "ANR",
  anr_revive_only: "revive-only",
  anr_shrink_only: "shrink-only",
  sgd: "SGD",
  shrink_pert: "S&P",
};

export function Ablations() {
  const [tab, setTab] = useState("Results");
  return (
    <DataGate>
      {(d) => {
        const A = d.ablation;
        const chart = A.variants.map((v) => ({
          short: SHORT[v.key] ?? v.key,
          name: v.name,
          acc: v.acc,
          accStd: v.accStd,
          decay: v.decay,
          decayStd: v.decayStd,
          reference: v.reference,
        }));
        const barColorAcc = (dd: any) => (dd.reference ? C.gray : C.blue);
        const barColorDecay = (dd: any) => (dd.reference ? C.gray : C.red);
        const maxAcc = Math.max(...A.variants.map((v) => v.acc));
        const minDecay = Math.min(...A.variants.map((v) => v.decay));
        const minNorm = Math.min(...A.variants.map((v) => v.norm));
        const minDormant = Math.min(...A.variants.map((v) => v.dormant));
        return (
          <>
            <PageHeader
              title="ANR Analysis"
              subtitle="Investigating the source of Adaptive-Norm-Revive's underperformance via component ablation."
            />
            <Tabs tabs={TABS} active={tab} onChange={setTab} />

            {tab === "Results" && (
              <>
                <Card className="mb4">
                  <CardHead title="Method Comparison" sub={`mean ± std over ${A.seeds} seeds`} />
                  <div className="tbl-wrap">
                    <table className="tbl">
                      <thead>
                        <tr>
                          <th>Variant</th>
                          <th className="num">Accuracy ↑</th>
                          <th className="num">Plasticity Decay ↓</th>
                          <th className="num">Final Norm ↓</th>
                          <th className="num">Dormant ↓</th>
                        </tr>
                      </thead>
                      <tbody>
                        {A.variants.map((v) => (
                          <tr key={v.key}>
                            <td>
                              {v.name}{" "}
                              {v.reference ? (
                                <Badge tone="gray">reference</Badge>
                              ) : v.tag ? (
                                <span className="std">· {v.tag}</span>
                              ) : null}
                            </td>
                            <td className="num">
                              <span style={Math.abs(v.acc - maxAcc) < 1e-9 ? { color: "var(--blue)", fontWeight: 600 } : undefined}>{acc(v.acc)}</span>{" "}
                              <span className="std">± {acc(v.accStd)}</span>
                            </td>
                            <td className="num">
                              <span style={Math.abs(v.decay - minDecay) < 1e-9 ? { color: "var(--blue)", fontWeight: 600 } : undefined}>{acc(v.decay)}</span>{" "}
                              <span className="std">± {acc(v.decayStd)}</span>
                            </td>
                            <td className="num">
                              <span style={Math.abs(v.norm - minNorm) < 1e-9 ? { color: "var(--blue)", fontWeight: 600 } : undefined}>{num(v.norm, 1)}</span>
                            </td>
                            <td className="num">
                              <span style={Math.abs(v.dormant - minDormant) < 1e-9 ? { color: "var(--blue)", fontWeight: 600 } : undefined}>{pct(v.dormant)}</span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>

                <Card pad className="mb4">
                  <div className="grid-2col">
                    <div>
                      <div className="card-title mb2">Accuracy <span className="muted">↑</span></div>
                      <BarSci data={chart} xKey="short" yKey="acc" stdKey="accStd" colorFor={barColorAcc}
                        yDomain={[0, 0.9]} height={210} valueFmt={(v) => acc(v)} metricLabel="Accuracy" seedN={A.seeds} />
                    </div>
                    <div>
                      <div className="card-title mb2">Plasticity Decay <span className="muted">↓</span></div>
                      <BarSci data={chart} xKey="short" yKey="decay" stdKey="decayStd" colorFor={barColorDecay}
                        yDomain={[0, 0.2]} height={210} valueFmt={(v) => acc(v)} metricLabel="Decay" seedN={A.seeds} />
                    </div>
                  </div>
                  <div className="small muted">Grey bars are external references (SGD, Shrink &amp; Perturb).</div>
                </Card>

                <Observation label="Observation">{A.observation}</Observation>
              </>
            )}

            {tab === "Table" && (
              <Card>
                <CardHead title="ANR ablation — full metrics" sub={`mean ± std over ${A.seeds} seeds`} />
                <div className="tbl-wrap">
                  <table className="tbl">
                    <thead>
                      <tr>
                        <th>Variant</th>
                        <th className="num">Accuracy ↑</th>
                        <th className="num">Plasticity Decay ↓</th>
                        <th className="num">Final Norm ↓</th>
                        <th className="num">Dormant ↓</th>
                      </tr>
                    </thead>
                    <tbody>
                      {A.variants.map((v) => (
                        <tr key={v.key}>
                          <td>{v.name}</td>
                          <td className="num">{acc(v.acc)} <span className="std">± {acc(v.accStd)}</span></td>
                          <td className="num">{acc(v.decay)} <span className="std">± {acc(v.decayStd)}</span></td>
                          <td className="num">{num(v.norm, 2)} <span className="std">± {num(v.normStd, 2)}</span></td>
                          <td className="num">{pct(v.dormant)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            )}

            {tab === "Runs" && (
              <Empty title="Per-seed runs not individually released for this ablation">
                The ANR ablation summary stores aggregates (mean ± std over {A.seeds} seeds) only.
              </Empty>
            )}

            <IntegrityBanner source={A.source} seeds={A.seeds} loaded={timestamp(d.generatedAt)} />
          </>
        );
      }}
    </DataGate>
  );
}
