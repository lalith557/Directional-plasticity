import { useState } from "react";
import { DataGate } from "../components/DataGate";
import { PageHeader } from "../components/PageHeader";
import { Tabs } from "../components/Tabs";
import { Card, CardHead, Observation, IntegrityBanner, StatLine, Badge } from "../components/ui";
import { BarSci, ScatterSci } from "../components/charts";
import { logTrend } from "../lib/regression";
import { corrColor, REGIME_COLOR, C } from "../lib/colors";
import { acc, num, signed, timestamp } from "../data/format";

const TABS = ["Scatter", "Across Regimes", "Statistics", "Per Arm"];

export function NormPlasticity() {
  const [tab, setTab] = useState("Scatter");
  const [on, setOn] = useState<Record<string, boolean>>({
    width16: true,
    width64: true,
    width128: true,
    rot_mlp: true,
    rot_cnn: true,
  });

  return (
    <DataGate>
      {(d) => {
        const P = d.normPlasticity.primary;
        const primaryData = P.points.map((p) => ({
          x: p.norm,
          y: p.decay,
          acc: p.acc,
          name: p.name,
        }));
        const trend = logTrend(P.points.map((p) => ({ norm: p.norm, decay: p.decay })));

        return (
          <>
            <PageHeader
              title="Norm vs Plasticity Analysis"
              subtitle="Relationship between final weight norm and plasticity decay across arms and regimes."
            />
            <Tabs tabs={TABS} active={tab} onChange={setTab} />

            {tab === "Scatter" && (
              <>
                <Card pad className="mb4">
                  <div className="row between aic mb2">
                    <div className="card-title">R20 (Primary Regime) — width 16</div>
                    <Badge tone="blue">Pearson r = {signed(P.pearson)}</Badge>
                  </div>
                  <ScatterSci
                    series={[{ name: "R20 arms", color: C.blue, data: primaryData, trend }]}
                    xLog
                    legend={false}
                    xLabel="final norm (log scale)"
                    yLabel="plasticity decay"
                    xFmt={(v) => (v >= 1 ? String(Math.round(v)) : v.toFixed(1))}
                    height={280}
                    tooltip={(p) => ({
                      title: p.name,
                      rows: [
                        { label: "Final norm", value: num(p.x, 2) },
                        { label: "Decay", value: acc(p.y) },
                        { label: "Accuracy", value: acc(p.acc) },
                      ],
                    })}
                  />
                </Card>
                <Observation label="Note" tone="neutral">
                  Correlation is computed at the arm level (n&nbsp;=&nbsp;{P.n}) and reflects an
                  association, not causation. The dashed line is a least-squares trend in log-norm
                  space, shown only as a visual aid.
                </Observation>
              </>
            )}

            {tab === "Across Regimes" && (
              <>
                <div className="grid-2col mb4">
                  <Card pad>
                    <div className="card-title mb2">Correlation (Norm vs Plasticity Decay) by regime</div>
                    <BarSci
                      data={d.regimes.map((r) => ({ short: r.short, r: r.pearson, name: r.name }))}
                      xKey="short"
                      yKey="r"
                      colorFor={(dd) => corrColor(dd.r)}
                      yDomain={[-1, 1]}
                      yTickFmt={(v) => signed(v, 1)}
                      height={230}
                      valueFmt={(v) => signed(v)}
                      metricLabel="Pearson r"
                      reference0
                    />
                  </Card>
                  <Card pad>
                    <div className="row between aic mb2">
                      <div className="card-title">Norm vs Decay (by regime)</div>
                    </div>
                    <div className="chips mb3">
                      {d.regimes.map((r) => (
                        <button
                          key={r.id}
                          className={`chip ${on[r.id] ? "on" : ""}`}
                          onClick={() => setOn((s) => ({ ...s, [r.id]: !s[r.id] }))}
                        >
                          <span className="dot" style={{ background: REGIME_COLOR[r.id] }} />
                          {r.short}
                        </button>
                      ))}
                    </div>
                    <ScatterSci
                      series={d.regimes
                        .filter((r) => on[r.id])
                        .map((r) => ({
                          name: r.short,
                          color: REGIME_COLOR[r.id],
                          data: r.points
                            .filter((p) => p.norm != null && p.decay != null)
                            .map((p) => ({ x: p.norm, y: p.decay, acc: p.acc, name: p.name, regime: r.short })),
                        }))}
                      xLog
                      xLabel="final norm (log scale)"
                      yLabel="plasticity decay"
                      xFmt={(v) => (v >= 1 ? String(Math.round(v)) : v.toFixed(1))}
                      height={230}
                      tooltip={(p) => ({
                        title: `${p.name} · ${p.regime}`,
                        rows: [
                          { label: "Final norm", value: num(p.x, 2) },
                          { label: "Decay", value: acc(p.y) },
                          { label: "Accuracy", value: acc(p.acc) },
                        ],
                      })}
                    />
                  </Card>
                </div>

                <Card className="mb4">
                  <CardHead title="Regime dependence" sub="descriptive arm-level association, not a seed-level causal estimate" />
                  <div className="tbl-wrap">
                    <table className="tbl">
                      <thead>
                        <tr>
                          <th>Regime</th>
                          <th>Dataset</th>
                          <th className="num">Seeds</th>
                          <th className="num">Arms</th>
                          <th className="num">Pearson r</th>
                        </tr>
                      </thead>
                      <tbody>
                        {d.regimes.map((r) => (
                          <tr key={r.id}>
                            <td>
                              <span className="dot" style={{ background: REGIME_COLOR[r.id], marginRight: 7 }} />
                              {r.name}
                            </td>
                            <td className="muted">{r.dataset}</td>
                            <td className="num">{r.seeds}</td>
                            <td className="num">{r.arms}</td>
                            <td className="num" style={{ color: corrColor(r.pearson), fontWeight: 600 }}>
                              {signed(r.pearson)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>

                <Observation label="Observation">
                  The association is strong and positive in the primary regime (width 16,{" "}
                  {signed(d.regimes[0].pearson)}) but weakens and changes sign in wider and
                  rotated-task regimes ({signed(d.regimes[2].pearson)} at width 128;{" "}
                  {signed(d.regimes[3].pearson)} rotated MLP), indicating regime dependence rather
                  than a universal mechanism.
                </Observation>
              </>
            )}

            {tab === "Statistics" && (
              <div className="grid" style={{ gap: 16, maxWidth: 720 }}>
                <Card pad>
                  <div className="card-title mb3">Primary regime (width 16, arm-level)</div>
                  <StatLine
                    stats={[
                      { label: "Pearson r (norm, decay)", value: signed(P.pearson) },
                      { label: "Spearman ρ", value: acc(P.spearman, 2) },
                      { label: "n (arms)", value: P.n },
                    ]}
                  />
                </Card>
                <Card pad>
                  <div className="card-title mb3">Secondary diagnostics</div>
                  <StatLine
                    stats={[
                      { label: "Eff. rank ↔ accuracy (r)", value: signed(d.normPlasticity.effRankAcc) },
                      { label: "Eff. rank ↔ decay (r)", value: signed(d.normPlasticity.effRankDecay) },
                      { label: "Norm ↔ accuracy (r)", value: signed(P.accPearson) },
                    ]}
                  />
                  <div className="small muted mt4">
                    Effective feature rank tracks accuracy but only weakly (and negatively) with decay,
                    so decay is not attributed to rank collapse.
                  </div>
                </Card>
                <Observation label="Interpretation" tone="neutral">
                  All correlations here are arm-level descriptive statistics over method means, not
                  seed-level causal estimates. The λ-sweep supplies within-family interventional
                  evidence; neither is presented as proof that norm growth causes plasticity loss.
                </Observation>
              </div>
            )}

            {tab === "Per Arm" && (
              <Card>
                <CardHead title="Per-arm norm & decay (primary regime)" sub={`n = ${P.n} arms`} />
                <div className="tbl-wrap">
                  <table className="tbl">
                    <thead>
                      <tr>
                        <th>Method</th>
                        <th className="num">Final Norm</th>
                        <th className="num">Plasticity Decay</th>
                        <th className="num">Accuracy</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...P.points]
                        .sort((a, b) => b.norm - a.norm)
                        .map((p) => (
                          <tr key={p.key}>
                            <td>{p.name}</td>
                            <td className="num">{num(p.norm, 2)}</td>
                            <td className="num">{acc(p.decay)}</td>
                            <td className="num">{acc(p.acc)}</td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            )}

            <IntegrityBanner source={d.normPlasticity.source} loaded={timestamp(d.generatedAt)} />
          </>
        );
      }}
    </DataGate>
  );
}
