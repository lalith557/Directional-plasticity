import { DataGate } from "../components/DataGate";
import { PageHeader } from "../components/PageHeader";
import { Card, CardHead, Observation, IntegrityBanner } from "../components/ui";
import { BarSci, ScatterSci } from "../components/charts";
import { corrColor, REGIME_COLOR } from "../lib/colors";
import { signed, acc, num, timestamp } from "../data/format";

const WIDTHS = ["width16", "width64", "width128"];

export function WidthStudy() {
  return (
    <DataGate>
      {(d) => {
        const regimes = WIDTHS.map((id) => d.regimes.find((r) => r.id === id)!).filter(Boolean);
        const corr = regimes.map((r) => ({ short: r.short, r: r.pearson, name: r.name }));
        const series = regimes.map((r) => ({
          name: r.short,
          color: REGIME_COLOR[r.id],
          data: r.points
            .filter((p) => p.norm != null && p.decay != null)
            .map((p) => ({ x: p.norm, y: p.decay, acc: p.acc, name: p.name, regime: r.short })),
        }));

        return (
          <>
            <PageHeader
              title="Width Study"
              subtitle="How the norm–plasticity association changes with MLP hidden width on Permuted MNIST."
            />

            <div className="grid-2col mb4">
              <Card pad>
                <div className="card-title mb2">
                  Arm-level correlation (Norm ↔ Decay) by width
                </div>
                <BarSci
                  data={corr}
                  xKey="short"
                  yKey="r"
                  colorFor={(dd) => corrColor(dd.r)}
                  yDomain={[-1, 1]}
                  yTickFmt={(v) => signed(v, 1)}
                  height={210}
                  valueFmt={(v) => signed(v)}
                  metricLabel="Pearson r"
                  reference0
                />
                <div className="small muted mt4">
                  Positive (blue) at width&nbsp;16, attenuating at width&nbsp;64, reversing
                  (red) at width&nbsp;128.
                </div>
              </Card>

              <Card pad>
                <div className="card-title mb2">Final Norm vs Plasticity Decay</div>
                <ScatterSci
                  series={series}
                  xLog
                  xLabel="final norm (log scale)"
                  yLabel="plasticity decay"
                  xFmt={(v) => (v >= 1 ? String(Math.round(v)) : v.toFixed(1))}
                  height={220}
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
              <CardHead
                title="Regime summary"
                sub="n = 7 method arms · arm-level association, not seed-level"
              />
              <div className="tbl-wrap">
                <table className="tbl">
                  <thead>
                    <tr>
                      <th>Width</th>
                      <th>Architecture</th>
                      <th className="num">Seeds</th>
                      <th className="num">Arms</th>
                      <th className="num">SGD decay</th>
                      <th className="num">Pearson r</th>
                    </tr>
                  </thead>
                  <tbody>
                    {regimes.map((r) => {
                      const sgd = r.points.find((p) => p.key === "sgd");
                      return (
                        <tr key={r.id}>
                          <td>{r.short}</td>
                          <td className="mono">{r.arch}</td>
                          <td className="num">{r.seeds}</td>
                          <td className="num">{r.arms}</td>
                          <td className="num">{acc(sgd?.decay ?? null)}</td>
                          <td
                            className="num"
                            style={{ color: corrColor(r.pearson), fontWeight: 600 }}
                          >
                            {signed(r.pearson)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Card>

            <Observation label="Observation" tone="amber">
              As hidden width increases the capacity-stress abates: SGD's plasticity decay
              shrinks and the arm-level norm–decay association weakens (
              {signed(regimes[0].pearson)} → {signed(regimes[1].pearson)}) and reverses sign
              at width&nbsp;128 ({signed(regimes[2].pearson)}). The positive coupling is a
              property of the capacity-constrained regime, not of continual learning in general.
            </Observation>

            <IntegrityBanner
              source="e20 · e22_arch64 · e24_arch128 summaries"
              extra="arm-level association (n = 7), not a causal estimate"
              loaded={timestamp(d.generatedAt)}
            />
          </>
        );
      }}
    </DataGate>
  );
}
