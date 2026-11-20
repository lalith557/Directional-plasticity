import { DataGate } from "../components/DataGate";
import { PageHeader } from "../components/PageHeader";
import { Card, CardHead, Badge, Observation, IntegrityBanner } from "../components/ui";
import { ScatterSci } from "../components/charts";
import { corrColor, C } from "../lib/colors";
import { acc, num, pct, signed, timestamp } from "../data/format";

export function RotatedMNIST() {
  return (
    <DataGate>
      {(d) => {
        const regime = d.regimes.find((r) => r.id === "rot_mlp")!;
        const pts = [...regime.points].sort((a, b) => (b.acc ?? -1) - (a.acc ?? -1));
        const scatter = [
          {
            name: regime.short,
            color: C.blue,
            data: regime.points
              .filter((p) => p.norm != null && p.decay != null)
              .map((p) => ({ x: p.norm, y: p.decay, acc: p.acc, name: p.name })),
          },
        ];

        return (
          <>
            <PageHeader
              title="Rotated MNIST"
              subtitle="Norm–plasticity relationship on a rotated-task geometry (MLP width 16)."
            />

            <div className="grid-2col mb4">
              <Card>
                <CardHead
                  title="Method comparison"
                  sub={`${regime.arch} · ${regime.seeds} seeds · ${regime.arms} arms`}
                  right={
                    <Badge tone={regime.pearson !== null && regime.pearson < 0 ? "red" : "blue"}>
                      r = {signed(regime.pearson)}
                    </Badge>
                  }
                />
                <div className="tbl-wrap">
                  <table className="tbl">
                    <thead>
                      <tr>
                        <th>Method</th>
                        <th className="num">Accuracy ↑</th>
                        <th className="num">Decay ↓</th>
                        <th className="num">Final Norm</th>
                        <th className="num">Dormant</th>
                      </tr>
                    </thead>
                    <tbody>
                      {pts.map((p) => (
                        <tr key={p.key}>
                          <td>{p.name}</td>
                          <td className="num">{acc(p.acc)}</td>
                          <td className="num">{acc(p.decay)}</td>
                          <td className="num">{num(p.norm, 2)}</td>
                          <td className="num">{pct(p.dormant)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>

              <Card pad>
                <div className="card-title mb2">Final Norm vs Plasticity Decay</div>
                <ScatterSci
                  series={scatter}
                  xLog
                  legend={false}
                  xLabel="final norm (log scale)"
                  yLabel="plasticity decay"
                  xFmt={(v) => (v >= 1 ? String(Math.round(v)) : v.toFixed(1))}
                  height={230}
                  tooltip={(p) => ({
                    title: p.name,
                    rows: [
                      { label: "Final norm", value: num(p.x, 2) },
                      { label: "Decay", value: acc(p.y) },
                      { label: "Accuracy", value: acc(p.acc) },
                    ],
                  })}
                />
                <div className="small muted mt2" style={{ color: corrColor(regime.pearson) }}>
                  Arm-level Pearson r = {signed(regime.pearson)} (n = {regime.arms}) — descriptive
                  association, not causal.
                </div>
              </Card>
            </div>

            <Observation label="Observation" tone="amber">
              On rotated MNIST plain SGD already retains plasticity (low decay), and the highest-norm
              arms (SGD, CBP) are among the best performers, so the arm-level norm–decay association
              is strongly negative ({signed(regime.pearson)}) — the opposite sign to the width-16
              stress regime. Weight-norm growth is therefore a regime-specific correlate, not a
              universal mechanism.
            </Observation>

            <IntegrityBanner
              source={regime.source}
              seeds={regime.seeds}
              extra={`arm-level Pearson r (n = ${regime.arms}), not a causal estimate`}
              loaded={timestamp(d.generatedAt)}
            />
          </>
        );
      }}
    </DataGate>
  );
}
