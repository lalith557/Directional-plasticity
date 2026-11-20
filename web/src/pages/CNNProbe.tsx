import { DataGate } from "../components/DataGate";
import { PageHeader } from "../components/PageHeader";
import { Card, CardHead, Badge, Observation, IntegrityBanner } from "../components/ui";
import { ScatterSci, LineSci } from "../components/charts";
import { corrColor, C } from "../lib/colors";
import { acc, num, pct, signed, timestamp } from "../data/format";

export function CNNProbe() {
  return (
    <DataGate>
      {(d) => {
        const regime = d.regimes.find((r) => r.id === "rot_cnn")!;
        const A = d.angle;

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
        const ap = A.points.map((p) => ({ ...p, clabel: p.cos.toFixed(2) }));

        return (
          <>
            <PageHeader
              title="CNN Probe"
              subtitle="Small convolutional network on rotated MNIST — regime dependence and an update-alignment sweep."
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
                          <td className="num">{p.norm != null ? num(p.norm, 2) : <span className="na">N/A</span>}</td>
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

            <div className="section-label mt5">Update-alignment sweep</div>
            <Card pad className="mb4">
              <div className="card-title mb2">
                Accuracy &amp; Final Norm vs gradient alignment (cos)
              </div>
              <LineSci
                data={ap}
                xKey="clabel"
                xLabel="cosine alignment with the gradient (1 = SGD)"
                lines={[
                  { key: "acc", name: "Accuracy", color: C.blue, axis: "left", stdKey: "accStd" },
                  { key: "norm", name: "Final norm", color: C.red, axis: "right" },
                ]}
                leftDomain={[0, 1]}
                leftFmt={(v) => v.toFixed(1)}
                rightLog
                rightFmt={(v) => (v >= 1 ? String(Math.round(v)) : v.toFixed(1))}
                legend
                height={240}
                tooltip={(p: any) => [
                  { label: "Cos", value: num(p.cos, 2) },
                  { label: "Accuracy", value: `${acc(p.acc)} ± ${acc(p.accStd)}` },
                  { label: "Final norm", value: num(p.norm, 1) },
                ]}
              />
            </Card>

            <Observation label="Observation" tone="amber">
              As the update is steered away from the gradient (cosine alignment 1.00 → 0.10), the
              final weight norm inflates sharply ({num(A.points[0].norm, 0)} →{" "}
              {num(A.points[A.points.length - 1].norm, 0)}) and accuracy falls (
              {acc(A.points[0].acc)} → {acc(A.points[A.points.length - 1].acc)}). Yet across the
              six arms the norm–decay association on this CNN is negative ({signed(regime.pearson)}),
              underlining that the norm–plasticity coupling is regime-specific.
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
