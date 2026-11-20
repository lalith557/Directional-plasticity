import { Card, CardHead, Badge } from "./ui";
import { ScatterSci } from "./charts";
import { corrColor, C } from "../lib/colors";
import { acc, num, pct, signed } from "../data/format";
import type { Regime } from "../data/types";

export function RegimeView({ regime }: { regime: Regime }) {
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
    <div className="grid" style={{ gap: 16 }}>
      <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", alignItems: "start" }}>
        <Card>
          <CardHead
            title="Method comparison"
            sub={`${regime.arch} · ${regime.seeds} seeds`}
            right={
              <Badge tone={regime.pearson && regime.pearson < 0 ? "red" : "blue"}>
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
          <div className="small muted" style={{ color: corrColor(regime.pearson) }}>
            Arm-level Pearson r = {signed(regime.pearson)} (n = {regime.arms}) — descriptive
            association, not causal.
          </div>
        </Card>
      </div>
    </div>
  );
}
