import { useState } from "react";
import { DataGate } from "../components/DataGate";
import { PageHeader } from "../components/PageHeader";
import { Card, CardHead, Observation, IntegrityBanner } from "../components/ui";
import { acc, num, pct, timestamp } from "../data/format";

type MetricKey = "acc" | "decay" | "norm" | "dormant";

const METRICS: {
  key: MetricKey;
  label: string;
  better: "hi" | "lo";
  fmt: (v: number) => string;
  cardSub: string;
}[] = [
  {
    key: "acc",
    label: "Accuracy",
    better: "hi",
    fmt: (v) => acc(v),
    cardSub: "higher is better · best per regime in blue",
  },
  {
    key: "decay",
    label: "Plasticity Decay",
    better: "lo",
    fmt: (v) => acc(v),
    cardSub: "lower is better · best per regime in blue",
  },
  {
    key: "norm",
    label: "Final Norm",
    better: "lo",
    fmt: (v) => num(v, 2),
    cardSub: "descriptive — smallest per regime highlighted · smaller norm is not universally better",
  },
  {
    key: "dormant",
    label: "Dormant Units",
    better: "lo",
    fmt: (v) => pct(v),
    cardSub: "lower indicates fewer dormant units · best per regime in blue",
  },
];

function MetricObservation({ metric }: { metric: MetricKey }) {
  if (metric === "acc")
    return (
      <Observation label="Reading" tone="neutral">
        The inward directional arm has substantially lower accuracy than the other evaluated methods
        across the displayed regimes. L2-init and Shrink &amp; Perturb perform strongly in the
        width-16 stress regime, but no single method leads every regime. Missing entries indicate
        that a method–regime combination is unavailable, not a zero score.
      </Observation>
    );
  if (metric === "decay")
    return (
      <Observation label="Reading" tone="neutral">
        In the width-16 stress regime, magnitude methods (L2-init, Shrink &amp; Perturb) achieve
        the lowest plasticity decay; SGD and Directional show higher decay. Across wider widths the
        spread narrows, and in the rotated-task regimes SGD and ANR reach very low decay. The inward
        arm shows low decay in some regimes despite collapsed accuracy — low decay alone does not
        indicate good learning. Rankings are descriptive and regime-specific, not causal.
      </Observation>
    );
  if (metric === "norm")
    return (
      <Observation label="Reading" tone="neutral">
        Norms vary substantially across methods and regimes. The inward arm consistently has the
        smallest final norm; SGD and Directional are among the largest. The norm–plasticity
        association reverses sign across regimes (positive at width&nbsp;16, negative at
        width&nbsp;128 and in the rotated-task settings), so a smaller final norm is not universally
        associated with better learning outcomes. Norm values here are descriptive diagnostics, not
        performance measures.
      </Observation>
    );
  return (
    <Observation label="Reading" tone="neutral">
      Shrink &amp; Perturb and ANR maintain low dormancy across most MLP regimes; ANR's revival
      mechanism specifically targets dormant units. SGD, CBP, and Directional accumulate high
      dormancy in the wider-width permuted-MNIST settings. The inward arm reaches very high dormancy
      in the CNN regime alongside collapsed accuracy — low dormancy is a useful diagnostic but does
      not guarantee accuracy or retained plasticity.
    </Observation>
  );
}

export function Comparative() {
  const [metric, setMetric] = useState<MetricKey>("acc");
  const m = METRICS.find((x) => x.key === metric)!;

  return (
    <DataGate>
      {(d) => {
        // union of methods across regimes, keyed by method key; first display name wins
        const methodOrder: string[] = [];
        const nameByKey: Record<string, string> = {};
        d.regimes.forEach((r) =>
          r.points.forEach((p) => {
            if (!(p.key in nameByKey)) {
              nameByKey[p.key] = p.name;
              methodOrder.push(p.key);
            }
          })
        );
        // lookup[regimeId][methodKey] = value (null = arm not present)
        const lookup: Record<string, Record<string, number | null>> = {};
        d.regimes.forEach((r) => {
          lookup[r.id] = {};
          r.points.forEach((p) => {
            lookup[r.id][p.key] = (p as any)[metric];
          });
        });
        // best per regime column (excludes null)
        const best: Record<string, number | null> = {};
        d.regimes.forEach((r) => {
          const vals = Object.values(lookup[r.id]).filter((v): v is number => v != null);
          best[r.id] = vals.length
            ? m.better === "hi"
              ? Math.max(...vals)
              : Math.min(...vals)
            : null;
        });

        return (
          <>
            <PageHeader
              title="Comparative View"
              subtitle="Every method across every regime for a chosen metric. Missing arms are marked, never zero-filled."
              right={
                <select
                  className="select"
                  value={metric}
                  onChange={(e) => setMetric(e.target.value as MetricKey)}
                  aria-label="Select comparison metric"
                >
                  {METRICS.map((x) => (
                    <option key={x.key} value={x.key}>
                      {x.label}
                    </option>
                  ))}
                </select>
              }
            />

            <Card className="mb4">
              <CardHead title={`${m.label} — method × regime`} sub={m.cardSub} />
              <div className="tbl-wrap">
                <table className="tbl">
                  <thead>
                    <tr>
                      <th>Method</th>
                      {d.regimes.map((r) => (
                        <th key={r.id} className="num">
                          {r.short}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {methodOrder.map((mk) => (
                      <tr key={mk}>
                        <td>{nameByKey[mk]}</td>
                        {d.regimes.map((r) => {
                          const v = lookup[r.id][mk];
                          const isBest =
                            v != null &&
                            best[r.id] != null &&
                            Math.abs(v - (best[r.id] as number)) < 1e-9;
                          return (
                            <td key={r.id} className="num">
                              {v == null ? (
                                <span className="na" title="Arm not present in this regime">
                                  N/A
                                </span>
                              ) : (
                                <span
                                  style={
                                    isBest
                                      ? { color: "var(--blue)", fontWeight: 600 }
                                      : undefined
                                  }
                                >
                                  {m.fmt(v)}
                                </span>
                              )}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>

            <MetricObservation metric={metric} />

            <IntegrityBanner
              source="all regime summaries (e20, e22_arch64, e24_arch128, e24_rotmnist, e25_cnn)"
              extra="per-regime arm means; MLP regimes: 5 seeds each, CNN regime: 3 seeds"
              loaded={timestamp(d.generatedAt)}
            />
          </>
        );
      }}
    </DataGate>
  );
}
