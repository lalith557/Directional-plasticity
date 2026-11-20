import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { DataGate } from "../components/DataGate";
import { PageHeader } from "../components/PageHeader";
import { Tabs } from "../components/Tabs";
import { Card, CardHead, Empty, IntegrityBanner, StatLine, Observation } from "../components/ui";
import { ConfigPanel } from "../components/ConfigPanel";
import { ResultTable, Column } from "../components/ResultTable";
import { acc, num, pct, timestamp } from "../data/format";
import type { PerSeedRow } from "../data/types";

const TABS = ["Metrics", "Logs", "Config", "Checkpoint"];

export function RunDetails() {
  const [params] = useSearchParams();
  const [tab, setTab] = useState("Metrics");
  const [method, setMethod] = useState(params.get("method") || "sgd");
  const [seedIdx, setSeedIdx] = useState(0);

  return (
    <DataGate>
      {(d) => {
        const activeMethod = d.r20.perSeed[method] ? method : "sgd";
        const rows = d.r20.perSeed[activeMethod] ?? [];
        const run = rows[Math.min(seedIdx, rows.length - 1)];
        const armName = d.r20.arms.find((a) => a.key === activeMethod)?.name ?? activeMethod;

        const seedCols: Column<PerSeedRow>[] = [
          { key: "seed", header: "Seed", value: (r) => r.seed, render: (r) => <span className="mono">{r.seed}</span> },
          { key: "acc", header: "Final Acc", numeric: true, sortable: true, value: (r) => r.acc, render: (r) => acc(r.acc) },
          { key: "decay", header: "Plasticity Decay", numeric: true, sortable: true, value: (r) => r.decay, render: (r) => acc(r.decay) },
          { key: "norm", header: "Final Norm", numeric: true, sortable: true, value: (r) => r.norm, render: (r) => num(r.norm, 2) },
          { key: "dormant", header: "Dormant", numeric: true, sortable: true, value: (r) => r.dormant, render: (r) => pct(r.dormant) },
        ];

        return (
          <>
            <PageHeader
              title="Run Details"
              subtitle="Inspect individual runs, metrics and configurations."
              right={
                <div className="row gap2 aic">
                  <select
                    className="select"
                    value={activeMethod}
                    onChange={(e) => {
                      setMethod(e.target.value);
                      setSeedIdx(0);
                    }}
                  >
                    {d.r20.arms.map((a) => (
                      <option key={a.key} value={a.key}>
                        {a.name}
                      </option>
                    ))}
                  </select>
                  <select className="select" value={Math.min(seedIdx, rows.length - 1)} onChange={(e) => setSeedIdx(Number(e.target.value))}>
                    {rows.map((r, i) => (
                      <option key={r.seed} value={i}>
                        seed {r.seed}
                      </option>
                    ))}
                  </select>
                </div>
              }
            />
            <Tabs tabs={TABS} active={tab} onChange={setTab} />

            {tab === "Metrics" &&
              (run ? (
                <>
                  <Card pad className="mb4">
                    <div className="card-title mb3">
                      {armName} — seed {run.seed}
                    </div>
                    <StatLine
                      stats={[
                        { label: "Train Accuracy", value: <span className="na">N/A</span> },
                        { label: "Test Accuracy (mean)", value: acc(run.acc) },
                        { label: "Peak Accuracy", value: acc(run.peakAcc ?? null) },
                        { label: "Weight Norm (final)", value: num(run.norm, 2) },
                        { label: "Plasticity Decay", value: acc(run.decay) },
                        { label: "Dormancy", value: pct(run.dormant) },
                        { label: "Effective Rank", value: num(run.effRank ?? null, 2) },
                      ]}
                    />
                    <div className="small muted mt4">
                      Train accuracy is not recorded separately in the released summaries, so it is
                      shown as <span className="na">N/A</span> rather than zero.
                    </div>
                  </Card>

                  <div className="grid mb4" style={{ gridTemplateColumns: "1fr 1fr" }}>
                    <Card>
                      <CardHead title="Training Metrics" sub="task vs accuracy" />
                      <div className="card-pad">
                        <Empty title="Per-task accuracy history not recorded for this run">
                          The released summaries store scalar aggregates only. Final test accuracy for
                          this run is <b>{acc(run.acc)}</b>.
                        </Empty>
                      </div>
                    </Card>
                    <Card>
                      <CardHead title="Norm Dynamics" sub="task vs weight norm" />
                      <div className="card-pad">
                        <Empty title="Per-task norm history not recorded for this run">
                          Only the final global weight norm is available: <b>{num(run.norm, 2)}</b>.
                        </Empty>
                      </div>
                    </Card>
                  </div>

                  <Card>
                    <CardHead title="Seed Runs" sub={`${rows.length} seeds · ${armName}`} />
                    <ResultTable
                      columns={seedCols}
                      rows={rows}
                      getRowId={(r) => String(r.seed)}
                      onRowClick={(r) => setSeedIdx(rows.findIndex((x) => x.seed === r.seed))}
                      selected={new Set([String(run.seed)])}
                      initialSort={{ key: "seed", dir: "asc" }}
                    />
                  </Card>
                </>
              ) : (
                <Observation label="No run" tone="neutral">
                  No per-seed rows are available for this method.
                </Observation>
              ))}

            {tab === "Logs" && (
              <Empty title="Per-run logs are not bundled with the released summaries">
                Aggregate stdout logs exist in the repository (e.g. results/tables/e22_stdout.log,
                e25_stdout.log) but are not split per method/seed, so no per-run log is shown here.
              </Empty>
            )}

            {tab === "Config" && run && (
              <div style={{ maxWidth: 520 }}>
                <ConfigPanel
                  title={`${d.r20.source} · ${activeMethod} · seed ${run.seed}`}
                  lines={[
                    ["dataset", "permuted-mnist"],
                    ["model", "mlp"],
                    ["width", d.r20.config.hidden],
                    ["lr", d.r20.config.lr],
                    ["steps_per_task", d.r20.config.stepsPerTask],
                    ["batch_size", d.r20.config.batch],
                    ["tasks", d.r20.config.tasks],
                    ["method", activeMethod],
                    ["seed", run.seed],
                  ]}
                />
              </div>
            )}

            {tab === "Checkpoint" && (
              <Empty title="No model checkpoints are exported in this repository">
                Results are released as per-run summaries (results/tables/*.json), not as model weight
                checkpoints. Nothing is fabricated to fill this view.
              </Empty>
            )}

            <IntegrityBanner source={d.r20.source} seeds={d.r20.seeds.length} loaded={timestamp(d.generatedAt)} />
          </>
        );
      }}
    </DataGate>
  );
}
