import { DataGate } from "../components/DataGate";
import { PageHeader } from "../components/PageHeader";
import { Card, CardHead, Observation, IntegrityBanner, Badge } from "../components/ui";
import { timestamp } from "../data/format";

const DATASETS = [
  {
    name: "Permuted MNIST",
    tag: "primary",
    body: "Each task applies a distinct, per-task random pixel permutation to MNIST drawn from the run seed and held fixed during training; labels are unchanged. A single shared 10-way head is used across all 100 tasks, avoiding the class-incremental head-race. Used by R20, the λ sweep, the radial sweep, and the width study (widths 16 / 64 / 128).",
    regimes: ["width16", "width64", "width128"],
  },
  {
    name: "Rotated MNIST",
    tag: "boundary",
    body: "Each task applies a per-task rotation angle drawn uniformly from [0°, 180°) and held fixed during training; labels are unchanged. Rotations preserve spatial structure and apply a smooth geometric change, giving a milder sequential-adaptation challenge than pixel permutations. Used to assess whether the norm–plasticity relationship generalizes across task geometry (MLP width 16) and architecture (small CNN).",
    regimes: ["rot_mlp", "rot_cnn"],
  },
];

export function Datasets() {
  return (
    <DataGate>
      {(d) => (
        <>
          <PageHeader
            title="Datasets"
            subtitle="Task streams used across the study. Scope is deliberately MNIST-family only."
          />

          <div className="grid-2col mb4">
            {DATASETS.map((ds) => {
              const regimes = d.regimes.filter((r) => ds.regimes.includes(r.id));
              return (
                <Card key={ds.name}>
                  <CardHead
                    title={ds.name}
                    right={<Badge tone={ds.tag === "primary" ? "blue" : "amber"}>{ds.tag}</Badge>}
                  />
                  <div className="card-pad">
                    <p className="small" style={{ marginTop: 0, color: "var(--ink-2)", lineHeight: 1.6 }}>
                      {ds.body}
                    </p>
                    <div className="tbl-wrap mt4">
                      <table className="tbl">
                        <thead>
                          <tr>
                            <th>Regime</th>
                            <th>Architecture</th>
                            <th className="num">Seeds</th>
                          </tr>
                        </thead>
                        <tbody>
                          {regimes.map((r) => (
                            <tr key={r.id}>
                              <td>{r.short}</td>
                              <td className="mono">{r.arch}</td>
                              <td className="num">{r.seeds}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </Card>
              );
            })}
          </div>

          <Observation label="Scope" tone="neutral">
            All results are on the MNIST family (permuted and rotated); no CIFAR / ImageNet-scale
            data is claimed. The source loader is <span className="mono">src/datasets.py</span> reading
            from <span className="mono">Deep Learning Dataset1</span>.
          </Observation>

          <IntegrityBanner source="src/datasets.py · results/tables/*" loaded={timestamp(d.generatedAt)} />
        </>
      )}
    </DataGate>
  );
}
