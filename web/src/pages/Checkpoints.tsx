import { DataGate } from "../components/DataGate";
import { PageHeader } from "../components/PageHeader";
import { Card, CardHead, Empty, Observation, IntegrityBanner } from "../components/ui";
import { bytes, timestamp } from "../data/format";

// Verified against results/tables/*.json and the experiment source files.
const DESCRIPTIONS: Record<string, string> = {
  "e20_summary.json":
    "Primary Permuted-MNIST stress experiment · MLP 784-16-16-10 · 7 arms · 5 seeds · per-arm aggregates and per-seed raw metrics",
  "e22_lambda_summary.json":
    "L2-init norm-control-strength sweep · 6 λ values · 5 seeds · MLP width 16",
  "e27_radial_sweep_summary.json":
    "Radial (inward) update-direction sweep · 6 γ values · 5 seeds · MLP width 16",
  "e22_arch64_summary.json":
    "Permuted-MNIST, MLP 784-64-64-10 · 7 arms · 5 seeds",
  "e24_arch128_summary.json":
    "Permuted-MNIST, MLP 784-128-128-10 · 7 arms · 5 seeds",
  "e24_rotmnist_summary.json":
    "Rotated-MNIST, MLP 784-16-16-10 · 7 arms · 5 seeds",
  "e25_cnn_summary.json":
    "Rotated-MNIST, SmallCNN(16,32,fc64) · 7 arms · 3 seeds · includes angle sweep",
  "e22_anr_ablation_summary.json":
    "ANR component ablation (shrink-only / revive-only / full) · 5 seeds · MLP width 16",
};

export function Checkpoints() {
  return (
    <DataGate>
      {(d) => (
        <>
          <PageHeader
            title="Checkpoints"
            subtitle="Model weight checkpoints for the released runs."
          />

          <Empty title="No model-weight checkpoints are included in this repository">
            The study releases per-run <b>experiment summaries</b> — JSON files containing
            aggregate metrics, per-seed results, and run configurations. These are not loadable
            model weights and cannot be used to restore a trained model state. Rather than show
            placeholder entries, this page lists what is actually released.
          </Empty>

          <Card className="mt4">
            <CardHead
              title="Released experiment summaries"
              sub="experiment summaries, not model weights · aggregate metrics and per-seed results · click a file to download"
            />
            <table className="tbl tbl-fixed">
              <thead>
                <tr>
                  <th>File</th>
                  <th className="num" style={{ width: 110 }}>Size</th>
                  <th style={{ width: 120 }}>Download</th>
                </tr>
              </thead>
              <tbody>
                {d.rawFiles.map((f) => {
                  const href = import.meta.env.BASE_URL + "data/raw/" + f.name;
                  return (
                    <tr key={f.path}>
                      <td>
                        <a className="mono linkish" href={href} download={f.name}>{f.name}</a>
                        <div className="small muted" style={{ marginTop: 2, fontFamily: "var(--mono)" }}>{f.path}</div>
                        {DESCRIPTIONS[f.name] && (
                          <div className="small muted" style={{ marginTop: 4 }}>{DESCRIPTIONS[f.name]}</div>
                        )}
                      </td>
                      <td className="num">{bytes(f.bytes)}</td>
                      <td>
                        <a className="dl-chip" href={href} download={f.name} aria-label={`Download ${f.name}`}>
                          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="14" height="14">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                            <polyline points="7 10 12 15 17 10" />
                            <line x1="12" y1="15" x2="12" y2="3" />
                          </svg>
                          <span>JSON</span>
                        </a>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </Card>

          <Observation label="Reproducibility scope" tone="neutral">
            The released artifacts support <b>reported-result verification</b>: every number in
            this dashboard is derived from these summary files via{" "}
            <span className="mono">web/scripts/build_data.py</span>, which is also released. The
            repository additionally includes training scripts, model architecture definitions,
            dataset loaders, and the random seeds used for each run — sufficient to{" "}
            <b>rerun experiments</b> given the same dataset and environment. It does not include
            saved model weights, optimizer states, or intermediate training checkpoints, so{" "}
            <b>exact state restoration</b> is not supported by these artifacts alone.
          </Observation>

          <IntegrityBanner source="results/tables/*.json" loaded={timestamp(d.generatedAt)} />
        </>
      )}
    </DataGate>
  );
}
