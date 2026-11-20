import { useNavigate } from "react-router-dom";
import { DataGate } from "../components/DataGate";
import { PageHeader } from "../components/PageHeader";
import { Card, CardHead, Observation, IntegrityBanner } from "../components/ui";
import { bytes, timestamp } from "../data/format";

// which experiment each summary powers in the UI
const POWERS: Record<string, { label: string; to: string }> = {
  "e20_summary.json": { label: "R20 · Norm↔Plasticity · Run Details · Update Geometry", to: "/experiments/r20" },
  "e22_lambda_summary.json": { label: "λ Sweep", to: "/experiments/lambda" },
  "e27_radial_sweep_summary.json": { label: "Radial Sweep · Update Geometry", to: "/experiments/radial" },
  "e22_arch64_summary.json": { label: "Width Study (64) · Norm↔Plasticity", to: "/experiments/width" },
  "e24_arch128_summary.json": { label: "Width Study (128) · Norm↔Plasticity", to: "/experiments/width" },
  "e24_rotmnist_summary.json": { label: "Rotated MNIST · Norm↔Plasticity", to: "/experiments/rotated" },
  "e25_cnn_summary.json": { label: "CNN Probe · Norm↔Plasticity", to: "/experiments/cnn" },
  "e22_anr_ablation_summary.json": { label: "ANR Ablation", to: "/analysis/ablations" },
};

export function RawResults() {
  const nav = useNavigate();
  return (
    <DataGate>
      {(d) => (
        <>
          <PageHeader
            title="Raw Results"
            subtitle="The released summary files that power every number in this dashboard."
            right={
              <button className="btn ghost" onClick={() => nav("/runs")}>
                Open Run Details {"→"}
              </button>
            }
          />

          <Card className="mb4">
            <CardHead title="Source summaries" sub={`${d.rawFiles.length} files · results/tables/`} />
            <div className="tbl-wrap">
              <table className="tbl">
                <thead>
                  <tr>
                    <th>File</th>
                    <th>Powers</th>
                    <th>Path</th>
                    <th className="num">Size</th>
                  </tr>
                </thead>
                <tbody>
                  {d.rawFiles.map((f) => {
                    const p = POWERS[f.name];
                    return (
                      <tr
                        key={f.path}
                        className={p ? "clickable" : ""}
                        onClick={p ? () => nav(p.to) : undefined}
                      >
                        <td className="mono">{f.name}</td>
                        <td>{p ? p.label : <span className="muted">{"—"}</span>}</td>
                        <td className="mono muted">{f.path}</td>
                        <td className="num">{bytes(f.bytes)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>

          <Observation label="Provenance" tone="neutral">
            Every value in this dashboard originates from the files above.{" "}
            <span className="mono">web/scripts/build_data.py</span> reads them and writes{" "}
            <span className="mono">web/public/data/plasticity.json</span>; the generator does not
            invent, interpolate, or mock any value. Two statistics are derived by the generator
            rather than read directly: the width-16 Pearson and Spearman r values, computed from
            the 7 arm-level means in <span className="mono">e20_summary.json</span>. Pearson r for
            the other four regimes (width 64, width 128, rotated MLP, CNN) is pre-computed by the
            experiment scripts and read directly from the respective source files. The log-space
            trend line on the Norm{"↔"}Plasticity scatter is a client-side visual aid computed
            in the browser from the loaded data points and not stored in{" "}
            <span className="mono">plasticity.json</span>. Non-finite values are sanitized in two
            steps: the generator converts them to JSON null; UI formatters render null as
            {" "}"Not available," never as zero.
          </Observation>

          <IntegrityBanner source="results/tables/*.json" loaded={timestamp(d.generatedAt)} />
        </>
      )}
    </DataGate>
  );
}
