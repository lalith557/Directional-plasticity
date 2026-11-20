import { useState } from "react";
import { Link } from "react-router-dom";
import { PageHeader } from "../../components/PageHeader";
import { Card, CardHead, Observation } from "../../components/ui";
import { ScatterSci } from "../../components/charts";
import { BackendGate, MeanStd, SourceNote } from "../../components/studio/StudioUI";
import { archLabel, DATASET_LABEL, RecordSummary, useApi } from "../../lib/api";
import { C, corrColor } from "../../lib/colors";
import { NA } from "../../data/format";

const DETAIL: Record<string, string> = { e20: "/experiments/r20", e22_arch64: "/experiments/width", e24_arch128: "/experiments/width",
                                        e24_rotmnist: "/experiments/rotated", e25_cnn: "/experiments/cnn" };

export function ArchitectureExplorer() {
  return (
    <>
      <PageHeader title="Architecture Explorer" subtitle="The same seven-arm protocol across width and task geometry. Dataset and architecture stay visible so regimes are not conflated." />
      <BackendGate feature="the regime comparison"><Body /></BackendGate>
    </>
  );
}

function Body() {
  const { data: regs } = useApi<any[]>("/analysis/regimes");
  const { data: recs } = useApi<RecordSummary[]>("/experiments?include_phase1=false");
  const [sel, setSel] = useState("e20");
  if (!regs || !recs) return <div className="studio-note">Loading…</div>;
  const arms = recs.filter((r) => r.group === sel && r.source === "legacy_artifact");
  const sgdDecay = (g: string) => recs.find((r) => r.group === g && r.method === "sgd")?.metrics?.decay?.mean ?? null;
  return (
    <div className="grid" style={{ gap: 16 }}>
      <Card>
        <CardHead title="Regime-level norm–decay association" sub="Pearson r across method means — descriptive, not causal" />
        <div className="tbl-wrap"><table className="tbl">
          <thead><tr><th>Regime</th><th>Dataset</th><th>Architecture</th><th className="num">Seeds</th><th className="num">Method means (n)</th>
            <th className="num">SGD decay</th><th className="num">r (artifact)</th><th className="num">r (paper)</th><th>Excluded</th><th /></tr></thead>
          <tbody>{regs.map((r) => (
            <tr key={r.key} className={`clickable ${sel === r.key ? "selected" : ""}`} onClick={() => setSel(r.key)}>
              <td><b>{r.label}</b></td><td>{DATASET_LABEL[r.dataset]}</td><td className="mono small">{archLabel(r.architecture)}</td>
              <td className="num">{r.seeds}</td><td className="num">{r.n}</td><td className="num mono">{sgdDecay(r.key)?.toFixed(3) ?? NA}</td>
              <td className="num mono" style={{ color: corrColor(r.pearson_r) }}>{r.pearson_r >= 0 ? "+" : ""}{r.pearson_r?.toFixed(2)}</td>
              <td className="num mono">{r.paper_r >= 0 ? "+" : ""}{r.paper_r.toFixed(2)}</td>
              <td className="small">{r.excluded.length ? r.excluded.join(", ") : "—"}</td>
              <td><Link className="linkish small" to={DETAIL[r.key]} onClick={(e) => e.stopPropagation()}>details →</Link></td>
            </tr>))}</tbody>
        </table></div>
      </Card>

      <div className="grid-2col">
        <Card pad>
          <CardHead title={regs.find((r) => r.key === sel)?.label ?? ""} sub="final ‖θ‖ vs decay, one point per method mean" />
          <ScatterSci xFmt={(v: number) => v.toFixed(1)} series={[{ name: "methods", color: C.blue, data: regs.find((r) => r.key === sel)!.points.filter((p: any) => p.x != null && p.y != null) }]}
                      xLabel="final ‖θ‖" yLabel="decay" legend={false}
                      tooltip={(d: any) => ({ title: d.label, rows: [{ label: "final ‖θ‖", value: d.x.toFixed(2) }, { label: "decay", value: d.y.toFixed(3) }] })} />
          <SourceNote>{regs.find((r) => r.key === sel)?.source}</SourceNote>
        </Card>
        <Card>
          <CardHead title="Methods in this regime" />
          <div className="tbl-wrap"><table className="tbl">
            <thead><tr><th>Method</th><th className="num">Mean acc</th><th className="num">Decay</th><th className="num">Final ‖θ‖</th></tr></thead>
            <tbody>{arms.map((r) => (
              <tr key={r.id}><td>{r.arm_label}</td><td className="num"><MeanStd m={r.metrics?.mean_acc} k="mean_acc" /></td>
                <td className="num"><MeanStd m={r.metrics?.decay} k="decay" /></td><td className="num"><MeanStd m={r.metrics?.final_weight_norm} k="final_weight_norm" /></td></tr>))}</tbody>
          </table></div>
        </Card>
      </div>
      <Observation label="Reading" tone="amber">
        The positive association belongs to the capacity-stressed width-16 regime. Where plain SGD barely loses plasticity
        (SGD decay ≤ 0.032, below the width-16 seed std of 0.036) the coefficient turns negative — the association vanishing,
        not an opposite mechanism. Each r uses 6–7 method means, so no confidence interval is shown and no causal claim is made.
        The CNN regime uses 3 seeds, η₀ = 0.1, and replaces CBP/ANR with norm projection.
      </Observation>
    </div>
  );
}
