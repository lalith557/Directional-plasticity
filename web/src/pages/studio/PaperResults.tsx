import { Link } from "react-router-dom";
import { PageHeader } from "../../components/PageHeader";
import { Badge, Card, CardHead, Observation } from "../../components/ui";
import { ScatterSci } from "../../components/charts";
import { BackendGate, SourceNote, fmtMetric } from "../../components/studio/StudioUI";
import { useApi } from "../../lib/api";
import { C } from "../../lib/colors";
import { NA } from "../../data/format";

function Check({ c, k }: { c: any; k: string }) {
  if (!c) return <span className="na">—</span>;
  const tone = c.status === "artifact_consistent" ? "green" : c.status === "mismatch" ? "red" : "gray";
  return (
    <span className="check-cell">
      <span className="mono">{typeof c.paper === "number" ? (k === "matched_max_dev" ? c.paper.toExponential(1) : c.paper) : c.paper}</span>
      <span className="muted"> · artifact </span>
      <span className="mono">{c.artifact == null ? NA : k === "matched_max_dev" ? c.artifact.toExponential(2) : k === "n" ? String(c.artifact) : Number(c.artifact).toFixed(k.includes("norm") ? 2 : 4)}</span>
      <Badge tone={tone as any}>{c.status === "artifact_consistent" ? "consistent" : c.status}</Badge>
    </span>
  );
}

export function PaperResults() {
  return (
    <>
      <PageHeader title="Paper Results" subtitle="Each claim in the paper, the value as printed, and the artifact it is recomputed from." />
      <div className="legend-strip mb4">
        <Badge tone="gray">Paper-reported</Badge><span className="small">printed in paper/main(iconcise).tex</span>
        <Badge tone="green">Existing artifact</Badge><span className="small">recomputed from results/tables/</span>
        <Badge tone="blue">Newly reproduced</Badge><span className="small">Studio run matched a per-seed record</span>
        <Badge tone="amber">Newly generated</Badge><span className="small">Studio run with no legacy counterpart</span>
      </div>
      <BackendGate feature="the paper–evidence mapping"><Body /></BackendGate>
    </>
  );
}

function Body() {
  const { data } = useApi<any>("/paper-results", 10000);
  if (!data) return <div className="studio-note">Loading…</div>;
  const [f1, f2, f3, f4, f5] = data.findings;
  const repro = data.studio_evidence.reproduced;
  return (
    <div className="grid" style={{ gap: 16 }}>
      <Card pad>
        <div className="row between aic wrap">
          <div><b>{data.consistency.artifact_consistent} / {data.consistency.checks}</b> paper values are consistent with the released artifacts at the paper's printed precision.</div>
          <div className="small">Studio reproductions: <b>{repro.filter((r: any) => r.status.startsWith("reproduced")).length}</b> matched ·{" "}
            <b>{repro.filter((r: any) => r.status === "mismatch").length}</b> mismatched · <b>{data.studio_evidence.generated_count}</b> new runs without a legacy counterpart</div>
        </div>
        {repro.length > 0 && (
          <div className="tbl-wrap mt3"><table className="tbl"><thead><tr><th>Studio run</th><th>Reproduces</th><th className="num">Seed</th><th>Result</th><th className="num">Max |Δ|</th></tr></thead>
            <tbody>{repro.map((r: any) => <tr key={r.run_id}><td className="mono small"><Link className="linkish" to={`/explorer/${r.run_id}`}>{r.run_id}</Link></td>
              <td className="mono small">{r.against}</td><td className="num">{r.seed}</td>
              <td><Badge tone={r.status === "mismatch" ? "red" : "blue"}>{r.status}</Badge></td><td className="num mono">{r.max_abs_diff?.toExponential(1) ?? "—"}</td></tr>)}</tbody></table></div>
        )}
      </Card>

      <Card pad>
        <CardHead title={`Finding 1 — ${f1.title}`} sub="method-level association, width-16 permuted MNIST" />
        <div className="grid-side">
          <ScatterSci xFmt={(v: number) => v.toFixed(1)} series={[{ name: "method means", color: C.blue, data: f1.points.filter((p: any) => p.x != null),
                                trend: f1.correlation.fit ? [Math.min(...f1.points.map((p: any) => p.x)), Math.max(...f1.points.map((p: any) => p.x))].map((x: number) => ({ x, y: f1.correlation.fit.slope * x + f1.correlation.fit.intercept })) : undefined }]}
                      xLabel="final ‖θ‖" yLabel="decay" legend={false}
                      tooltip={(d: any) => ({ title: d.label, rows: [{ label: "final ‖θ‖", value: d.x.toFixed(2) }, { label: "decay", value: d.y.toFixed(3) }] })} />
          <div>
            <div className="kv">
              <div className="kv-k">Pearson r</div><div className="kv-v"><Check c={f1.checks.pearson_r} k="r" /></div>
              <div className="kv-k">Spearman ρ</div><div className="kv-v"><Check c={f1.checks.spearman_rho} k="r" /></div>
              <div className="kv-k">n</div><div className="kv-v"><Check c={f1.checks.n} k="n" /></div>
              <div className="kv-k">Per-seed r</div><div className="kv-v mono">{f1.per_seed_r.map((x: number) => x.toFixed(2)).join(", ")}</div>
              <div className="kv-k">Leave-one-out r</div><div className="kv-v mono">{Math.min(...f1.leave_one_out.map((x: any) => x.pearson_r)).toFixed(2)} – {Math.max(...f1.leave_one_out.map((x: any) => x.pearson_r)).toFixed(2)}</div>
              <div className="kv-k">95% CI</div><div className="kv-v small">{f1.correlation.ci_note}</div>
            </div>
            <Observation label="Scope" tone="amber">{f1.correlation.scope_note}</Observation>
          </div>
        </div>
        <SourceNote>{f1.artifact} · records {f1.records.length}</SourceNote>
      </Card>

      <Card pad>
        <CardHead title={`Finding 2 — ${f2.title}`} sub={f2.kind} />
        <div className="identity mono">{f2.statement}</div>
        {f2.corollaries.map((c: string) => <div key={c} className="small mt2">{c}</div>)}
        {f2.artifact && (
          <div className="kv mt3">
            <div className="kv-k">Measured: ‖θ_T‖² − ‖θ₀‖²</div><div className="kv-v mono">{f2.artifact.endpoint_delta_mean.toFixed(2)}</div>
            <div className="kv-k">Measured: Σ η_t²</div><div className="kv-v mono">{f2.artifact.sum_eta_sq_mean.toFixed(2)}</div>
            <div className="kv-k">Relative error</div><div className="kv-v mono">{f2.artifact.relative_error_mean.toExponential(1)} (max {f2.artifact.relative_error_max.toExponential(1)}) · paper {f2.paper.relative_error}</div>
            <div className="kv-k">Mean cos(update, θ)</div><div className="kv-v mono">{f2.artifact.radial_cos_mean.toExponential(1)} · paper {f2.paper.cosine}</div>
          </div>
        )}
        <div className="small mt3">Interactive geometry: <Link className="linkish" to="/analysis/geometry">Update Geometry</Link> (a mathematical illustration, separate from these measurements).</div>
        <SourceNote>{f2.artifact?.source}</SourceNote>
      </Card>

      <Card>
        <CardHead title={`Finding 3 — ${f3.title}`} sub="Table I: paper value · artifact value · consistency" />
        <div className="tbl-wrap"><table className="tbl">
          <thead><tr><th>Method</th><th>Mean acc</th><th>Decay</th><th>Final ‖θ‖</th><th>Dormant</th><th>Match dev.</th></tr></thead>
          <tbody>{f3.table.map((r: any) => (
            <tr key={r.key}><td><Link className="linkish" to={`/explorer/${r.id}`}>{r.label}</Link></td>
              <td><Check c={r.checks.mean_acc} k="mean_acc" /></td><td><Check c={r.checks.decay} k="decay" /></td>
              <td><Check c={r.checks.final_weight_norm} k="final_weight_norm" /></td><td><Check c={r.checks.final_dormant} k="final_dormant" /></td>
              <td>{r.checks.matched_max_dev ? <Check c={r.checks.matched_max_dev} k="matched_max_dev" /> : <span className="na">{r.key === "sgd" ? "ref" : "—"}</span>}</td></tr>))}</tbody>
        </table></div>
        <div className="card-pad">
          <div className="section-label">Paired-seed accuracy differences (5 seeds)</div>
          <div className="tbl-wrap"><table className="tbl"><thead><tr><th>Comparison</th><th className="num">Paper</th><th className="num">Artifact mean diff</th><th className="num">Seeds better</th><th className="num">Sign test p</th></tr></thead>
            <tbody>{Object.entries<any>(f3.paired_mean_acc).map(([k, p]) => (
              <tr key={k}><td>{k}</td><td className="num mono">{p.paper >= 0 ? "+" : ""}{p.paper.toFixed(3)}</td>
                <td className="num mono">{p.descriptive.mean_difference >= 0 ? "+" : ""}{p.descriptive.mean_difference.toFixed(3)}</td>
                <td className="num">{p.descriptive.a_better_on}/{p.descriptive.of}</td><td className="num mono">{p.tests.sign_test_p_two_sided.toFixed(4)}</td></tr>))}</tbody></table></div>
          <div className="small muted mt2">The paper reports paired differences rather than significance tests; with n = 5 the exact sign test cannot reach p &lt; 0.05.</div>
          <div className="section-label mt4">Radial sweep (Fig. 3)</div>
          <div className="tbl-wrap"><table className="tbl"><thead><tr><th>γ</th><th className="num">Mean acc</th><th className="num">Final ‖θ‖ (paper)</th><th className="num">Final ‖θ‖ (artifact)</th><th className="num">Decay</th></tr></thead>
            <tbody>{f3.radial.map((r: any, i: number) => (
              <tr key={r.id}><td className="mono">{r.gamma}</td><td className="num mono">{fmtMetric(r.mean_acc, "mean_acc")}</td>
                <td className="num mono">{String(f3.radial_paper.norms[i])}</td><td className="num mono">{r.final_weight_norm?.toFixed(3) ?? NA}</td>
                <td className="num mono">{fmtMetric(r.decay, "decay")}</td></tr>))}</tbody></table></div>
        </div>
      </Card>

      <Card>
        <CardHead title={`Finding 4 — ${f4.title}`} />
        <div className="tbl-wrap"><table className="tbl"><thead><tr><th>Variant</th><th className="num">Mean acc</th><th className="num">Decay</th><th className="num">Final ‖θ‖</th><th className="num">Dormant</th></tr></thead>
          <tbody>{f4.variants.map((v: any) => <tr key={v.id}><td><Link className="linkish" to={`/explorer/${v.id}`}>{v.label}</Link></td>
            <td className="num mono">{fmtMetric(v.mean_acc, "mean_acc")}</td><td className="num mono">{fmtMetric(v.decay, "decay")}</td>
            <td className="num mono">{fmtMetric(v.final_weight_norm, "final_weight_norm")}</td><td className="num mono">{fmtMetric(v.final_dormant, "final_dormant")}</td></tr>)}</tbody></table></div>
        <div className="card-pad"><div className="checks-grid">{Object.entries<any>(f4.checks).map(([k, c]) => <div key={k}><span className="small muted">{k.replace(/_/g, " ")}</span><br /><Check c={c} k={k} /></div>)}</div></div>
      </Card>

      <Card>
        <CardHead title={`Finding 5 — ${f5.title}`} sub="paper-reported reference values linked to the artifacts they come from" />
        <div className="tbl-wrap"><table className="tbl">
          <thead><tr><th>Regime</th><th>Dataset · architecture</th><th className="num">Seeds</th><th className="num">n arms</th><th>Pearson r</th><th>SGD decay</th><th>Artifact</th></tr></thead>
          <tbody>{f5.regimes.map((r: any) => (
            <tr key={r.key}><td><b>{r.label}</b></td><td className="small">{r.artifact.dataset} · {r.artifact.architecture?.name}{r.artifact.architecture?.hidden ? ` h=${r.artifact.architecture.hidden}` : ""}</td>
              <td className="num">{r.artifact.seeds}</td><td className="num"><Check c={r.checks.n_arms} k="n" /></td>
              <td><Check c={r.checks.r} k="r" /></td><td><Check c={r.checks.sgd_decay} k="decay" /></td>
              <td className="mono small">{r.artifact.source}</td></tr>))}</tbody>
        </table></div>
        <div className="card-pad small muted">“Consistent” means the artifact value rounds to the printed value; it is not an independent reproduction. A regime becomes “newly reproduced” only when Studio runs with matching configurations reproduce its per-seed values.</div>
      </Card>
      <SourceNote><span className="mono">GET /paper-results</span> — paper values carry their exact LaTeX source and are checked against the .tex by <span className="mono">studio/tests/test_paper_results.py</span></SourceNote>
    </div>
  );
}
