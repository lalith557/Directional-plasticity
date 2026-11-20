import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { PageHeader } from "../../components/PageHeader";
import { Card, CardHead, Observation } from "../../components/ui";
import { Tabs } from "../../components/Tabs";
import { BarSci, LineSci, ScatterSci } from "../../components/charts";
import { BackendGate, MeanStd, PALETTE, SeriesChart, SourceNote, fmtMetric } from "../../components/studio/StudioUI";
import { api, RecordSummary, useApi } from "../../lib/api";
import { C, FAMILY_COLOR } from "../../lib/colors";
import { NA } from "../../data/format";

const GROUPS: [string, string][] = [
  ["e20", "Permuted MNIST · MLP h=16 (e20, Table I)"], ["e22_arch64", "Permuted MNIST · MLP h=64 (e22)"],
  ["e24_arch128", "Permuted MNIST · MLP h=128 (e24)"], ["e24_rotmnist", "Rotated MNIST · MLP h=16 (e24)"],
  ["e25_cnn", "Rotated MNIST · small CNN, 3 seeds (e25)"],
];
const TABS = ["Plasticity", "Weight norm", "Norm–plasticity", "Radial coefficient", "Dormant neurons", "Effective rank"];
const fam = (f?: string) => FAMILY_COLOR[f === "reset" ? "other" : f ?? "other"] ?? C.gray;
const m = (r: RecordSummary, k: string) => r.metrics?.[k]?.mean ?? null;
const sd = (r: RecordSummary, k: string) => r.metrics?.[k]?.std ?? null;

export function ScientificAnalysis() {
  const [tab, setTab] = useState(TABS[0]);
  return (
    <>
      <PageHeader title="Scientific Analysis" subtitle="Measured results from the released artifacts and from Studio runs. Every chart names its source." />
      <BackendGate feature="the analysis studio">
        <Tabs tabs={TABS} active={tab} onChange={setTab} />
        <div className="mt4"><Analysis tab={tab} /></div>
      </BackendGate>
    </>
  );
}

function GroupSelect({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <select className="select" value={value} onChange={(e) => onChange(e.target.value)} aria-label="Experiment group">
      {GROUPS.map(([k, l]) => <option key={k} value={k}>{l}</option>)}
    </select>
  );
}

function Analysis({ tab }: { tab: string }) {
  const { data } = useApi<RecordSummary[]>("/experiments?include_phase1=false");
  const [group, setGroup] = useState("e20");
  const studioGroups = useStudioGroups(data ?? []);          // hooks before any early return
  if (!data) return <div className="studio-note">Loading records…</div>;
  const arms = data.filter((r) => r.group === group && r.source === "legacy_artifact");
  switch (tab) {
    case "Plasticity": return <PlasticityTab arms={arms} group={group} setGroup={setGroup} studioGroups={studioGroups} />;
    case "Weight norm": return <NormTab data={data} arms={arms} group={group} setGroup={setGroup} studioGroups={studioGroups} />;
    case "Norm–plasticity": return <AssocTab />;
    case "Radial coefficient": return <RadialTab data={data} />;
    case "Dormant neurons": return <DormancyTab data={data} arms={arms} group={group} setGroup={setGroup} studioGroups={studioGroups} />;
    default: return <RankTab arms={arms} group={group} setGroup={setGroup} studioGroups={studioGroups} />;
  }
}

/* ---------- studio per-task curves (shared) ---------- */
function useStudioGroups(data: RecordSummary[]) {
  return useMemo(() => {
    const g = new Map<string, RecordSummary[]>();
    data.filter((r) => r.source === "studio" && r.status === "completed").forEach((r) => g.set(r.group!, [...(g.get(r.group!) ?? []), r]));
    return [...g.entries()].map(([gid, members]) => ({ id: members.length > 1 ? `group:${gid}` : members[0].id, name: members[0].name, n: members.length }));
  }, [data]);
}

function StudioCurves({ studioGroups, metric, label, fmt }: { studioGroups: { id: string; name: string; n: number }[]; metric: string; label: string; fmt: (v: number) => string }) {
  const [sel, setSel] = useState<string[]>([]);
  const [curves, setCurves] = useState<any>(null);
  useEffect(() => {
    if (!sel.length) { setCurves(null); return; }
    api("/analysis/curves", { method: "POST", json: { ids: sel } }).then(setCurves).catch(() => setCurves(null));
  }, [sel.join(",")]); // eslint-disable-line react-hooks/exhaustive-deps
  const pal = PALETTE();
  return (
    <Card pad>
      <CardHead title={`${label} across tasks — Studio runs`} sub="mean ± 1 sd across seeds where a group has several seeds" />
      {studioGroups.length === 0 ? (
        <div className="studio-note">No completed Studio runs yet. Per-task curves were not persisted by the original experiment scripts, so they exist only for runs executed through the <Link className="linkish" to="/studio">Experiment Studio</Link>.</div>
      ) : (
        <>
          <div className="chips mb3">
            {studioGroups.map((g) => (
              <span key={g.id} className={`chip ${sel.includes(g.id) ? "on" : ""}`}
                    onClick={() => setSel((s) => (s.includes(g.id) ? s.filter((x) => x !== g.id) : [...s, g.id]))}>
                {g.name}{g.n > 1 ? ` · ${g.n} seeds` : ""}
              </span>
            ))}
          </div>
          {curves ? (
            <SeriesChart series={Object.entries<any>(curves).filter(([, c]) => c.curves?.[metric]).map(([id, c], i) => ({
              key: `s${i}`, name: c.name ?? id, color: pal[i % pal.length], mean: c.curves[metric].mean, std: c.curves[metric].std,
            }))} yFmt={fmt} />
          ) : <div className="studio-note">Select one or more runs.</div>}
          <SourceNote><span className="mono">POST /analysis/curves</span> → events.jsonl / result.json of the selected runs</SourceNote>
        </>
      )}
    </Card>
  );
}

/* ---------- tabs ---------- */
function PlasticityTab({ arms, group, setGroup, studioGroups }: any) {
  const ref = arms.find((r: RecordSummary) => r.method === "sgd");
  return (
    <div className="grid" style={{ gap: 16 }}>
      <Card pad>
        <CardHead title="Mean accuracy and plasticity decay by method" right={<GroupSelect value={group} onChange={setGroup} />} />
        <div className="grid-2col">
          <BarSci data={arms.map((r: RecordSummary) => ({ name: r.arm_label, s: (r.arm_label ?? "").slice(0, 12), v: m(r, "mean_acc"), e: sd(r, "mean_acc"), f: r.family }))}
                  xKey="s" yKey="v" stdKey="e" colorFor={(d) => fam(d.f)} valueFmt={(v) => v.toFixed(3)} metricLabel="Mean acc" />
          <BarSci data={arms.map((r: RecordSummary) => ({ name: r.arm_label, s: (r.arm_label ?? "").slice(0, 12), v: m(r, "decay"), e: sd(r, "decay"), f: r.family }))}
                  xKey="s" yKey="v" stdKey="e" colorFor={(d) => fam(d.f)} valueFmt={(v) => v.toFixed(3)} metricLabel="Decay" />
        </div>
        <div className="tbl-wrap mt3"><table className="tbl">
          <thead><tr><th>Method</th><th className="num">Mean acc</th><th className="num">Δ acc vs SGD</th><th className="num">Peak acc</th><th className="num">Tail acc (last 10%)</th><th className="num">Decay</th></tr></thead>
          <tbody>{arms.map((r: RecordSummary) => (
            <tr key={r.id}><td>{r.arm_label}</td><td className="num"><MeanStd m={r.metrics?.mean_acc} k="mean_acc" /></td>
              <td className="num mono">{ref && m(r, "mean_acc") != null ? ((m(r, "mean_acc")! - m(ref, "mean_acc")!) >= 0 ? "+" : "") + (m(r, "mean_acc")! - m(ref, "mean_acc")!).toFixed(3) : NA}</td>
              <td className="num mono">{fmtMetric(m(r, "peak_acc"), "mean_acc")}</td><td className="num mono">{fmtMetric(m(r, "end_acc"), "mean_acc")}</td>
              <td className="num"><MeanStd m={r.metrics?.decay} k="decay" /></td></tr>))}</tbody>
        </table></div>
        <div className="small muted mt2">Δ vs SGD is a descriptive difference of means; paired-seed statistics are in <Link className="linkish" to="/compare">Method Comparison</Link>. Peak/tail values exist only where the artifact stored them (e20).</div>
        <SourceNote>{arms[0]?.artifact}</SourceNote>
      </Card>
      <StudioCurves studioGroups={studioGroups} metric="acc" label="Task accuracy a_t" fmt={(v) => v.toFixed(2)} />
    </div>
  );
}

function NormTab({ data, arms, group, setGroup, studioGroups }: any) {
  const lam = data.filter((r: RecordSummary) => r.group === "e22_lambda").sort((a: any, b: any) => a.sweep.value - b.sweep.value);
  const { data: cor1 } = useApi<any>("/experiments/legacy.cor1.orthogonal");
  const avg = (k: string) => { const v = (cor1?.per_seed ?? []).map((r: any) => r[k]).filter((x: any) => x != null);
                               return v.length ? v.reduce((a: number, b: number) => a + b, 0) / v.length : null; };
  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="grid-2col">
        <Card pad>
          <CardHead title="Final global norm by method" right={<GroupSelect value={group} onChange={setGroup} />} />
          <BarSci data={arms.map((r: RecordSummary) => ({ name: r.arm_label, s: (r.arm_label ?? "").slice(0, 12), v: m(r, "final_weight_norm"), e: sd(r, "final_weight_norm"), f: r.family }))}
                  xKey="s" yKey="v" stdKey="e" colorFor={(d) => fam(d.f)} valueFmt={(v) => v.toFixed(2)} metricLabel="Final ‖θ‖" />
          <div className="small muted">Diverged arms (non-finite norm) are omitted, not drawn as 0.</div>
          <SourceNote>{arms[0]?.artifact}</SourceNote>
        </Card>
        <Card pad>
          <CardHead title="Norm-control strength λ → final norm, decay, accuracy" sub="L2-init, width 16, 5 seeds (e22-B)" />
          <LineSci data={lam.map((r: any) => ({ lam: r.sweep.value === 0 ? "0" : r.sweep.value.toExponential(0), norm: m(r, "final_weight_norm"),
                                                 decay: m(r, "decay"), acc: m(r, "mean_acc") }))}
                   xKey="lam" lines={[{ key: "acc", name: "mean acc", color: C.blue }, { key: "decay", name: "decay", color: C.red },
                                      { key: "norm", name: "final ‖θ‖ (right)", color: C.navy, axis: "right" }]}
                   legend leftFmt={(v) => v.toFixed(2)} xLabel="λ" />
          <SourceNote>results/tables/e22_lambda_summary.json</SourceNote>
        </Card>
      </div>
      <Card pad>
        <CardHead title="Initial vs final norm" sub="what each artifact actually recorded" />
        <div className="small">
          Existing artifacts store only the <b>final</b> backbone norm (the paper's “≈3.3 at initialization” is the width-16 backbone at init).
          The <Link className="linkish" to="/explorer/legacy.cor1.orthogonal">Cor. 1 audit</Link> measured the <b>global</b> norm (all parameters, incl. head)
          for the orthogonal arm: <b className="mono">{fmtMetric(avg("initial_global_norm"), "final_weight_norm")}</b> at initialization →{" "}
          <b className="mono">{fmtMetric(avg("final_global_norm"), "final_weight_norm")}</b> after 100 tasks (mean of 5 seeds).
          Studio runs store the norm after every task plus the global ‖θ₀‖² and ‖θ_T‖² (MLP), shown on each run's detail page.
        </div>
      </Card>
      <StudioCurves studioGroups={studioGroups} metric="weight_norm" label="Global weight norm" fmt={(v) => v.toFixed(1)} />
    </div>
  );
}

function AssocTab() {
  const { data: regs } = useApi<any[]>("/analysis/regimes");
  const [sel, setSel] = useState("e20");
  if (!regs) return <div className="studio-note">Loading…</div>;
  const r = regs.find((x) => x.key === sel)!;
  const pts = r.points.filter((p: any) => p.x != null && p.y != null);
  const xs = pts.map((p: any) => p.x);
  const trend = r.fit && xs.length ? [Math.min(...xs), Math.max(...xs)].map((x) => ({ x, y: r.fit.slope * x + r.fit.intercept })) : undefined;
  return (
    <div className="grid" style={{ gap: 16 }}>
      <Card pad>
        <CardHead title="Final global norm vs plasticity decay" sub="each point is one method's mean over seeds"
                  right={<select className="select" value={sel} onChange={(e) => setSel(e.target.value)}>{regs.map((x) => <option key={x.key} value={x.key}>{x.label}</option>)}</select>} />
        <div className="grid-side">
          <ScatterSci xFmt={(v: number) => v.toFixed(1)} series={[{ name: "method means", color: C.blue, data: pts, trend }]} xLabel="final ‖θ‖" yLabel="decay" legend={false}
                      tooltip={(d: any) => ({ title: d.label, rows: [{ label: "final ‖θ‖", value: d.x.toFixed(2) }, { label: "decay", value: d.y.toFixed(3) }] })} />
          <div>
            <div className="kv">
              <div className="kv-k">Pearson r</div><div className="kv-v mono">{r.pearson_r?.toFixed(3) ?? NA}</div>
              <div className="kv-k">Spearman ρ</div><div className="kv-v mono">{r.spearman_rho?.toFixed(3) ?? NA}</div>
              <div className="kv-k">n</div><div className="kv-v">{r.n} method-level means ({r.seeds} seeds each)</div>
              <div className="kv-k">Excluded</div><div className="kv-v">{r.excluded.length ? r.excluded.join(", ") + " (non-finite)" : "none"}</div>
              <div className="kv-k">Paper value</div><div className="kv-v mono">{r.paper_r >= 0 ? "+" : ""}{r.paper_r.toFixed(2)}</div>
              <div className="kv-k">95% CI</div><div className="kv-v small">{r.ci95 ? r.ci95.map((v: number) => v.toFixed(2)).join(" – ") : r.ci_note}</div>
            </div>
            <Observation label="Scope" tone="amber">{r.scope_note} The dashed line is a least-squares fit across method means, shown for orientation only.</Observation>
          </div>
        </div>
        <SourceNote>{r.source} · <span className="mono">GET /analysis/regimes</span></SourceNote>
      </Card>
    </div>
  );
}

function RadialTab({ data }: { data: RecordSummary[] }) {
  const rad = data.filter((r) => r.group === "e27_radial").sort((a, b) => a.sweep!.value - b.sweep!.value);
  const studio = data.filter((r) => r.source === "studio" && r.method === "dir_inward" && r.status === "completed");
  const rows = rad.map((r) => ({ g: r.sweep!.value.toFixed(1), acc: m(r, "mean_acc"), decay: m(r, "decay"), dormant: m(r, "final_dormant"),
                                 norm: m(r, "final_weight_norm") != null && m(r, "final_weight_norm")! > 0 ? m(r, "final_weight_norm") : null }));
  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="grid-2col">
        <Card pad>
          <CardHead title="γ → accuracy and final norm" sub="right axis log; γ = 1.0 norm is exactly 0 (below the log floor)" />
          <LineSci data={rows} xKey="g" lines={[{ key: "acc", name: "mean acc", color: C.blue }, { key: "norm", name: "final ‖θ‖ (right, log)", color: C.navy, axis: "right" }]}
                   rightLog legend leftFmt={(v) => v.toFixed(2)} xLabel="inward radial coefficient γ" />
        </Card>
        <Card pad>
          <CardHead title="γ → decay and dormant fraction" />
          <LineSci data={rows} xKey="g" lines={[{ key: "decay", name: "decay", color: C.red }, { key: "dormant", name: "dormant fraction", color: C.amber }]}
                   legend leftFmt={(v) => v.toFixed(2)} xLabel="inward radial coefficient γ" />
        </Card>
      </div>
      <Card pad>
        <CardHead title="Trade-off" />
        <div className="tbl-wrap"><table className="tbl"><thead><tr><th>γ</th><th className="num">Mean acc</th><th className="num">Decay</th><th className="num">Final ‖θ‖</th><th className="num">Dormant</th></tr></thead>
          <tbody>{rad.map((r) => <tr key={r.id}><td className="mono">{r.sweep!.value}</td><td className="num"><MeanStd m={r.metrics?.mean_acc} k="mean_acc" /></td>
            <td className="num"><MeanStd m={r.metrics?.decay} k="decay" /></td><td className="num"><MeanStd m={r.metrics?.final_weight_norm} k="final_weight_norm" /></td>
            <td className="num mono">{fmtMetric(m(r, "final_dormant"), "final_dormant")}</td></tr>)}</tbody></table></div>
        <Observation label="Reading">Reducing the norm with direction alone requires spending part of the fixed step inward (Cor. 2). A mild bias (γ = 0.2) slightly beats SGD; larger γ trades task descent for norm and collapses learning at γ ≥ 0.8. The released artifact does not record the per-step radial allocation; Studio runs of the inward rule do (below).</Observation>
        <div className="section-label mt3">Measured radial allocation — Studio runs of inward steering</div>
        {studio.length ? (
          <StudioAllocation ids={studio.map((r) => r.id)} />
        ) : <div className="studio-note">No completed Studio run of inward steering yet.</div>}
        <SourceNote>results/tables/e27_radial_sweep_summary.json (sweep) · Studio result.json (allocation)</SourceNote>
      </Card>
    </div>
  );
}

function StudioAllocation({ ids }: { ids: string[] }) {
  const [rows, setRows] = useState<any[]>([]);
  useEffect(() => {
    Promise.all(ids.slice(0, 12).map((i) => api(`/experiments/${i}`))).then((rs) => setRows(rs.map((r) => {
      const pt = r.result?.summary?.per_task;
      const mean = (a?: (number | null)[]) => { const v = (a ?? []).filter((x): x is number => x != null); return v.length ? v.reduce((p, c) => p + c, 0) / v.length : null; };
      return { id: r.id, g: r.config.method.params.dir_coeff, seed: r.config.seed, energy: mean(pt?.radial_energy_frac), acc: r.result?.summary?.mean_acc };
    })));
  }, [ids.join(",")]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div className="tbl-wrap"><table className="tbl"><thead><tr><th>Run</th><th className="num">γ</th><th className="num">Seed</th><th className="num">Mean share of ‖Δθ'‖² spent radially</th><th className="num">Mean acc</th></tr></thead>
      <tbody>{rows.map((r) => <tr key={r.id}><td className="mono small"><Link className="linkish" to={`/explorer/${r.id}`}>{r.id}</Link></td><td className="num">{r.g}</td><td className="num">{r.seed}</td>
        <td className="num mono">{r.energy != null ? `${(100 * r.energy).toFixed(2)}%` : NA}</td><td className="num mono">{fmtMetric(r.acc, "mean_acc")}</td></tr>)}</tbody></table></div>
  );
}

function DormancyTab({ data, arms, group, setGroup, studioGroups }: any) {
  const anr = data.filter((r: RecordSummary) => r.group === "e22_anr");
  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="grid-2col">
        <Card pad>
          <CardHead title="Final dormant fraction by method" right={<GroupSelect value={group} onChange={setGroup} />} />
          <BarSci data={arms.map((r: RecordSummary) => ({ name: r.arm_label, s: (r.arm_label ?? "").slice(0, 12), v: m(r, "final_dormant"), f: r.family }))}
                  xKey="s" yKey="v" colorFor={(d) => fam(d.f)} valueFmt={(v) => `${(100 * v).toFixed(1)}%`} metricLabel="Dormant" />
          <div className="small muted">Aggregate fraction of hidden units with mean |activation| &lt; 10⁻³ on the probe batch. Individual neuron activations were not logged and are not shown.</div>
          <SourceNote>{arms[0]?.artifact}</SourceNote>
        </Card>
        <Card pad>
          <CardHead title="ANR ablation — dormancy reduction vs learning" sub="width 16, 5 seeds (e22-C)" />
          <div className="tbl-wrap"><table className="tbl"><thead><tr><th>Variant</th><th className="num">Dormant</th><th className="num">Mean acc</th><th className="num">Decay</th><th className="num">Final ‖θ‖</th></tr></thead>
            <tbody>{anr.map((r: RecordSummary) => <tr key={r.id}><td>{r.arm_label}</td><td className="num mono">{fmtMetric(m(r, "final_dormant"), "final_dormant")}</td>
              <td className="num"><MeanStd m={r.metrics?.mean_acc} k="mean_acc" /></td><td className="num"><MeanStd m={r.metrics?.decay} k="decay" /></td>
              <td className="num mono">{fmtMetric(m(r, "final_weight_norm"), "final_weight_norm")}</td></tr>)}</tbody></table></div>
          <Observation label="Reading">Revival alone removes dormancy and gives the best ANR accuracy; the per-unit shrink raises dormancy and lowers accuracy. Eliminating dormant units is not by itself sufficient to match global magnitude control.</Observation>
          <SourceNote>results/tables/e22_anr_ablation_summary.json</SourceNote>
        </Card>
      </div>
      <StudioCurves studioGroups={studioGroups} metric="dormant" label="Dormant fraction" fmt={(v) => v.toFixed(2)} />
    </div>
  );
}

function RankTab({ arms, group, setGroup, studioGroups }: any) {
  const [cAcc, setCAcc] = useState<any>(null);
  const [cDec, setCDec] = useState<any>(null);
  useEffect(() => {
    const ids = arms.map((r: RecordSummary) => r.id);
    if (ids.length < 3) return;
    api("/analysis/correlation", { method: "POST", json: { ids, x: "final_eff_rank", y: "mean_acc" } }).then(setCAcc);
    api("/analysis/correlation", { method: "POST", json: { ids, x: "final_eff_rank", y: "decay" } }).then(setCDec);
  }, [group, arms.length]); // eslint-disable-line react-hooks/exhaustive-deps
  const sc = (c: any, yl: string) => c && (
    <div>
      <ScatterSci xFmt={(v: number) => v.toFixed(1)} series={[{ name: "methods", color: C.green, data: c.points.filter((p: any) => p.x != null && p.y != null) }]} xLabel="final effective rank" yLabel={yl} legend={false}
                  height={220} tooltip={(d: any) => ({ title: d.label, rows: [{ label: "eff. rank", value: d.x.toFixed(2) }, { label: yl, value: d.y.toFixed(3) }] })} />
      <div className="small mono">r = {c.pearson_r?.toFixed(2) ?? NA} · ρ = {c.spearman_rho?.toFixed(2) ?? NA} · n = {c.n} method means</div>
    </div>
  );
  return (
    <div className="grid" style={{ gap: 16 }}>
      <Card pad>
        <CardHead title="Effective feature rank by method" sub="stable rank ‖F‖²_F / ‖F‖²₂ of penultimate features" right={<GroupSelect value={group} onChange={setGroup} />} />
        <BarSci data={arms.filter((r: RecordSummary) => m(r, "final_eff_rank") != null).map((r: RecordSummary) => ({ name: r.arm_label, s: (r.arm_label ?? "").slice(0, 12), v: m(r, "final_eff_rank"), e: sd(r, "final_eff_rank"), f: r.family }))}
                xKey="s" yKey="v" stdKey="e" colorFor={(d) => fam(d.f)} valueFmt={(v) => v.toFixed(2)} metricLabel="Eff. rank" />
        <div className="grid-2col mt3">{sc(cAcc, "mean acc")}{sc(cDec, "decay")}</div>
        <Observation label="Scope" tone="amber">Effective rank is a diagnostic. In the width-16 regime it tracks accuracy (paper r = 0.89) but barely tracks decay (r = −0.21); these associations across method means do not identify the cause of plasticity loss.</Observation>
        <SourceNote>{arms[0]?.artifact} · <span className="mono">POST /analysis/correlation</span></SourceNote>
      </Card>
      <StudioCurves studioGroups={studioGroups} metric="eff_rank" label="Effective rank" fmt={(v) => v.toFixed(2)} />
    </div>
  );
}
