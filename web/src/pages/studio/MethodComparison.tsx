import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { PageHeader } from "../../components/PageHeader";
import { Card, CardHead, Observation } from "../../components/ui";
import { BarSci } from "../../components/charts";
import {
  BackendGate, MeanStd, METRIC_LABEL, PALETTE, SeriesChart, SourceBadge, SourceNote, fmtMetric,
} from "../../components/studio/StudioUI";
import { api, apiText, archLabel, DATASET_LABEL, downloadText, RecordSummary, useApi } from "../../lib/api";

const KEYS = ["mean_acc", "decay", "final_weight_norm", "final_dormant", "final_eff_rank", "matched_max_dev"];

export function MethodComparison() {
  return (
    <>
      <PageHeader title="Method Comparison" subtitle="Side-by-side metrics, progression curves and paired-seed differences — computed only where the design supports them." />
      <BackendGate feature="method comparison"><Compare /></BackendGate>
    </>
  );
}

function Compare() {
  const [params, setParams] = useSearchParams();
  const ids = useMemo(() => (params.get("ids") ?? "").split(",").filter(Boolean), [params]);
  const { data: all } = useApi<RecordSummary[]>("/experiments?include_phase1=false");
  const [pick, setPick] = useState("");
  const [res, setRes] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);

  const options = useMemo(() => {
    const recs = (all ?? []).filter((r) => r.category !== "X" && r.metrics && Object.keys(r.metrics).length);
    const groups = new Map<string, RecordSummary[]>();
    recs.filter((r) => r.source === "studio" && r.group).forEach((r) => groups.set(r.group!, [...(groups.get(r.group!) ?? []), r]));
    const gopts = [...groups.entries()].filter(([, m]) => m.length > 1)
      .map(([g, m]) => ({ id: `group:${g}`, name: `${m[0].name} — group of ${m.length} seeds` }));
    return [...gopts, ...recs.map((r) => ({ id: r.id, name: r.name }))];
  }, [all]);

  useEffect(() => {
    if (!ids.length) { setRes(null); return; }
    let alive = true;
    api("/experiments/compare", { method: "POST", json: { ids } })
      .then((r) => { if (alive) { setRes(r); setErr(null); } })
      .catch((e) => alive && setErr(String(e.message ?? e)));
    return () => { alive = false; };
  }, [ids.join(",")]); // eslint-disable-line react-hooks/exhaustive-deps

  const setIds = (next: string[]) => setParams(next.length ? { ids: next.join(",") } : {});
  const pal = PALETTE();

  const exportReport = async (fmt: "markdown" | "json") => {
    if (fmt === "json") return downloadText("comparison.json", JSON.stringify(res, null, 2), "application/json");
    const t = await apiText("/experiments/compare", { method: "POST", json: { ids, format: "markdown" } });
    downloadText("comparison_report.md", t, "text/markdown");
  };

  const curveSets = res ? Object.entries<any>(res.curves).filter(([, c]) => c && c.acc) : [];

  return (
    <>
      <Card pad className="mb4">
        <div className="filter-bar">
          <select className="select" value={pick} onChange={(e) => setPick(e.target.value)} style={{ flex: 1 }} aria-label="Add experiment">
            <option value="">Add an experiment or Studio seed group…</option>
            {options.filter((o) => !ids.includes(o.id)).map((o) => <option key={o.id} value={o.id}>{o.name} · {o.id}</option>)}
          </select>
          <button className="btn" disabled={!pick || ids.length >= 20} onClick={() => { setIds([...ids, pick]); setPick(""); }}>Add</button>
          <button className="btn ghost" disabled={!ids.length} onClick={() => setIds([])}>Clear</button>
          <button className="btn ghost" disabled={!res} onClick={() => exportReport("markdown")}>Export Markdown</button>
          <button className="btn ghost" disabled={!res} onClick={() => exportReport("json")}>Export JSON</button>
        </div>
        <div className="chips mt3">
          {ids.map((i) => <span key={i} className="chip on" onClick={() => setIds(ids.filter((x) => x !== i))} title="remove">{i} ✕</span>)}
        </div>
      </Card>
      {err && <div className="error-box mb4"><div className="eb-title">Comparison failed</div><pre>{err}</pre></div>}
      {!ids.length && <div className="studio-note">Select experiments here or tick rows in the Results Explorer.</div>}

      {res && (
        <>
          <Card className="mb4">
            <CardHead title="Side-by-side metrics" sub="mean ± population std across seeds; N/A where a metric does not apply" />
            <div className="tbl-wrap"><table className="tbl">
              <thead><tr><th>Experiment</th><th>Setup</th><th className="num">Seeds</th>
                {KEYS.map((k) => <th key={k} className="num">{METRIC_LABEL[k]}</th>)}
                <th className="num">Duration</th><th className="num">GPU-h</th></tr></thead>
              <tbody>{res.table.map((r: any) => (
                <tr key={r.id}>
                  <td><div className="result-method">{r.name}</div><div className="row aic" style={{ gap: 6 }}><SourceBadge source={r.source} /><span className="small mono muted">{r.id}</span></div></td>
                  <td className="small">{DATASET_LABEL[r.dataset] ?? r.dataset} · {archLabel(r.architecture)}</td>
                  <td className="num">{r.n_seeds}</td>
                  {KEYS.map((k) => <td key={k} className="num">{k === "matched_max_dev" && r.matched_step !== "applicable"
                    ? <span className="na">{r.matched_step === "reference" ? "ref" : "N/A"}</span> : <MeanStd m={r.metrics[k]} k={k} />}</td>)}
                  <td className="num">{r.duration_s != null ? `${r.duration_s.toFixed(1)} s` : <span className="na">not recorded</span>}</td>
                  <td className="num">{r.gpu_hours != null ? r.gpu_hours.toFixed(4) : <span className="na">—</span>}</td>
                </tr>))}</tbody>
            </table></div>
          </Card>

          <div className="grid-2col mb4">
            <Card pad><CardHead title="Mean accuracy" sub="error bars: ±1 std across seeds" />
              <BarSci data={res.table.map((r: any) => ({ name: r.name, short: (r.method_label ?? r.name).slice(0, 14), v: r.metrics.mean_acc?.mean, s: r.metrics.mean_acc?.std }))}
                      xKey="short" yKey="v" stdKey="s" valueFmt={(v) => v.toFixed(3)} metricLabel="Mean acc" /></Card>
            <Card pad><CardHead title="Plasticity decay (peak − tail)" sub="lower is better" />
              <BarSci data={res.table.map((r: any) => ({ name: r.name, short: (r.method_label ?? r.name).slice(0, 14), v: r.metrics.decay?.mean, s: r.metrics.decay?.std }))}
                      xKey="short" yKey="v" stdKey="s" color={pal[1]} valueFmt={(v) => v.toFixed(3)} metricLabel="Decay" /></Card>
          </div>

          <Card pad className="mb4">
            <CardHead title="Progression" sub="per-task curves exist only for Studio runs (existing artifacts did not persist them)" />
            {curveSets.length ? (
              <div className="grid-2col">
                <div><div className="small muted mb2">Task accuracy</div>
                  <SeriesChart series={curveSets.map(([id, c], i) => ({ key: `a${i}`, name: id, color: pal[i % pal.length], mean: c.acc.mean, std: c.acc.std }))} yFmt={(v) => v.toFixed(2)} /></div>
                <div><div className="small muted mb2">Global ‖θ‖</div>
                  <SeriesChart series={curveSets.filter(([, c]) => c.weight_norm).map(([id, c], i) => ({ key: `n${i}`, name: id, color: pal[i % pal.length], mean: c.weight_norm.mean, std: c.weight_norm.std }))} yFmt={(v) => v.toFixed(1)} /></div>
              </div>
            ) : <div className="studio-note">None of the selected records has per-task data. Run them through the Studio to obtain curves.</div>}
          </Card>

          <Card pad className="mb4">
            <CardHead title="Paired-seed comparisons" sub="only for identical protocols (all settings except the update rule) with shared seeds" />
            {res.pairs.length === 0 && <div className="studio-note">Select at least two records.</div>}
            {res.pairs.map((p: any) => (
              <div key={p.a + p.b} className="pair-block">
                <div className="row between aic wrap"><b className="mono small">{p.a} vs {p.b}</b>
                  <span className={`badge ${p.valid ? "green" : "gray"}`}>{p.valid ? "pairing valid" : "not paired"}</span></div>
                {!p.valid && <div className="small muted mt2">{p.reasons.join("; ")}</div>}
                {p.valid && (
                  <div className="tbl-wrap mt2"><table className="tbl">
                    <thead><tr><th>Metric</th><th className="num">n seeds</th><th className="num">Mean diff (a − b)</th><th className="num">a better on</th>
                      <th className="num">Sign test p</th><th className="num">Paired t p</th><th>Note</th></tr></thead>
                    <tbody>{Object.entries<any>(p.results).filter(([, x]) => x.ok).map(([k, x]) => (
                      <tr key={k}><td>{METRIC_LABEL[k] ?? k}</td><td className="num">{x.n}</td>
                        <td className="num mono">{(x.descriptive.mean_difference >= 0 ? "+" : "") + fmtMetric(x.descriptive.mean_difference, k === "final_weight_norm" ? k : "decay")}</td>
                        <td className="num">{x.descriptive.a_better_on}/{x.descriptive.of}</td>
                        <td className="num mono">{x.tests.sign_test_p_two_sided?.toFixed(4) ?? "—"}</td>
                        <td className="num mono">{x.tests.paired_t_p_two_sided?.toFixed(4) ?? "—"}</td>
                        <td className="small muted">{x.interpretation_note}</td></tr>))}</tbody>
                  </table></div>
                )}
              </div>
            ))}
          </Card>
          <Observation label="Interpretation discipline" tone="amber">
            {res.notes.join(" ")} A higher mean alone is not a claim of superiority; with five seeds the exact sign test cannot
            go below p = 0.0625.
          </Observation>
          <SourceNote><span className="mono">POST /experiments/compare</span> over the selected registry records</SourceNote>
        </>
      )}
    </>
  );
}
