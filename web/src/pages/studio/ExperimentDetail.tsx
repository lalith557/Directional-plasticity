import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { PageHeader } from "../../components/PageHeader";
import { Card, CardHead, Observation } from "../../components/ui";
import {
  BackendGate, KV, MeanStd, METRIC_LABEL, PALETTE, SeriesChart, SourceBadge, SourceNote, StatusBadge,
  ValidationBadge, fmtMetric,
} from "../../components/studio/StudioUI";
import { api, ApiError, apiText, archLabel, DATASET_LABEL, downloadText, useApi } from "../../lib/api";
import { NA } from "../../data/format";

export function ExperimentDetail() {
  const { id = "" } = useParams();
  return (
    <BackendGate feature="experiment details">
      <Detail id={decodeURIComponent(id)} />
    </BackendGate>
  );
}

const METRICS = ["mean_acc", "decay", "final_weight_norm", "final_dormant", "final_eff_rank", "matched_max_dev",
                 "peak_acc", "end_acc"];

function Detail({ id }: { id: string }) {
  const nav = useNavigate();
  const { data: r, error } = useApi<any>(`/experiments/${encodeURIComponent(id)}`, 4000);
  const { data: logs } = useApi<any>(r?.source === "studio" ? `/experiments/${id}/logs?tail=120` : null);
  const [repro, setRepro] = useState<null | "ask" | "busy">(null);
  const [reproSeeds, setReproSeeds] = useState<"first" | "all">("first");
  const [msg, setMsg] = useState<string | null>(null);

  if (error) return <div className="error-box"><div className="eb-title">Not found</div><pre>{String(error.message)}</pre></div>;
  if (!r) return <div className="studio-note">Loading…</div>;
  const cfg = r.config;
  const summary = r.result?.summary;
  const pt = summary?.per_task;
  const pal = PALETTE();

  const download = async (fmt: "json" | "yaml") => {
    const t = await apiText(`/experiments/${encodeURIComponent(id)}/config?format=${fmt}`);
    downloadText(`${id.replace(":", "_")}.config.${fmt}`, t);
  };
  const downloadResults = async () => {
    const t = await apiText(`/experiments/${encodeURIComponent(id)}/results`);
    downloadText(`${id.replace(":", "_")}.results.json`, t, "application/json");
  };
  const reproduce = async (confirm = false) => {
    setRepro("busy"); setMsg(null);
    try {
      const seeds = r.seeds?.length ? (reproSeeds === "all" ? r.seeds : [r.seeds[0]]) : undefined;
      const res = await api("/experiments/run", { method: "POST", json: { reproduce_of: id, seeds, confirm_long: confirm } });
      nav(`/monitor/${res.queued[0]}`);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        setMsg(`Estimated ${e.detail.estimate?.seconds ?? "unknown"} s — press "Confirm long run" to proceed.`);
        setRepro("ask");
      } else { setMsg(String(e)); setRepro(null); }
    }
  };

  return (
    <>
      <PageHeader title={r.name} subtitle={r.id}
        badge={<div className="row" style={{ gap: 6 }}><SourceBadge source={r.source} /><StatusBadge status={r.status} /><ValidationBadge v={r.validation} /></div>} />

      <div className="row wrap mb4" style={{ gap: 8 }}>
        <Link className="btn ghost sm" to="/explorer">← Results Explorer</Link>
        <Link className="btn ghost sm" to={`/compare?ids=${encodeURIComponent(id)}`}>Compare with…</Link>
        {cfg && <button className="btn ghost sm" onClick={() => download("json")}>Download config (JSON)</button>}
        {cfg && <button className="btn ghost sm" onClick={() => download("yaml")}>Download config (YAML)</button>}
        <button className="btn ghost sm" onClick={downloadResults}>Download result summary</button>
        {r.source === "studio" && <Link className="btn ghost sm" to={`/monitor/${id}`}>Live monitor</Link>}
        {cfg && r.status !== "incomplete" && !repro && <button className="btn sm" onClick={() => setRepro("ask")}>Reproduce…</button>}
      </div>

      {repro && (
        <Card pad className="confirm-card mb4">
          <CardHead title="Reproduce with the original configuration" />
          <p className="small">A new experiment ID is created; this record is never overwritten.
            {r.source === "legacy_artifact" && " The configuration is reconstructed from the experiment script; the Studio will compare the new result with this artifact's per-seed values where they exist."}</p>
          {r.seeds?.length > 1 && (
            <div className="row aic mb3" style={{ gap: 14 }}>
              <label className="small"><input type="radio" checked={reproSeeds === "first"} onChange={() => setReproSeeds("first")} /> first seed only ({r.seeds[0]})</label>
              <label className="small"><input type="radio" checked={reproSeeds === "all"} onChange={() => setReproSeeds("all")} /> all {r.seeds.length} seeds</label>
            </div>
          )}
          <div className="row" style={{ gap: 8 }}>
            <button className="btn" disabled={repro === "busy"} onClick={() => reproduce(false)}>Queue reproduction</button>
            {msg && <button className="btn" onClick={() => reproduce(true)}>Confirm long run</button>}
            <button className="btn ghost" onClick={() => { setRepro(null); setMsg(null); }}>Cancel</button>
          </div>
          {msg && <div className="studio-note mt3">{msg}</div>}
        </Card>
      )}

      <div className="grid-2col mb4">
        <Card pad>
          <CardHead title="Identity and provenance" />
          <KV rows={[
            ["Category", r.category_label ?? r.category],
            ["Research hypothesis", r.hypothesis ?? (r.source === "legacy_artifact" ? "see paper / decision.md" : NA)],
            ["Paper element", r.paper_ref ?? "not in the paper"],
            ["Artifact / directory", <span className="mono small">{r.artifact ?? NA}</span>],
            ["Experiment script", r.script ? <span className="mono small">{r.script}</span> : (r.source === "studio" ? "studio runner (run_arm / run_arm_cnn)" : NA)],
            ["Created", r.created_at ?? NA],
            ["Started / finished", r.source === "studio" ? `${r.started_at ?? "—"} / ${r.finished_at ?? "—"}` : "not recorded in artifact"],
            ["Duration", r.duration_s != null ? `${r.duration_s.toFixed(1)} s` : NA],
            ["Reproduction of", r.reproduce_of ? <Link className="linkish" to={`/explorer/${encodeURIComponent(r.reproduce_of)}`}>{r.reproduce_of}</Link> : "—"],
            ["Error", r.error ?? "—"],
          ]} />
          {r.notes?.length > 0 && <div className="mt3">{r.notes.map((n: string) => <div key={n} className="msg warn">{n}</div>)}</div>}
          {r.verdict && <div className="msg mt3">Recorded verdict: <span className="mono">{String(r.verdict)}</span></div>}
        </Card>
        <Card pad>
          <CardHead title="Configuration" sub={r.source === "legacy_artifact" ? "reconstructed from the script constants" : "immutable snapshot"} />
          {cfg ? (
            <KV rows={[
              ["Dataset", `${DATASET_LABEL[cfg.dataset.name]} · ${cfg.dataset.n_tasks} tasks · subset ${cfg.dataset.train_subset}${cfg.dataset.test_subset ? ` · test ${cfg.dataset.test_subset}` : ""}`],
              ["Architecture", archLabel(cfg.architecture)],
              ["Update rule", `${r.method_label}${Object.keys(cfg.method.params).length ? " · " + Object.entries(cfg.method.params).map(([k, v]) => `${k}=${v}`).join(", ") : ""}`],
              ["Optimization", `SGD η₀=${cfg.optimization.lr} · batch ${cfg.optimization.batch} · ${cfg.optimization.steps_per_task} steps/task`],
              ["Diagnostic probe", `${cfg.evaluation.diag_probe_size} test images`],
              ["Seeds", (r.seeds ?? []).join(", ") || NA],
              ["Config hash", r.config_hash ? <span className="mono small">{r.config_hash}</span> : "— (legacy record: per-seed hashes are computed when reproduced)"],
              ["Snapshot check", r.snapshot_check ? (r.snapshot_check.ok ? "hash matches" : <span className="text-red">{r.snapshot_check.reason}</span>) : NA],
              ["Checkpoint policy", cfg.checkpoint_policy],
            ]} />
          ) : <div className="studio-note">No reproducible configuration for this record.</div>}
        </Card>
      </div>

      <Card pad className="mb4">
        <CardHead title="Metrics" sub={r.n_seeds > 1 ? `mean ± population std over ${r.n_seeds} seeds` : "single seed"} />
        <div className="tbl-wrap"><table className="tbl">
          <thead><tr>{METRICS.filter((k) => r.metrics?.[k]).map((k) => <th key={k} className="num">{METRIC_LABEL[k] ?? k}</th>)}
            {Object.keys(r.metrics ?? {}).filter((k) => !METRICS.includes(k)).map((k) => <th key={k} className="num">{k}</th>)}</tr></thead>
          <tbody><tr>{METRICS.filter((k) => r.metrics?.[k]).map((k) => (
            <td key={k} className="num">{k === "matched_max_dev" && r.matched_step !== "applicable" ? <span className="na">N/A</span> : <MeanStd m={r.metrics[k]} k={k} />}</td>))}
            {Object.keys(r.metrics ?? {}).filter((k) => !METRICS.includes(k)).map((k) => <td key={k} className="num"><MeanStd m={r.metrics[k]} k={k} /></td>)}</tr></tbody>
        </table></div>
        {r.per_seed?.length > 0 && (
          <>
            <div className="section-label mt4">Per-seed values</div>
            <div className="tbl-wrap"><table className="tbl">
              <thead><tr><th>Seed</th>{Object.keys(r.per_seed[0]).filter((k) => k !== "seed").map((k) => <th key={k} className="num">{METRIC_LABEL[k] ?? k}</th>)}</tr></thead>
              <tbody>{r.per_seed.map((row: any) => (
                <tr key={row.seed}><td className="mono">{row.seed}</td>
                  {Object.keys(r.per_seed[0]).filter((k) => k !== "seed").map((k) => <td key={k} className="num mono">{fmtMetric(row[k], k)}</td>)}</tr>))}</tbody>
            </table></div>
          </>
        )}
        {r.layer_norms && (
          <>
            <div className="section-label mt4">Layer-wise norms (e26)</div>
            <KV rows={r.layer_norms.labels.map((l: string, i: number) => [l, `${r.layer_norms.init[i].toFixed(2)} → ${r.layer_norms.final?.[i]?.toFixed(2) ?? NA}`])} />
          </>
        )}
      </Card>

      <Card pad className="mb4">
        <CardHead title="Per-task trajectories" />
        {pt ? (
          <div className="grid-2col">
            <div><div className="small muted mb2">Task accuracy a_t</div><SeriesChart series={[{ key: "a", name: "a_t", color: pal[0], mean: pt.acc }]} yFmt={(v) => v.toFixed(2)} /></div>
            <div><div className="small muted mb2">Global ‖θ‖ ({summary.weight_norm_definition})</div><SeriesChart series={[{ key: "n", name: "‖θ‖", color: pal[4], mean: pt.weight_norm }]} yFmt={(v) => v.toFixed(1)} /></div>
            <div><div className="small muted mb2">Dormant fraction</div><SeriesChart series={[{ key: "d", name: "dormant", color: pal[3], mean: pt.dormant }]} yFmt={(v) => v.toFixed(2)} /></div>
            <div><div className="small muted mb2">Radial vs tangential step component</div>
              <SeriesChart series={[{ key: "r", name: "radial", color: pal[1], mean: pt.radial_component_mean },
                                    { key: "t", name: "tangential", color: pal[0], mean: pt.tangential_component_mean }]} yFmt={(v) => v.toExponential(1)} /></div>
          </div>
        ) : <div className="studio-note">{r.per_task_note ?? "Per-task values become available when the run completes."}</div>}
      </Card>

      {r.source === "studio" && (
        <div className="grid-2col mb4">
          <Card pad>
            <CardHead title="Validation" />
            <KV rows={[
              ["Status", <ValidationBadge v={r.validation} />],
              ["Compared against", r.validation?.against ?? "—"],
              ["Max |difference|", r.validation?.max_abs_diff != null ? r.validation.max_abs_diff.toExponential(2) : "—"],
              ["Detail", r.validation?.detail ?? (r.validation?.diffs ? Object.entries(r.validation.diffs).map(([k, v]: any) => `${k}: ${v.toExponential(1)}`).join(" · ") : "—")],
              ["Sanity flags", r.flags?.length ? r.flags.join("; ") : "none"],
            ]} />
          </Card>
          <Card pad>
            <CardHead title="Environment and checkpoints" />
            <KV rows={[
              ["Device", r.device_name ?? NA],
              ["torch / CUDA", r.environment ? `${r.environment.environment.torch} / ${r.environment.environment.cuda_runtime ?? "—"}` : NA],
              ["cuDNN deterministic", r.environment ? String(r.environment.environment.cudnn_deterministic) : NA],
              ["src/plasticity_stress.py", r.environment ? <span className="mono small">{r.environment.code_sha256["src/plasticity_stress.py"]?.slice(0, 16)}…</span> : NA],
              ["MNIST train images", r.environment ? <span className="mono small">{r.environment.dataset_sha256.train_images?.sha256?.slice(0, 16)}…</span> : NA],
              ["Checkpoints", r.checkpoints?.length ? r.checkpoints.map((c: any) => `${c.name} (${(c.bytes / 1024).toFixed(0)} KB, sha256 ${c.sha256.slice(0, 10)}…)`).join("; ") : "none (policy: " + cfg.checkpoint_policy + ")"],
            ]} />
          </Card>
        </div>
      )}
      {r.source === "legacy_artifact" && r.hardware && (
        <Card pad className="mb4"><CardHead title="Hardware recorded in the artifact" />
          <KV rows={Object.entries(r.hardware).map(([k, v]) => [k, String(v)])} /></Card>
      )}
      {r.source === "studio" && (
        <Card pad className="mb4"><CardHead title="Log (last 120 lines)" />
          <pre className="log-view">{(logs?.lines ?? []).join("\n") || "No log."}</pre></Card>
      )}
      <Observation label="Traceability" tone="neutral">
        {r.source === "legacy_artifact"
          ? <>Values are copied from <span className="mono">{r.artifact}</span>. Non-finite values are shown as “Not available”, never 0.</>
          : <>Values come from <span className="mono">{r.artifact}</span> (result.json, events.jsonl) written by the runner process.</>}
      </Observation>
      <SourceNote><span className="mono">GET /experiments/{id}</span></SourceNote>
    </>
  );
}
