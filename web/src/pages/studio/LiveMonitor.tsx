import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { PageHeader } from "../../components/PageHeader";
import { Card, CardHead, Observation } from "../../components/ui";
import {
  BackendGate, KV, PALETTE, Progress, SeriesChart, SourceNote, StatusBadge, ValidationBadge, fmtMetric,
} from "../../components/studio/StudioUI";
import { api, archLabel, DATASET_LABEL, TERMINAL, useApi, wsUrl } from "../../lib/api";
import { NA } from "../../data/format";

export function LiveMonitor() {
  return (
    <>
      <PageHeader title="Live Monitor" subtitle="Real progress and metrics emitted by the running PyTorch process — nothing is simulated." />
      <BackendGate feature="live monitoring">
        <MonitorBody />
      </BackendGate>
    </>
  );
}

function MonitorBody() {
  const { id } = useParams();
  const nav = useNavigate();
  const { data: list } = useApi<any[]>("/experiments?source=studio&include_phase1=false", 3000);
  const { data: queue } = useApi<any>("/queue", 2000);

  useEffect(() => {
    if (id || !list) return;
    const active = queue?.active?.[0] ?? list.find((r) => !TERMINAL.has(r.status) && r.status !== "proposed")?.id;
    if (active) nav(`/monitor/${active}`, { replace: true });
  }, [id, list, queue]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="grid" style={{ gap: 16 }}>
      <Card pad>
        <div className="row between aic wrap" style={{ gap: 12 }}>
          <div className="row aic wrap" style={{ gap: 10 }}>
            <span className="field-label" style={{ margin: 0 }}>Run</span>
            <select className="select" value={id ?? ""} onChange={(e) => nav(`/monitor/${e.target.value}`)} style={{ minWidth: 340 }}>
              <option value="" disabled>{list?.length ? "select a Studio run" : "no Studio runs yet"}</option>
              {(list ?? []).map((r) => (
                <option key={r.id} value={r.id}>{r.status.padEnd(11, " ")} · {r.id} · {r.name}</option>
              ))}
            </select>
          </div>
          <div className="small muted">
            Queue: {queue ? `${queue.active.length} running · ${queue.queued.length} waiting (one run at a time on the GPU)` : "…"}
          </div>
        </div>
      </Card>
      {id ? <RunMonitor key={id} id={id} /> :
        <div className="studio-note">No active run. Start one in the <Link className="linkish" to="/studio">Experiment Studio</Link>.</div>}
    </div>
  );
}

function RunMonitor({ id }: { id: string }) {
  const [status, setStatus] = useState<string | null>(null);
  // re-read the record while the run is active (device, errors and results are written by the runner)
  const { data: rec, reload } = useApi<any>(`/experiments/${id}`, status !== null && !TERMINAL.has(status) ? 3000 : undefined);
  const [events, setEvents] = useState<any[]>([]);
  const [progress, setProgress] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [transport, setTransport] = useState<"websocket" | "polling">("websocket");
  const [confirmCancel, setConfirmCancel] = useState(false);
  const [cancelMsg, setCancelMsg] = useState<string | null>(null);
  const active = status !== null && !TERMINAL.has(status);
  const { data: gpu } = useApi<any>(active ? "/resources/gpu" : null, 2000);
  const { data: logs } = useApi<any>(`/experiments/${id}/logs?tail=200`, active ? 2000 : undefined);
  const logRef = useRef<HTMLPreElement>(null);

  // WebSocket stream with polling fallback
  useEffect(() => {
    let closed = false, poll: number | undefined;
    const startPolling = () => {
      setTransport("polling");
      const tick = async () => {
        try {
          const [st, m] = await Promise.all([api(`/experiments/${id}/status`), api(`/experiments/${id}/metrics`)]);
          if (closed) return;
          setStatus(st.status); setError(st.error); setProgress(st.progress); setEvents(m.events ?? []);
          if (TERMINAL.has(st.status)) { window.clearInterval(poll); reload(); }
        } catch { /* backend gate handles offline */ }
      };
      tick(); poll = window.setInterval(tick, 1500);
    };
    let ws: WebSocket | null = null;
    try {
      ws = new WebSocket(wsUrl(`/ws/experiments/${id}`));
      ws.onmessage = (m) => {
        const msg = JSON.parse(m.data);
        if (msg.type === "events") setEvents((e) => (msg.offset === e.length ? [...e, ...msg.events] : e));
        else if (msg.type === "progress") setProgress(msg);
        else if (msg.type === "status") { setStatus(msg.status); setError(msg.error); }
        else if (msg.type === "end") { setStatus(msg.status); reload(); }
        else if (msg.type === "error") setError(msg.message);
      };
      ws.onerror = () => { if (!closed) { ws?.close(); startPolling(); } };
    } catch { startPolling(); }
    return () => { closed = true; ws?.close(); if (poll) window.clearInterval(poll); };
  }, [id]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { logRef.current?.scrollTo(0, logRef.current.scrollHeight); }, [logs]);

  if (!rec) return <div className="studio-note">Loading run…</div>;
  const cfg = rec.config;
  const last = events[events.length - 1];
  const nT = cfg.dataset.n_tasks, spt = cfg.optimization.steps_per_task;
  const pal = PALETTE();
  const col = (k: string) => events.map((e) => e[k] ?? null);
  const directional = rec.matched_step === "applicable";

  const doCancel = async () => {
    const r = await api(`/experiments/${id}/cancel`, { method: "POST" });
    setCancelMsg(r.result === "cancelling" ? "Cancellation requested — the run stops at its next step check." : `Run ${r.result}.`);
    setConfirmCancel(false);
  };

  return (
    <>
      <div className="grid-21">
        <Card pad>
          <div className="row between aic wrap mb3" style={{ gap: 10 }}>
            <div>
              <div className="card-title">{rec.name}</div>
              <div className="small muted mono">{id}</div>
            </div>
            <div className="row aic" style={{ gap: 8 }}>
              <StatusBadge status={status ?? rec.status} />
              <span className="small muted">via {transport}</span>
            </div>
          </div>
          <Progress label="Tasks completed" value={events.length} max={nT} />
          <div className="mt3">
            <Progress label={`Steps in current task${progress ? ` (task ${progress.task + 1})` : ""}`}
                      value={progress && events.length < nT ? progress.step : events.length >= nT ? spt : 0} max={spt} />
          </div>
          <div className="mt3">
            <Progress label="Total SGD steps" value={progress ? Math.min(progress.global_step, nT * spt) : 0} max={nT * spt} />
          </div>
          <div className="mt4">
            <KV rows={[
              ["Configuration", `${rec.method_label} · ${archLabel(cfg.architecture)} · ${DATASET_LABEL[cfg.dataset.name]}`],
              ["Seed", cfg.seed],
              ["Elapsed", progress ? `${progress.elapsed_s.toFixed(1)} s` : NA],
              ["Device", rec.device_name ?? (active ? "initializing…" : NA)],
              ["Run peak GPU memory", progress?.gpu_mem_peak_mb != null ? `${progress.gpu_mem_peak_mb} MB (torch, this process)` : NA],
            ]} />
          </div>
          {(error || rec.error) && <div className="msg err mt3">{error || rec.error}</div>}
          {cancelMsg && <div className="studio-note mt3">{cancelMsg}</div>}
          {active && (
            <div className="mt4">
              {!confirmCancel ? (
                <button className="btn ghost" onClick={() => setConfirmCancel(true)}>Cancel run…</button>
              ) : (
                <div className="confirm-inline">
                  <span className="small">Stop this run? Completed task events are kept; the run cannot be resumed (re-run it as a new experiment).</span>
                  <button className="btn sm" onClick={doCancel}>Yes, cancel</button>
                  <button className="btn sm ghost" onClick={() => setConfirmCancel(false)}>Keep running</button>
                </div>
              )}
            </div>
          )}
        </Card>

        <Card pad>
          <CardHead title="GPU (device-wide)" sub="NVML — includes other processes" />
          {active && gpu ? (
            <KV rows={[
              ["Device", gpu.device ?? "CPU only"],
              ["Utilization", gpu.utilization_pct != null ? `${gpu.utilization_pct}%` : NA],
              ["Memory used", gpu.memory_used_mb != null ? `${gpu.memory_used_mb} / ${gpu.memory_total_mb} MB` : NA],
              ["Temperature", gpu.temperature_c != null ? `${gpu.temperature_c} °C` : NA],
              ["Power", gpu.power_w != null ? `${gpu.power_w} W` : NA],
            ]} />
          ) : <div className="studio-note">Polled only while the run is active.</div>}
          <div className="section-label mt4">Latest task {last ? `(${last.task + 1} / ${nT})` : ""}</div>
          {last ? (
            <KV rows={[
              ["Task accuracy", fmtMetric(last.acc, "mean_acc")],
              ["Global ‖θ‖", fmtMetric(last.weight_norm, "final_weight_norm")],
              ["Decay (running estimate)", fmtMetric(last.decay_running_estimate, "decay")],
              ["Dormant fraction", fmtMetric(last.dormant, "final_dormant")],
              ["Effective rank", fmtMetric(last.eff_rank, "final_eff_rank")],
              ["Mean step ‖Δθ'‖", last.step_norm_mean?.toExponential(3) ?? NA],
              ["Matched-step deviation", directional ? fmtMetric(last.matched_dev_max, "matched_max_dev") : "N/A (rule not step-matched)"],
              ["Radial / tangential", `${last.radial_component_mean?.toExponential(2) ?? NA} / ${last.tangential_component_mean?.toExponential(2) ?? NA}`],
            ]} />
          ) : <div className="studio-note">Waiting for the first task to finish.</div>}
        </Card>
      </div>

      <div className="grid-2col">
        <Card pad><CardHead title="Task accuracy a_t" />
          <SeriesChart series={[{ key: "acc", name: "a_t", color: pal[0], mean: col("acc") }]} yFmt={(v) => v.toFixed(2)} /></Card>
        <Card pad><CardHead title="Global weight norm ‖θ‖" sub={rec.architecture?.name === "mlp" ? "backbone, as in the paper" : "all parameters"} />
          <SeriesChart series={[{ key: "n", name: "‖θ‖", color: pal[4], mean: col("weight_norm") }]} yFmt={(v) => v.toFixed(1)} /></Card>
        <Card pad><CardHead title="Dormant fraction and effective rank" />
          <SeriesChart series={[{ key: "d", name: "dormant", color: pal[3], mean: col("dormant") },
                                { key: "r", name: "eff. rank / 10", color: pal[2], mean: events.map((e) => (e.eff_rank != null ? e.eff_rank / 10 : null)) }]}
                       yFmt={(v) => v.toFixed(2)} /></Card>
        <Card pad><CardHead title="Update geometry per task" sub="mean radial and tangential components of the applied step" />
          <SeriesChart series={[{ key: "rad", name: "radial ‖Δθ'‖cos", color: pal[1], mean: col("radial_component_mean") },
                                { key: "tan", name: "tangential", color: pal[0], mean: col("tangential_component_mean") }]}
                       yFmt={(v) => v.toExponential(1)} /></Card>
      </div>

      <Card pad>
        <CardHead title="Execution log" sub="stdout / stderr of the runner process" />
        <pre className="log-view" ref={logRef}>{(logs?.lines ?? []).join("\n") || "No log output yet."}</pre>
      </Card>

      {TERMINAL.has(status ?? rec.status) && rec.result && (
        <Card pad>
          <CardHead title="Final result" right={<Link className="linkish" to={`/explorer/${id}`}>Open in Results Explorer →</Link>} />
          <KV rows={[
            ["Mean accuracy", fmtMetric(rec.result.summary.mean_acc, "mean_acc")],
            ["Plasticity decay (peak − tail)", fmtMetric(rec.result.summary.decay, "decay")],
            ["Final ‖θ‖", fmtMetric(rec.result.summary.final_weight_norm, "final_weight_norm")],
            ["Validation", <ValidationBadge v={rec.result.validation} />],
            ["Flags", rec.result.flags?.length ? rec.result.flags.join("; ") : "none"],
          ]} />
        </Card>
      )}
      <Observation label="What is shown" tone="neutral">
        Every value above is read from files the runner process writes (<span className="mono">events.jsonl</span>,{" "}
        <span className="mono">progress.json</span>, <span className="mono">log.txt</span>). Radial and tangential components are
        computed from the per-step cosine cos(Δθ', θ) and step length the training loop already logs. Pausing is not
        offered because the training loop cannot pause.
      </Observation>
      <SourceNote><span className="mono">studio/experiments/{id}/</span> · WebSocket <span className="mono">/ws/experiments/{id}</span></SourceNote>
    </>
  );
}
