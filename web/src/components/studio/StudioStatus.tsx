import { Link } from "react-router-dom";
import { Card, CardHead } from "../ui";
import { StatusBadge, ValidationBadge } from "./StudioUI";
import { IS_HOSTED, REPO_URL, useApi, useBackend } from "../../lib/api";

/** Overview strip: registry counts, GPU state and recent runs — all from the backend. */
export function StudioStatus() {
  const { online } = useBackend(5000);
  const { data: s } = useApi<any>(online ? "/registry/summary" : null, 5000);
  const { data: gpu } = useApi<any>(online ? "/resources/gpu" : null, 5000);
  if (IS_HOSTED)
    return (
      <Card pad className="mb4">
        <CardHead title="Experimental Studio" sub="runs locally" />
        <div className="small">
          The Studio configures, launches and monitors real PyTorch training runs on a GPU, so it runs on
          your own machine — see the{" "}
          <a className="linkish" href={`${REPO_URL}#quick-start`} target="_blank" rel="noreferrer">Quick start</a>.
          Everything below comes from the released result files.
        </div>
      </Card>
    );
  if (online === false)
    return (
      <Card pad className="mb4">
        <CardHead title="Experimental Studio" sub="backend offline" />
        <div className="small">Start the backend to configure, run and monitor experiments:{" "}
          <span className="mono">python -m uvicorn studio.api.server:app --host 127.0.0.1 --port 8765</span>.
          Values below this card come from the released artifacts and remain available.</div>
      </Card>
    );
  if (!s) return null;
  const st = s.studio.by_status ?? {};
  const cell = (v: any, l: string) => <div className="metric"><div className="metric-value">{v ?? 0}</div><div className="metric-label">{l}</div></div>;
  return (
    <Card pad className="mb4">
      <CardHead title="Experimental Studio" sub="live registry — Studio runs and existing artifact records"
                right={<div className="row" style={{ gap: 8 }}><Link className="btn sm" to="/studio">New experiment</Link><Link className="btn sm ghost" to="/explorer">Results Explorer</Link></div>} />
      <div className="metrics mb3">
        {cell(s.legacy.total + s.studio.total, "Total experiment records")}
        {cell((s.legacy.by_status?.completed ?? 0) + (st.completed ?? 0), "Completed")}
        {cell((st.running ?? 0) + (st.initializing ?? 0) + (st.queued ?? 0), "Running / queued")}
        {cell(st.failed ?? 0, "Failed runs")}
        {cell(s.studio.validated, "Validated reproductions")}
      </div>
      <div className="grid-2col">
        <div className="small">
          <b>Datasets</b> {s.datasets.join(", ")} · <b>Architectures</b> {s.architectures.join(", ")}<br />
          <b>GPU</b> {gpu?.cuda_available ? `${gpu.device} · ${gpu.utilization_pct ?? "—"}% util · ${gpu.memory_used_mb ?? "—"}/${gpu.memory_total_mb ?? "—"} MB` : "not available (CPU)"}<br />
          <b>Existing artifacts</b> {Object.entries<number>(s.legacy.by_status).map(([k, v]) => `${v} ${k}`).join(" · ")}
        </div>
        <div>
          {s.recent.length === 0 ? <div className="small muted">No Studio runs yet.</div> : (
            <div className="tbl-wrap"><table className="tbl"><tbody>{s.recent.slice(0, 4).map((r: any) => (
              <tr key={r.id}><td className="small"><Link className="linkish" to={`/explorer/${r.id}`}>{r.name}</Link></td>
                <td><StatusBadge status={r.status} /></td><td><ValidationBadge v={r.validation} /></td></tr>))}</tbody></table></div>
          )}
        </div>
      </div>
    </Card>
  );
}
