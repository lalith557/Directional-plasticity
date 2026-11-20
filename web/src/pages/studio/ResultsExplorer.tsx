import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { PageHeader } from "../../components/PageHeader";
import { Card } from "../../components/ui";
import { MetricCard } from "../../components/MetricCard";
import { ResultTable, Column } from "../../components/ResultTable";
import {
  BackendGate, MeanStd, SourceBadge, SourceNote, StatusBadge, ValidationBadge,
} from "../../components/studio/StudioUI";
import { archLabel, DATASET_LABEL, RecordSummary, useApi } from "../../lib/api";

const CATS: [string, string][] = [
  ["", "All categories"], ["A", "A · Primary permuted MNIST"], ["B", "B · λ sweep"], ["C", "C · Radial sweep"],
  ["D", "D · Width"], ["E", "E · Task geometry"], ["F", "F · ANR ablation"], ["S", "S · Supplementary"],
  ["X", "X · Superseded / incomplete"], ["P1", "P1 · Phase 1 (different protocol)"],
];

export function ResultsExplorer() {
  return (
    <>
      <PageHeader title="Results Explorer" subtitle="Every existing artifact and every Studio run, with its source, status and validation state." />
      <BackendGate feature="the experiment archive">
        <Explorer />
      </BackendGate>
    </>
  );
}

function Explorer() {
  const nav = useNavigate();
  const { data, loading } = useApi<RecordSummary[]>("/experiments", 5000);
  const [q, setQ] = useState("");
  const [cat, setCat] = useState("");
  const [source, setSource] = useState("");
  const [status, setStatus] = useState("");
  const [sel, setSel] = useState<Set<string>>(new Set());

  const rows = useMemo(() => (data ?? []).filter((r) => {
    if (cat && r.category !== cat) return false;
    if (source && r.source !== source) return false;
    if (status && r.status !== status) return false;
    if (q) {
      const hay = `${r.id} ${r.name} ${r.method_label} ${r.experiment} ${r.paper_ref ?? ""} ${r.status}`.toLowerCase();
      if (!hay.includes(q.toLowerCase())) return false;
    }
    return true;
  }), [data, q, cat, source, status]);

  const counts = useMemo(() => {
    const d = data ?? [];
    return {
      legacy: d.filter((r) => r.source === "legacy_artifact").length,
      studio: d.filter((r) => r.source === "studio").length,
      validated: d.filter((r) => r.validation?.status?.startsWith("reproduced")).length,
      incomplete: d.filter((r) => ["incomplete", "superseded", "completed_diverged", "failed"].includes(r.status)).length,
    };
  }, [data]);

  const statuses = Array.from(new Set((data ?? []).map((r) => r.status))).sort();

  const cols: Column<RecordSummary>[] = [
    { key: "name", header: "Experiment", sortable: true, value: (r) => r.name,
      render: (r) => (
        <div>
          <div className="result-method">{r.name}</div>
          <div className="small muted mono">{r.id}</div>
        </div>
      ) },
    { key: "source", header: "Source", render: (r) => <SourceBadge source={r.source} /> },
    { key: "cat", header: "Cat.", sortable: true, value: (r) => r.category, render: (r) => <span className="mono">{r.category}</span> },
    { key: "status", header: "Status", sortable: true, value: (r) => r.status, render: (r) => <StatusBadge status={r.status} /> },
    { key: "val", header: "Validation", render: (r) => <ValidationBadge v={r.validation} /> },
    { key: "setup", header: "Dataset · arch", render: (r) => r.dataset ? <span className="small">{DATASET_LABEL[r.dataset] ?? r.dataset} · {archLabel(r.architecture)}</span> : <span className="na">—</span> },
    { key: "seeds", header: "Seeds", numeric: true, sortable: true, value: (r) => r.n_seeds ?? 0, render: (r) => r.n_seeds ?? "—" },
    { key: "acc", header: "Mean acc", numeric: true, sortable: true, value: (r) => r.metrics?.mean_acc?.mean ?? null,
      render: (r) => <MeanStd m={r.metrics?.mean_acc} k="mean_acc" /> },
    { key: "decay", header: "Decay", numeric: true, sortable: true, value: (r) => r.metrics?.decay?.mean ?? null,
      render: (r) => <MeanStd m={r.metrics?.decay} k="decay" /> },
    { key: "norm", header: "Final ‖θ‖", numeric: true, sortable: true, value: (r) => r.metrics?.final_weight_norm?.mean ?? null,
      render: (r) => <MeanStd m={r.metrics?.final_weight_norm} k="final_weight_norm" /> },
    { key: "paper", header: "Paper", render: (r) => r.paper_ref ? <span className="small">{r.paper_ref}</span> : <span className="na">—</span> },
  ];

  const toggle = (id: string) => setSel((s) => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n; });

  return (
    <>
      <div className="metrics mb4">
        {[["Existing artifact records", counts.legacy], ["Studio runs", counts.studio],
          ["Validated reproductions", counts.validated], ["Incomplete / diverged / failed", counts.incomplete]]
          .map(([l, v]) => <MetricCard key={l as string} value={v as number} label={l as string} />)}
      </div>
      <Card pad className="mb4">
        <div className="filter-bar">
          <input className="input" placeholder="Search id, name, method, paper element…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search" />
          <select className="select" value={cat} onChange={(e) => setCat(e.target.value)}>{CATS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>
          <select className="select" value={source} onChange={(e) => setSource(e.target.value)}>
            <option value="">All sources</option><option value="legacy_artifact">Existing artifacts</option><option value="studio">Studio runs</option>
          </select>
          <select className="select" value={status} onChange={(e) => setStatus(e.target.value)}>
            <option value="">All statuses</option>{statuses.map((s) => <option key={s}>{s}</option>)}
          </select>
          <button className="btn" disabled={sel.size < 2} onClick={() => nav(`/compare?ids=${encodeURIComponent([...sel].join(","))}`)}>
            Compare {sel.size || ""} selected
          </button>
        </div>
      </Card>
      {loading && !data ? <div className="studio-note">Loading registry…</div> : (
        <Card>
          <ResultTable columns={cols} rows={rows} getRowId={(r) => r.id} onRowClick={(r) => nav(`/explorer/${encodeURIComponent(r.id)}`)}
                       selectable selected={sel} onToggleSelect={toggle} initialSort={{ key: "cat", dir: "asc" }} />
          <div className="card-pad small muted">{rows.length} of {data?.length ?? 0} records</div>
        </Card>
      )}
      <SourceNote>registry API <span className="mono">GET /experiments</span> — Studio runs in <span className="mono">studio/experiments/</span>; existing artifacts read (never written) from <span className="mono">results/tables/</span></SourceNote>
    </>
  );
}
