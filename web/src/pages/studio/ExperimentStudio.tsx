import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { PageHeader } from "../../components/PageHeader";
import { Card, CardHead, Observation } from "../../components/ui";
import { BackendGate, KV, SourceNote } from "../../components/studio/StudioUI";
import { api, ApiError, downloadText, useApi } from "../../lib/api";

type Spec = { type: "int" | "float" | "choice"; default: any; min?: number; max?: number; choices?: string[]; label?: string };
const defaults = (spec: Record<string, Spec>) =>
  Object.fromEntries(Object.entries(spec).map(([k, v]) => [k, v.default]));

function ParamInput({ name, spec, value, onChange }: { name: string; spec: Spec; value: any; onChange: (v: any) => void }) {
  if (spec.type === "choice")
    return (
      <label className="field">
        <span className="field-label">{spec.label ?? name}</span>
        <select className="select" value={value} onChange={(e) => onChange(e.target.value)}>
          {spec.choices!.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
      </label>
    );
  return (
    <label className="field">
      <span className="field-label">{spec.label ?? name}</span>
      <input
        className="input mono"
        type="number"
        value={value ?? ""}
        min={spec.min}
        max={spec.max}
        step={spec.type === "int" ? 1 : "any"}
        onChange={(e) => onChange(e.target.value === "" ? "" : spec.type === "int" ? parseInt(e.target.value, 10) : parseFloat(e.target.value))}
      />
      <span className="field-hint">{spec.min !== undefined ? `range ${spec.min} – ${spec.max}` : ""} · default {String(spec.default)}</span>
    </label>
  );
}

function Block({ title, spec, values, onChange }: { title: string; spec: Record<string, Spec>; values: any; onChange: (k: string, v: any) => void }) {
  const keys = Object.keys(spec);
  if (!keys.length) return <div className="studio-note">{title}: no tunable parameters in the implementation.</div>;
  return (
    <div>
      <div className="section-label">{title}</div>
      <div className="form-grid">
        {keys.map((k) => <ParamInput key={k} name={k} spec={spec[k]} value={values?.[k]} onChange={(v) => onChange(k, v)} />)}
      </div>
    </div>
  );
}

export function ExperimentStudio() {
  return (
    <>
      <PageHeader title="Experiment Studio" subtitle="Configure a run of the existing PyTorch implementation. Only parameters the training code supports are exposed." />
      <BackendGate feature="configuring and running experiments">
        <StudioForm />
      </BackendGate>
    </>
  );
}

function StudioForm() {
  const nav = useNavigate();
  const [params] = useSearchParams();
  const { data: schema } = useApi<any>("/schema");
  const { data: presets } = useApi<any[]>("/presets");
  const [cfg, setCfg] = useState<any>(null);
  const [seedsText, setSeedsText] = useState("20260824");
  const [meta, setMeta] = useState({ name: "", hypothesis: "", notes: "" });
  const [preview, setPreview] = useState<any>(null);
  const [previewErr, setPreviewErr] = useState<string | null>(null);
  const [fmt, setFmt] = useState<"json" | "yaml">("yaml");
  const [busy, setBusy] = useState(false);
  const [confirm, setConfirm] = useState<any>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [presetId, setPresetId] = useState("");
  const seq = useRef(0);

  const build = (ds: string, ar: string, me: string, prev?: any) => {
    const d = schema.datasets[ds], a = schema.architectures[ar], m = schema.methods[me];
    const opt = { ...defaults(schema.optimization), ...(prev?.optimization ?? {}) };
    if (!prev || prev.architecture?.name !== ar) opt.lr = ar === "cnn" ? 0.1 : 0.2;   // paper protocol per arch
    return {
      dataset: { name: ds, ...defaults(d.params), ...(prev?.dataset?.name === ds ? prev.dataset : {}) },
      architecture: { name: ar, ...defaults(a.params), ...(prev?.architecture?.name === ar ? prev.architecture : {}) },
      method: { name: me, params: { ...defaults(m.params), ...(prev?.method?.name === me ? prev.method.params : {}) } },
      optimization: opt,
      evaluation: { ...defaults(schema.evaluation), ...(prev?.evaluation ?? {}) },
      device: prev?.device ?? "auto",
      checkpoint_policy: prev?.checkpoint_policy ?? "none",
    };
  };

  // initial config (optionally from ?preset=)
  useEffect(() => {
    if (!schema || cfg) return;
    setCfg(build("permuted_mnist", "mlp", "sgd"));
  }, [schema]); // eslint-disable-line react-hooks/exhaustive-deps

  const applyPreset = (id: string) => {
    setPresetId(id);
    const p = presets?.find((x) => x.id === id);
    if (!p) return;
    const { seed, metadata, schema_version, ...rest } = p.config;
    setCfg(rest);
    setSeedsText(String(p.seeds?.[0] ?? seed));
    setMeta({ name: p.id === "smoke" ? "Smoke test" : `Reproduce ${p.label}`, hypothesis: "", notes: p.note ?? "" });
  };
  useEffect(() => {
    const p = params.get("preset");
    if (p && presets && schema && cfg && !presetId) applyPreset(p);
  }, [presets, schema, cfg]); // eslint-disable-line react-hooks/exhaustive-deps

  const seeds = useMemo(() => seedsText.split(/[\s,]+/).filter(Boolean).map((s) => Number(s)), [seedsText]);
  const seedsValid = seeds.length > 0 && seeds.every((s) => Number.isInteger(s) && s >= 0);

  // live preview (debounced, server-side validation — the single source of truth)
  useEffect(() => {
    if (!cfg || !seedsValid) return;
    const my = ++seq.current;
    const t = window.setTimeout(async () => {
      try {
        const pv = await api("/experiments/configure", { method: "POST", json: { config: { ...cfg, seed: seeds[0] }, seeds } });
        if (my === seq.current) { setPreview(pv); setPreviewErr(null); }
      } catch (e) {
        if (my === seq.current) { setPreview(null); setPreviewErr(e instanceof ApiError ? (e.detail?.errors ?? [e.message]).join("; ") : String(e)); }
      }
    }, 350);
    return () => window.clearTimeout(t);
  }, [cfg, seedsText]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!schema || !cfg) return <div className="studio-note">Loading schema…</div>;

  const ds = cfg.dataset.name, ar = cfg.architecture.name, me = cfg.method.name;
  const set = (blk: string, k: string, v: any) => setCfg((c: any) => ({ ...c, [blk]: { ...c[blk], [k]: v } }));
  const setMethodParam = (k: string, v: any) => setCfg((c: any) => ({ ...c, method: { ...c.method, params: { ...c.method.params, [k]: v } } }));

  const submit = async (confirmLong = false, saveOnly = false) => {
    setBusy(true); setMsg(null);
    try {
      const body = { config: { ...cfg, seed: seeds[0] }, seeds, metadata: meta };
      if (saveOnly) {
        const r = await api("/experiments/configure", { method: "POST", json: { ...body, save: true } });
        setMsg(`Saved ${r.saved_ids.length} proposed run(s): ${r.saved_ids.join(", ")} — not executed.`);
      } else {
        const r = await api("/experiments/run", { method: "POST", json: { ...body, confirm_long: confirmLong } });
        nav(`/monitor/${r.queued[0]}`);
      }
      setConfirm(null);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409 && e.detail?.requires_confirmation) setConfirm(e.detail);
      else setMsg(e instanceof ApiError ? JSON.stringify(e.detail) : String(e));
    } finally {
      setBusy(false);
    }
  };

  const groups: Record<string, any[]> = {};
  (presets ?? []).forEach((p) => (groups[p.category] ??= []).push(p));
  const catLabel: Record<string, string> = { smoke: "Pipeline check", A: "A · Primary permuted MNIST", B: "B · λ sweep", C: "C · Radial sweep",
    D: "D · Width", E: "E · Task geometry", F: "F · ANR ablation" };

  return (
    <div className="studio-grid">
      <div className="grid" style={{ gap: 16 }}>
        <Card pad>
          <CardHead title="Start from an existing configuration" sub="Presets are reconstructed from the experiment scripts' constants" />
          <select className="select" value={presetId} onChange={(e) => applyPreset(e.target.value)} aria-label="Preset">
            <option value="">— custom configuration —</option>
            {Object.entries(groups).map(([cat, ps]) => (
              <optgroup key={cat} label={catLabel[cat] ?? cat}>
                {ps.map((p) => <option key={p.id} value={p.id}>{p.label}{p.paper_ref ? ` · ${p.paper_ref}` : ""}</option>)}
              </optgroup>
            ))}
          </select>
        </Card>

        <Card pad>
          <CardHead title="Dataset · Architecture · Update rule" />
          <div className="form-grid">
            <label className="field">
              <span className="field-label">Dataset</span>
              <select className="select" value={ds} onChange={(e) => {
                const nds = e.target.value;
                const nar = schema.datasets[nds].archs.includes(ar) ? ar : schema.datasets[nds].archs[0];
                const nme = schema.methods[me].archs.includes(nar) ? me : "sgd";
                setCfg(build(nds, nar, nme, cfg));
              }}>
                {Object.entries<any>(schema.datasets).map(([k, d]) => <option key={k} value={k}>{d.label}</option>)}
              </select>
            </label>
            <label className="field">
              <span className="field-label">Architecture</span>
              <select className="select" value={ar} onChange={(e) => {
                const nar = e.target.value;
                const nme = schema.methods[me].archs.includes(nar) ? me : "sgd";
                setCfg(build(ds, nar, nme, cfg));
              }}>
                {Object.entries<any>(schema.architectures).map(([k, a]) => (
                  <option key={k} value={k} disabled={!schema.datasets[ds].archs.includes(k)}>
                    {a.label}{!schema.datasets[ds].archs.includes(k) ? " — not defined for this dataset" : ""}
                  </option>
                ))}
              </select>
            </label>
            <label className="field" style={{ gridColumn: "1 / -1" }}>
              <span className="field-label">Update method</span>
              <select className="select" value={me} onChange={(e) => setCfg(build(ds, ar, e.target.value, cfg))}>
                {["primary", "control"].map((g) => (
                  <optgroup key={g} label={g === "primary" ? "Paper update rules" : "Existing control arms (e23 / e25)"}>
                    {Object.entries<any>(schema.methods).filter(([, m]) => m.group === g).map(([k, m]) => (
                      <option key={k} value={k} disabled={!m.archs.includes(ar)}>
                        {m.label}{!m.archs.includes(ar) ? " — MLP only (acts on hidden units)" : ""}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
              <span className="field-hint mono">{schema.methods[me].equation}</span>
            </label>
          </div>
        </Card>

        <Card pad>
          <div className="grid" style={{ gap: 18 }}>
            <Block title={`Method parameters — ${schema.methods[me].label}`} spec={schema.methods[me].params} values={cfg.method.params} onChange={setMethodParam} />
            <Block title="Dataset" spec={schema.datasets[ds].params} values={cfg.dataset} onChange={(k, v) => set("dataset", k, v)} />
            <Block title="Architecture" spec={schema.architectures[ar].params} values={cfg.architecture} onChange={(k, v) => set("architecture", k, v)} />
            <Block title="Optimization (plain SGD step; no momentum or weight decay in the code)" spec={schema.optimization} values={cfg.optimization} onChange={(k, v) => set("optimization", k, v)} />
            <Block title="Evaluation" spec={schema.evaluation} values={cfg.evaluation} onChange={(k, v) => set("evaluation", k, v)} />
          </div>
        </Card>

        <Card pad>
          <CardHead title="Seeds, execution and description" />
          <div className="form-grid">
            <label className="field" style={{ gridColumn: "1 / -1" }}>
              <span className="field-label">Seeds (one run per seed)</span>
              <input className="input mono" value={seedsText} onChange={(e) => setSeedsText(e.target.value)} />
              <span className="field-hint">
                <button className="linkish" type="button" onClick={() => setSeedsText(schema.paper_seeds.join(", "))}>use the five paper seeds</button>
                {!seedsValid && <span className="text-red"> · seeds must be non-negative integers</span>}
              </span>
            </label>
            <label className="field">
              <span className="field-label">Device</span>
              <select className="select" value={cfg.device} onChange={(e) => setCfg({ ...cfg, device: e.target.value })}>
                {schema.devices.map((d: string) => <option key={d}>{d}</option>)}
              </select>
            </label>
            <label className="field">
              <span className="field-label">Checkpoint policy</span>
              <select className="select" value={cfg.checkpoint_policy} onChange={(e) => setCfg({ ...cfg, checkpoint_policy: e.target.value })}>
                {schema.checkpoint_policies.map((d: string) => <option key={d}>{d}</option>)}
              </select>
            </label>
            <label className="field" style={{ gridColumn: "1 / -1" }}>
              <span className="field-label">Experiment name</span>
              <input className="input" value={meta.name} placeholder={preview?.description} onChange={(e) => setMeta({ ...meta, name: e.target.value })} />
            </label>
            <label className="field" style={{ gridColumn: "1 / -1" }}>
              <span className="field-label">Research hypothesis</span>
              <textarea className="input" rows={2} value={meta.hypothesis} onChange={(e) => setMeta({ ...meta, hypothesis: e.target.value })}
                        placeholder="e.g. A mild inward bias (γ = 0.2) lowers decay relative to SGD at matched step size." />
            </label>
          </div>
        </Card>
      </div>

      <div className="grid" style={{ gap: 16, alignContent: "start" }}>
        <Card pad>
          <CardHead title="Configuration preview" sub="Validated by the backend against the implementation" />
          {previewErr && <div className="error-box"><div className="eb-title">Invalid configuration</div><pre>{previewErr}</pre></div>}
          {preview && (
            <>
              <div className="mb3"><b>{preview.description}</b></div>
              {preview.errors.map((e: string) => <div key={e} className="msg err">✕ {e}</div>)}
              {preview.warnings.map((w: string) => <div key={w} className="msg warn">⚠ {w}</div>)}
              <KV rows={[
                ["Tasks", preview.n_tasks],
                ["Steps per run", preview.total_steps_per_run.toLocaleString()],
                ["Seeds", `${preview.n_seeds} (${preview.seeds.join(", ")})`],
                ["Estimate", preview.estimate.measurable
                  ? `${preview.estimate.seconds_per_run} s per run · ${preview.estimate.seconds} s total · ${preview.estimate.gpu_hours} GPU-h`
                  : "not measurable yet"],
                ["Estimate basis", <span className="small">{preview.estimate.basis}</span>],
                ["Config hash", <span className="mono small">{preview.config_hashes[0].slice(0, 16)}…</span>],
                ["Output directory", <span className="mono small">{preview.output_dir_pattern}</span>],
                ["Legacy counterpart", preview.legacy_counterpart.length
                  ? <span className="small">{preview.legacy_counterpart.map((c: any) => `seed ${c.seed} → ${c.record}`).join("; ")} — results will be checked against it</span>
                  : <span className="small muted">none — results will be labelled as new</span>],
              ]} />
              <div className="row between aic mt4 mb2">
                <div className="seg">
                  {(["yaml", "json"] as const).map((f) => (
                    <button key={f} className={`seg-btn ${fmt === f ? "on" : ""}`} onClick={() => setFmt(f)}>{f.toUpperCase()}</button>
                  ))}
                </div>
                <div className="row" style={{ gap: 6 }}>
                  <button className="btn sm ghost" onClick={() => downloadText("experiment.config.json", preview.json, "application/json")}>Download JSON</button>
                  <button className="btn sm ghost" onClick={() => downloadText("experiment.config.yaml", preview.yaml, "application/x-yaml")}>Download YAML</button>
                </div>
              </div>
              <pre className="code-block">{fmt === "yaml" ? preview.yaml : preview.json}</pre>
            </>
          )}
        </Card>

        {confirm && (
          <Card pad className="confirm-card">
            <CardHead title="Confirm a long run" />
            <p className="small">
              {confirm.estimate.measurable
                ? <>Estimated <b>{confirm.estimate.seconds} s</b> ({confirm.estimate.gpu_hours} GPU-hours) for {confirm.estimate.total_steps.toLocaleString()} SGD steps.</>
                : <>{confirm.estimate.total_steps.toLocaleString()} SGD steps with no measured timing for this architecture.</>}
              {" "}Runs execute one at a time on the single GPU.
            </p>
            <div className="row" style={{ gap: 8 }}>
              <button className="btn" disabled={busy} onClick={() => submit(true)}>Run anyway</button>
              <button className="btn ghost" onClick={() => setConfirm(null)}>Cancel</button>
            </div>
          </Card>
        )}

        <div className="row" style={{ gap: 8 }}>
          <button className="btn" disabled={busy || !preview?.valid || !seedsValid} onClick={() => submit(false)}>
            {busy ? "Submitting…" : `Run ${preview?.n_seeds ?? ""} experiment${preview?.n_seeds === 1 ? "" : "s"}`}
          </button>
          <button className="btn ghost" disabled={busy || !preview?.valid} onClick={() => submit(false, true)}>Save as proposed</button>
        </div>
        {msg && <div className="studio-note">{msg}</div>}
        <Observation label="Integrity" tone="neutral">
          Each seed becomes its own run with a new ID and a read-only configuration snapshot. Nothing runs
          until you press Run; long submissions require explicit confirmation.
        </Observation>
        <SourceNote>schema and validation from <span className="mono">studio/engine/config_loader.py</span></SourceNote>
      </div>
    </div>
  );
}
