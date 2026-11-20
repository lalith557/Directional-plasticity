"""Continual Learning Experimental Studio — FastAPI backend.

Start:  python -m uvicorn studio.api.server:app --host 127.0.0.1 --port 8765

Binds to localhost only. Every response is computed from the registry (Studio runs on
disk + read-only legacy artifacts); there are no placeholder or simulated values.
"""
from __future__ import annotations

import asyncio
import contextlib
import math
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from studio import __version__
from studio.engine import config_loader as cl
from studio.engine import evaluation_service as ev
from studio.engine import experiment_registry as reg
from studio.engine import experiment_validator as val
from studio.engine import legacy_artifacts as la
from studio.engine import paper_results as pr
from studio.engine import resource_monitor as rm
from studio.engine.execution_queue import ExecutionManager
from studio.engine.paths import RUN_ID_RE, runs_dir

MANAGER = ExecutionManager(max_concurrent=1)
# Submissions above these limits need an explicit confirm_long=true (no silent long jobs).
LONG_SECONDS = 600
LONG_STEPS_UNMEASURED = 3000


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    MANAGER.start()
    yield
    MANAGER.stop()


app = FastAPI(title="Continual Learning Experimental Studio", version=__version__, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
                   allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------- #
# Models                                                                       #
# --------------------------------------------------------------------------- #
class Metadata(BaseModel):
    name: str | None = None
    hypothesis: str | None = None
    notes: str | None = None


class ConfigureRequest(BaseModel):
    config: dict[str, Any]
    seeds: list[int] | None = None
    metadata: Metadata | None = None
    save: bool = False


class RunRequest(BaseModel):
    config: dict[str, Any] | None = None
    seeds: list[int] | None = None
    metadata: Metadata | None = None
    experiment_ids: list[str] | None = None
    reproduce_of: str | None = None
    confirm_long: bool = False


class CompareRequest(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=20)
    metrics: list[str] | None = None
    format: str = "json"


class CorrelationRequest(BaseModel):
    ids: list[str] = Field(min_length=3, max_length=200)
    x: str = "final_weight_norm"
    y: str = "decay"


class CurvesRequest(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=20)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _clean(o):
    """JSON-safe: non-finite floats → None (never 0)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def _device_name() -> str | None:
    g = rm.gpu_status()
    return g.get("device") if g.get("cuda_available") else "cpu"


def _history() -> list[dict]:
    out = []
    for rid in reg.list_run_ids():
        m = reg.read_meta(rid)
        if m["status"] == "completed":
            out.append({"config": reg.read_config(rid), "duration_s": m.get("duration_s"),
                        "steps_done": m.get("steps_done"), "device_name": m.get("device_name")})
    return out


def _prepare(config: dict, seeds: list[int] | None) -> list[dict]:
    try:
        base = cl.normalize(config)
    except cl.ConfigError as e:
        raise HTTPException(400, {"errors": [str(e)]}) from None
    seeds = seeds or [base["seed"]]
    if len(seeds) != len(set(seeds)):
        raise HTTPException(400, {"errors": ["duplicate seeds"]})
    if len(seeds) > 10:
        raise HTTPException(400, {"errors": ["at most 10 seeds per submission"]})
    cfgs = []
    for s in seeds:
        c = dict(base, seed=int(s))
        cfgs.append(cl.normalize(c))
    return cfgs


def _preview(cfgs: list[dict]) -> dict:
    errors, warnings = val.validate_config(cfgs[0])
    for c in cfgs[1:]:
        e2, _ = val.validate_config(c)
        errors += [x for x in e2 if x not in errors]
    est = rm.estimate(cfgs[0], len(cfgs), _history(), _device_name())
    long = (est["measurable"] and est["seconds"] > LONG_SECONDS) or \
           (not est["measurable"] and est["total_steps"] > LONG_STEPS_UNMEASURED)
    return {
        "valid": not errors, "errors": errors, "warnings": warnings,
        "description": cl.describe(cfgs[0]), "n_seeds": len(cfgs), "seeds": [c["seed"] for c in cfgs],
        "n_tasks": cfgs[0]["dataset"]["n_tasks"],
        "total_steps_per_run": cfgs[0]["dataset"]["n_tasks"] * cfgs[0]["optimization"]["steps_per_task"],
        "configs": cfgs, "config_hashes": [cl.config_hash(c) for c in cfgs],
        "protocol_key": cl.protocol_key(cfgs[0]),
        "json": cl.to_json(cfgs[0]), "yaml": cl.to_yaml(cfgs[0]),
        "estimate": est, "requires_confirmation": long,
        "output_dir_pattern": f"{runs_dir().as_posix()}/run-<YYYYMMDD-HHMMSS>-<id>/",
        "legacy_counterpart": [
            {"seed": c["seed"], "record": hit[0]["id"]} for c in cfgs
            if (hit := la.per_seed_reference(c)) is not None],
    }


def _summary(r: dict) -> dict:
    keep = ("id", "source", "name", "category", "category_label", "experiment", "status", "validation",
            "dataset", "architecture", "method", "method_label", "family", "matched_step", "params",
            "seeds", "n_seeds", "group", "metrics", "paper_ref", "artifact", "created_at", "has_per_task",
            "config_hash", "duration_s", "device_name", "arm_label", "sweep", "verdict", "error", "notes")
    return {k: r.get(k) for k in keep if k in r}


def _group_record(gid: str) -> dict | None:
    rids = [rid for rid in reg.list_run_ids() if reg.read_meta(rid).get("group_id") == gid]
    recs = [reg.studio_record(r, full=True) for r in rids]
    done = [r for r in recs if r["status"] == "completed"]
    if not recs:
        return None
    base = recs[0]
    metrics = {}
    for k in ("mean_acc", "decay", "final_weight_norm", "final_dormant", "final_eff_rank", "matched_max_dev"):
        agg = ev.aggregate([(r["metrics"].get(k) or {}).get("mean") for r in done])
        if agg["n"]:
            metrics[k] = {"mean": agg["mean"], "std": agg["std"], "n": agg["n"]}
    rows = [row for r in done for row in (r["per_seed"] or [])]
    curves = {}
    for key in ("acc", "weight_norm", "dormant", "eff_rank", "radial_component_mean",
                "tangential_component_mean", "step_norm_mean"):
        cs = [((r.get("result") or {}).get("summary") or {}).get("per_task", {}).get(key) for r in done]
        cs = [c for c in cs if c]
        if cs:
            curves[key] = ev.aggregate_curves(cs)
    return {**{k: base[k] for k in ("dataset", "architecture", "method", "method_label", "family",
                                     "matched_step", "params", "protocol_key", "config")},
            "id": f"group:{gid}", "source": "studio_group", "name": f"{base['name']} (group of {len(recs)})",
            "status": "completed" if len(done) == len(recs) else f"{len(done)}/{len(recs)} completed",
            "seeds": [r["seeds"][0] for r in done], "n_seeds": len(done), "metrics": metrics,
            "per_seed": rows, "group": gid, "has_per_task": bool(curves), "curves": curves,
            "members": rids, "duration_s": sum(r.get("duration_s") or 0 for r in done) or None}


def _resolve(record_id: str) -> dict:
    if record_id.startswith("group:"):
        r = _group_record(record_id[6:])
    else:
        r = reg.get_record(record_id, full=True)
    if r is None:
        raise HTTPException(404, f"experiment {record_id!r} not found")
    return r


def _curves(r: dict) -> dict:
    if r.get("source") == "studio_group":
        return r.get("curves", {})
    pt = (((r.get("result") or {}).get("summary") or {}).get("per_task")) if r.get("source") == "studio" else None
    return {k: {"mean": v, "std": None, "n": None} for k, v in (pt or {}).items()}


# --------------------------------------------------------------------------- #
# Endpoints                                                                    #
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    g = rm.gpu_status()
    return {"status": "ok", "version": __version__, "runs_dir": str(runs_dir()),
            "legacy_records": len(la.index()), "cuda_available": g.get("cuda_available"),
            "device": g.get("device"), "queue": MANAGER.state()}


@app.get("/schema")
def get_schema():
    return cl.schema()


@app.get("/presets")
def presets():
    """Configurations of existing experiments (to reproduce or adapt) + a smoke test."""
    out = [{"id": "smoke", "label": "Smoke test — SGD, 5 tasks (≈2 s on GPU)", "category": "smoke",
            "config": cl.normalize({"dataset": {"name": "permuted_mnist", "n_tasks": 5}}),
            "seeds": [cl.PAPER_SEEDS[0]], "paper_ref": None,
            "note": "Short pipeline check; not a paper configuration."}]
    for r in la.index():
        if r["category"] in ("A", "B", "C", "D", "E", "F") and r.get("config"):
            out.append({"id": r["id"], "label": r["name"], "category": r["category"],
                        "config": r["config"], "seeds": r["seeds"], "paper_ref": r.get("paper_ref"),
                        "note": f"Reconstructed from {r['script']} constants; artifact {r['artifact']}."})
    return _clean(out)


@app.get("/registry/summary")
def registry_summary():
    return _clean(reg.summary_counts())


@app.get("/experiments")
def list_experiments(source: str | None = None, category: str | None = None, status: str | None = None,
                     dataset: str | None = None, method: str | None = None, q: str | None = None,
                     include_phase1: bool = True):
    recs = reg.all_records(include_phase1=include_phase1)
    def ok(r):
        if source and r.get("source") != source: return False
        if category and r.get("category") != category: return False
        if status and r.get("status") != status: return False
        if dataset and r.get("dataset") != dataset: return False
        if method and r.get("method") != method: return False
        if q:
            hay = " ".join(str(r.get(k, "")) for k in ("id", "name", "method_label", "experiment",
                                                        "paper_ref", "category_label", "status"))
            if q.lower() not in hay.lower(): return False
        return True
    return _clean([_summary(r) for r in recs if ok(r)])


@app.get("/experiments/{record_id}")
def get_experiment(record_id: str):
    return _clean(_resolve(record_id))


@app.post("/experiments/configure")
def configure(req: ConfigureRequest):
    cfgs = _prepare(req.config, req.seeds)
    pv = _preview(cfgs)
    if req.save:
        if not pv["valid"]:
            raise HTTPException(400, {"errors": pv["errors"]})
        md = req.metadata.model_dump(exclude_none=True) if req.metadata else {}
        gid, ids = None, []
        for c in cfgs:                      # first run's id becomes the group id
            rid = reg.create_run(c, metadata=md, group_id=gid, status="proposed")
            gid = gid or rid
            ids.append(rid)
        pv["saved_ids"] = ids
        pv["group_id"] = gid
    return _clean(pv)


@app.post("/experiments/run")
def run(req: RunRequest):
    if req.experiment_ids:
        ids = req.experiment_ids
        for rid in ids:
            if not RUN_ID_RE.match(rid) or reg.get_record(rid, full=False) is None:
                raise HTTPException(404, f"unknown run {rid}")
        cfgs = [reg.read_config(rid) for rid in ids]
        pv = _preview(cfgs)
        if pv["requires_confirmation"] and not req.confirm_long:
            raise HTTPException(409, {"requires_confirmation": True, "estimate": pv["estimate"]})
        try:
            MANAGER.submit(ids)
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        return {"queued": ids, "estimate": pv["estimate"], "queue": MANAGER.state()}

    reproduce_of = None
    if req.reproduce_of:
        src = _resolve(req.reproduce_of)
        if not src.get("config"):
            raise HTTPException(400, f"{req.reproduce_of} has no reproducible configuration")
        config = {k: v for k, v in src["config"].items() if k != "metadata"}
        seeds = req.seeds or src.get("seeds") or [config["seed"]]
        reproduce_of = req.reproduce_of
        md = {"name": f"Reproduction of {src['name']}",
              "hypothesis": "The Studio engine reproduces the original configuration's results.",
              "notes": f"reproduce_of={reproduce_of}"}
    elif req.config is not None:            # {} is valid: all paper defaults
        config, seeds = req.config, req.seeds
        md = req.metadata.model_dump(exclude_none=True) if req.metadata else {}
    else:
        raise HTTPException(400, "provide config, experiment_ids or reproduce_of")

    cfgs = _prepare(config, seeds)
    pv = _preview(cfgs)
    if not pv["valid"]:
        raise HTTPException(400, {"errors": pv["errors"]})
    if pv["requires_confirmation"] and not req.confirm_long:
        raise HTTPException(409, {"requires_confirmation": True, "estimate": pv["estimate"],
                                  "message": "Estimated compute exceeds the automatic-launch limit; "
                                             "resubmit with confirm_long=true."})
    ids, gid = [], None
    for c in cfgs:                          # first run's id becomes the group id
        rid = reg.create_run(c, metadata=md, group_id=gid, status="proposed", reproduce_of=reproduce_of)
        gid = gid or rid
        ids.append(rid)
    MANAGER.submit(ids)
    return _clean({"queued": ids, "group_id": gid, "estimate": pv["estimate"], "queue": MANAGER.state(),
                   "legacy_counterpart": pv["legacy_counterpart"]})


@app.post("/experiments/{record_id}/cancel")
def cancel(record_id: str):
    if not RUN_ID_RE.match(record_id):
        raise HTTPException(400, "only Studio runs can be cancelled")
    if reg.get_record(record_id, full=False) is None:
        raise HTTPException(404, "not found")
    return {"id": record_id, "result": MANAGER.cancel(record_id)}


@app.get("/experiments/{record_id}/status")
def status(record_id: str):
    if not RUN_ID_RE.match(record_id):
        r = _resolve(record_id)
        return {"id": record_id, "status": r.get("status"), "source": r.get("source")}
    if reg.get_record(record_id, full=False) is None:
        raise HTTPException(404, "not found")
    m = reg.read_meta(record_id)
    st = MANAGER.state()
    pos = st["queued"].index(record_id) + 1 if record_id in st["queued"] else None
    return _clean({"id": record_id, "status": m["status"], "error": m.get("error"),
                   "progress": reg.read_json(record_id, "progress.json"), "queue_position": pos,
                   "cancelling": record_id in st["cancelling"], "started_at": m.get("started_at"),
                   "finished_at": m.get("finished_at"), "device_name": m.get("device_name")})


@app.get("/experiments/{record_id}/metrics")
def metrics(record_id: str):
    r = _resolve(record_id)
    if r.get("source") == "studio":
        return _clean({"id": record_id, "kind": "per_task", "events": reg.read_events(record_id),
                       "progress": reg.read_json(record_id, "progress.json"),
                       "emitted_fields": ["acc", "weight_norm", "dormant", "eff_rank", "step_norm_mean",
                                          "sgd_norm_mean", "matched_dev_max", "grad_align_mean",
                                          "radial_cos_mean", "radial_component_mean",
                                          "tangential_component_mean", "radial_energy_frac",
                                          "decay_running_estimate", "online_acc"]})
    return _clean({"id": record_id, "kind": "aggregate", "metrics": r.get("metrics"),
                   "per_seed": r.get("per_seed"), "curves": r.get("curves"),
                   "note": r.get("per_task_note", "Per-task trajectories not available.")})


@app.get("/experiments/{record_id}/logs")
def logs(record_id: str, tail: int = Query(200, ge=1, le=5000)):
    if not RUN_ID_RE.match(record_id):
        return {"id": record_id, "lines": [], "note": "Legacy artifacts have no Studio log; see "
                                                       "results/tables/*_stdout.log where present."}
    if reg.get_record(record_id, full=False) is None:
        raise HTTPException(404, "not found")
    return {"id": record_id, "lines": reg.read_log(record_id, tail)}


@app.get("/experiments/{record_id}/results")
def results(record_id: str):
    r = _resolve(record_id)
    if r.get("source") == "studio":
        return _clean(reg.read_json(record_id, "result.json") or {"status": r["status"], "result": None})
    return _clean({"metrics": r.get("metrics"), "per_seed": r.get("per_seed"), "artifact": r.get("artifact")})


@app.get("/experiments/{record_id}/config")
def config_file(record_id: str, format: str = Query("json", pattern="^(json|yaml)$")):
    r = _resolve(record_id)
    if not r.get("config"):
        raise HTTPException(404, "this record has no configuration")
    cfg = dict(r["config"])
    if r.get("source") == "legacy_artifact":
        cfg["metadata"] = {"name": r["name"],
                           "notes": f"Reconstructed from {r.get('script')} constants; artifact "
                                    f"{r.get('artifact')}; seeds {r.get('seeds')}"}
    body = cl.to_yaml(cfg) if format == "yaml" else cl.to_json(cfg)
    fname = f"{record_id.replace(':', '_')}.config.{format}"
    return PlainTextResponse(body, media_type="application/x-yaml" if format == "yaml" else "application/json",
                             headers={"Content-Disposition": f'attachment; filename="{fname}"'})


def _protocol_diff(a: dict, b: dict) -> list[str]:
    ca, cb = a.get("config") or {}, b.get("config") or {}
    diffs = []
    for blk in ("dataset", "architecture", "optimization", "evaluation"):
        for k in set(ca.get(blk, {})) | set(cb.get(blk, {})):
            if ca.get(blk, {}).get(k) != cb.get(blk, {}).get(k):
                diffs.append(f"{blk}.{k}: {ca.get(blk, {}).get(k)} vs {cb.get(blk, {}).get(k)}")
    return diffs


@app.post("/experiments/compare")
def compare(req: CompareRequest):
    recs = [_resolve(i) for i in req.ids]
    keys = req.metrics or ["mean_acc", "decay", "final_weight_norm", "final_dormant", "final_eff_rank",
                           "matched_max_dev"]
    table = []
    for r in recs:
        row = {"id": r["id"], "name": r.get("name"), "source": r.get("source"),
               "method_label": r.get("method_label"), "n_seeds": r.get("n_seeds"),
               "seeds": r.get("seeds"), "dataset": r.get("dataset"), "architecture": r.get("architecture"),
               "matched_step": r.get("matched_step"), "duration_s": r.get("duration_s"),
               "gpu_hours": (round(r["duration_s"] / 3600, 5) if r.get("duration_s") and
                             (r.get("device_name") or "cpu") != "cpu" else None),
               "metrics": {k: (r.get("metrics") or {}).get(k) for k in keys},
               "initial_weight_norm_task0": (((r.get("result") or {}).get("summary") or {})
                                             .get("initial_weight_norm_task0"))}
        table.append(row)
    pairs = []
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            a, b = recs[i], recs[j]
            reasons = []
            if a.get("protocol_key") is None or a.get("protocol_key") != b.get("protocol_key"):
                d = _protocol_diff(a, b)
                reasons.append("protocols differ" + (f" ({'; '.join(d[:6])})" if d else ""))
            if not a.get("per_seed") or not b.get("per_seed"):
                reasons.append("per-seed values not available for both")
            entry = {"a": a["id"], "b": b["id"], "valid": not reasons, "reasons": reasons}
            if not reasons:
                entry["results"] = {}
                for k in ("mean_acc", "decay", "final_weight_norm"):
                    ma = {row["seed"]: row.get(k) for row in a["per_seed"]}
                    mb = {row["seed"]: row.get(k) for row in b["per_seed"]}
                    entry["results"][k] = ev.paired(ma, mb, k, ev.HIGHER_IS_BETTER.get(k, True))
                if not any(res.get("ok") for res in entry["results"].values()):
                    entry["valid"] = False
                    entry["reasons"] = ["no shared seeds"]
            pairs.append(entry)
    out = _clean({"table": table, "pairs": pairs, "curves": {r["id"]: _curves(r) for r in recs},
                  "notes": ["Paired comparisons are computed only for records with identical protocols "
                            "(everything except the update rule) and shared seeds.",
                            "Descriptive differences, test statistics and interpretation are reported "
                            "separately; no method is declared superior automatically."]})
    if req.format == "markdown":
        return PlainTextResponse(_markdown_report(out), media_type="text/markdown",
                                 headers={"Content-Disposition": 'attachment; filename="comparison_report.md"'})
    return out


def _fmt(v, d=4):
    return "n/a" if v is None else (f"{v:.{d}g}" if abs(v) < 1e-3 and v != 0 else f"{v:.{d}f}")


def _markdown_report(o: dict) -> str:
    L = ["# Method comparison report", "", "Generated by the Continual Learning Experimental Studio "
         "from registry records (measured values only).", "", "## Metrics", ""]
    keys = list(o["table"][0]["metrics"]) if o["table"] else []
    L.append("| Experiment | Source | Seeds | " + " | ".join(keys) + " |")
    L.append("|---|---|---|" + "---|" * len(keys))
    for r in o["table"]:
        cells = []
        for k in keys:
            m = r["metrics"].get(k) or {}
            cells.append(_fmt(m.get("mean")) + (f" ± {_fmt(m['std'])}" if m.get("std") is not None else ""))
        L.append(f"| {r['name']} (`{r['id']}`) | {r['source']} | {r['n_seeds']} | " + " | ".join(cells) + " |")
    L += ["", "## Paired comparisons", ""]
    for p in o["pairs"]:
        if not p["valid"]:
            L.append(f"- `{p['a']}` vs `{p['b']}`: **not paired** — {'; '.join(p['reasons'])}")
            continue
        for k, res in p["results"].items():
            if not res.get("ok"):
                continue
            d = res["descriptive"]
            L.append(f"- `{p['a']}` − `{p['b']}` on **{k}**: mean difference {_fmt(d['mean_difference'])} "
                     f"(first better on {d['a_better_on']}/{d['of']} seeds); sign test p = "
                     f"{_fmt(res['tests'].get('sign_test_p_two_sided'))}. {res['interpretation_note']}")
    L += ["", "## Notes", ""] + [f"- {n}" for n in o["notes"]]
    return "\n".join(L) + "\n"


@app.post("/analysis/correlation")
def correlation(req: CorrelationRequest):
    recs = [_resolve(i) for i in req.ids]
    xs = [((r.get("metrics") or {}).get(req.x) or {}).get("mean") for r in recs]
    ys = [((r.get("metrics") or {}).get(req.y) or {}).get("mean") for r in recs]
    level = "method"
    c = ev.correlation(xs, ys, labels=[r.get("arm_label") or r.get("name") for r in recs], level=level)
    c["points"] = [{"id": r["id"], "label": r.get("arm_label") or r.get("name"), "x": x, "y": y,
                    "family": r.get("family"), "n_seeds": r.get("n_seeds")} for r, x, y in zip(recs, xs, ys)]
    c["x"], c["y"] = req.x, req.y
    return _clean(c)


@app.get("/analysis/regimes")
def regimes():
    out = []
    for rg in pr.PAPER["regimes"]:
        c = pr.regime_correlation(rg["key"])
        recs = [r for r in la.index() if r["group"] == rg["key"]]
        c["points"] = [{"id": r["id"], "label": r["arm_label"], "family": r["family"],
                        "x": (r["metrics"].get("final_weight_norm") or {}).get("mean"),
                        "y": (r["metrics"].get("decay") or {}).get("mean")} for r in recs]
        c.update(key=rg["key"], label=rg["label"], paper_r=rg["r"][0], seeds=recs[0]["n_seeds"] if recs else None,
                 dataset=recs[0]["dataset"] if recs else None,
                 architecture=recs[0]["architecture"] if recs else None,
                 source=recs[0]["artifact"] if recs else None)
        out.append(c)
    return _clean(out)


@app.post("/analysis/curves")
def curves(req: CurvesRequest):
    out = {}
    for i in req.ids:
        r = _resolve(i)
        out[i] = {"name": r.get("name"), "source": r.get("source"), "curves": _curves(r),
                  "note": None if _curves(r) else (r.get("per_task_note") or "No per-task data.")}
    return _clean(out)


@app.get("/resources/gpu")
def gpu():
    return _clean({**rm.gpu_status(), "queue": MANAGER.state()})


@app.get("/queue")
def queue():
    return MANAGER.state()


@app.get("/paper-results")
def paper_results():
    return _clean(pr.findings())


@app.websocket("/ws/experiments/{record_id}")
async def ws_experiment(websocket: WebSocket, record_id: str):
    await websocket.accept()
    if not RUN_ID_RE.match(record_id) or reg.get_record(record_id, full=False) is None:
        await websocket.send_json({"type": "error", "message": "unknown Studio run"})
        await websocket.close()
        return
    sent, last_prog, last_status, ticks = 0, None, None, 0
    try:
        while True:
            ticks += 1
            if ticks % 10 == 0:                      # keep-alive so clients never block forever
                await websocket.send_json({"type": "heartbeat"})
            meta = reg.read_meta(record_id)
            new = reg.read_events(record_id, start=sent)
            if new:
                await websocket.send_json(_clean({"type": "events", "offset": sent, "events": new}))
                sent += len(new)
            prog = reg.read_json(record_id, "progress.json")
            if prog and prog != last_prog:
                await websocket.send_json(_clean({"type": "progress", **prog}))
                last_prog = prog
            if meta["status"] != last_status:
                await websocket.send_json({"type": "status", "status": meta["status"],
                                           "error": meta.get("error"),
                                           "validation": meta.get("validation")})
                last_status = meta["status"]
            if meta["status"] in reg.TERMINAL and not new:
                await websocket.send_json({"type": "end", "status": meta["status"]})
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return
