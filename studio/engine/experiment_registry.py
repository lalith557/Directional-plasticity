"""Experiment registry: Studio runs (on disk) + legacy artifacts (read-only).

Run directory layout (``studio/experiments/<run_id>/``)::

    config.json      immutable configuration snapshot (read-only file, SHA-256 in meta)
    meta.json        identity, lifecycle status, timestamps, hardware, validation
    environment.json software / hardware / code / dataset fingerprints (written by runner)
    events.jsonl     one line per finished task (written by runner)
    progress.json    latest step position (written by runner)
    result.json      run summary + per-task arrays (written by runner on completion)
    log.txt          runner stdout/stderr
    checkpoints/     only if checkpoint_policy == "final"

Lifecycle: proposed → queued → initializing → running → completed | failed | cancelled.
"Paused" is not offered (the training loop cannot pause). Every new run gets a new id;
nothing is ever overwritten.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import secrets
import stat
import threading
from pathlib import Path

from . import config_loader as cl
from . import fsutil
from . import legacy_artifacts as la
from .checkpoint_manager import list_checkpoints
from .experiment_validator import verify_snapshot
from .paths import RUN_ID_RE, run_dir, runs_dir

TERMINAL = {"completed", "failed", "cancelled"}
ACTIVE = {"initializing", "running"}
STATUSES = ["proposed", "queued", "initializing", "running", "completed", "failed", "cancelled"]
_lock = threading.RLock()


def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _atomic_write(path: Path, text: str) -> None:
    if not fsutil.atomic_write(path, text):
        raise OSError(f"could not replace {path} (file kept open by another process)")


def new_run_id() -> str:
    with _lock:
        for _ in range(100):
            rid = f"run-{_dt.datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
            if not (runs_dir() / rid).exists():
                return rid
    raise RuntimeError("could not allocate a unique run id")


def research_category(cfg: dict) -> str:
    """Which research category (A–F) a configuration belongs to, by its content."""
    ds, ar, me = cfg["dataset"]["name"], cfg["architecture"], cfg["method"]
    if ds == "rotated_mnist":
        return "E"
    if ar["name"] == "mlp" and ar["hidden"] != 16:
        return "D"
    if me["name"] == "l2_init" and me["params"]["l2_init_lambda"] != 1e-2:
        return "B"
    if me["name"] == "dir_inward" and me["params"]["dir_coeff"] != 0.6:
        return "C"
    if me["name"] == "adaptive_norm_revive" and me["params"]["anr_mode"] != "full":
        return "F"
    if me["name"] in ("norm_project", "random_dir", "angle"):
        return "S"
    return "A"


def create_run(cfg: dict, *, metadata: dict | None = None, group_id: str | None = None,
               status: str = "proposed", reproduce_of: str | None = None) -> str:
    """Create a run directory with an immutable configuration snapshot."""
    assert status in ("proposed", "queued")
    rid = new_run_id()
    rd = run_dir(rid)
    rd.mkdir(parents=True, exist_ok=False)
    md = {**(cfg.get("metadata") or {}), **(metadata or {})}
    snapshot = {k: v for k, v in cfg.items() if k != "metadata"}
    if md:
        snapshot["metadata"] = md
    cfg_path = rd / "config.json"
    cfg_path.write_text(cl.to_json(snapshot), encoding="utf-8")
    os.chmod(cfg_path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)       # immutable snapshot
    meta = {
        "id": rid, "config_hash": cl.config_hash(snapshot), "protocol_key": cl.protocol_key(snapshot),
        "name": md.get("name") or cl.describe(snapshot), "hypothesis": md.get("hypothesis"),
        "notes": md.get("notes"), "group_id": group_id or rid, "status": status,
        "research_category": research_category(snapshot), "reproduce_of": reproduce_of,
        "created_at": now_iso(), "queued_at": now_iso() if status == "queued" else None,
        "started_at": None, "finished_at": None, "duration_s": None, "steps_done": None,
        "device": snapshot["device"], "device_name": None, "error": None, "exit_code": None,
        "validation": {"status": "not_validated"}, "flags": [],
        "checkpoint_policy": snapshot["checkpoint_policy"],
    }
    _atomic_write(rd / "meta.json", json.dumps(meta, indent=2))
    return rid


def read_meta(rid: str) -> dict:
    meta = fsutil.read_json(run_dir(rid) / "meta.json")
    if meta is None:
        raise FileNotFoundError(f"{rid}: meta.json missing")
    return meta


def update_meta(rid: str, **fields) -> dict:
    with _lock:
        meta = read_meta(rid)
        meta.update(fields)
        _atomic_write(run_dir(rid) / "meta.json", json.dumps(meta, indent=2))
        return meta


def read_config(rid: str) -> dict:
    return fsutil.read_json(run_dir(rid) / "config.json")


def read_json(rid: str, name: str):
    try:
        return fsutil.read_json(run_dir(rid) / name)
    except (json.JSONDecodeError, PermissionError):
        return None                     # being written right now; caller polls again


def read_events(rid: str, start: int = 0) -> list[dict]:
    p = run_dir(rid) / "events.jsonl"
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    break      # partially written last line; picked up next poll
    return out


def read_log(rid: str, tail: int = 200) -> list[str]:
    p = run_dir(rid) / "log.txt"
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-max(1, min(tail, 5000)):]


def list_run_ids() -> list[str]:
    return sorted((p.name for p in runs_dir().iterdir() if p.is_dir() and RUN_ID_RE.match(p.name)),
                  reverse=True)


# --------------------------------------------------------------------------- #
# Uniform records                                                              #
# --------------------------------------------------------------------------- #
def studio_record(rid: str, full: bool = False) -> dict:
    meta = read_meta(rid)
    cfg = read_config(rid)
    res = read_json(rid, "result.json")
    me = cl.METHODS[cfg["method"]["name"]]
    summ = (res or {}).get("summary", {})
    metric_keys = ("mean_acc", "decay", "final_weight_norm", "final_dormant", "final_eff_rank",
                   "matched_max_dev", "peak_acc", "end_acc", "grad_align_mean", "radial_align_mean",
                   "cor1_relative_error")
    metrics = {k: {"mean": summ.get(k), "std": None} for k in metric_keys if k in summ}
    row = {"seed": cfg["seed"], **{k: summ.get(k) for k in metric_keys if k in summ}}
    rec = {
        "id": rid, "source": "studio", "name": meta["name"], "hypothesis": meta.get("hypothesis"),
        "category": meta.get("research_category", "A"),
        "category_label": la.CATEGORIES.get(meta.get("research_category", "A")),
        "experiment": "studio", "status": meta["status"], "validation": meta.get("validation"),
        "config": cfg, "config_hash": meta["config_hash"], "protocol_key": meta["protocol_key"],
        "dataset": cfg["dataset"]["name"], "architecture": cfg["architecture"],
        "method": cfg["method"]["name"], "method_label": me["label"], "family": me["family"],
        "matched_step": me["matched_step"], "params": cfg["method"]["params"],
        "seeds": [cfg["seed"]], "n_seeds": 1, "group": meta.get("group_id"),
        "metrics": metrics, "per_seed": [row] if summ else None,
        "has_per_task": (run_dir(rid) / "events.jsonl").exists(),
        "created_at": meta["created_at"], "started_at": meta.get("started_at"),
        "finished_at": meta.get("finished_at"), "duration_s": meta.get("duration_s"),
        "device_name": meta.get("device_name"), "error": meta.get("error"),
        "flags": meta.get("flags", []), "reproduce_of": meta.get("reproduce_of"),
        "paper_ref": None, "artifact": f"studio/experiments/{rid}/",
    }
    if full:
        rec.update(meta=meta, progress=read_json(rid, "progress.json"),
                   environment=read_json(rid, "environment.json"),
                   checkpoints=list_checkpoints(run_dir(rid)),
                   snapshot_check=verify_snapshot(run_dir(rid)),
                   result=res, log_available=(run_dir(rid) / "log.txt").exists())
    return rec


def all_records(include_phase1: bool = True) -> list[dict]:
    recs = []
    for rid in list_run_ids():
        try:
            recs.append(studio_record(rid))
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    recs += [r for r in la.index() if include_phase1 or r["category"] != "P1"]
    return recs


def get_record(record_id: str, full: bool = True) -> dict | None:
    if RUN_ID_RE.match(record_id):
        if not (runs_dir() / record_id).exists():
            return None
        return studio_record(record_id, full=full)
    return la.get(record_id)


def summary_counts() -> dict:
    studio = [read_meta(r) for r in list_run_ids()]
    legacy = la.index()
    by = lambda xs, k: {s: sum(1 for x in xs if x.get(k) == s) for s in {x.get(k) for x in xs}}
    validated = sum(1 for m in studio if (m.get("validation") or {}).get("status", "").startswith("reproduced"))
    return {
        "legacy": {"total": len(legacy), "by_status": by(legacy, "status"),
                   "by_category": by(legacy, "category")},
        "studio": {"total": len(studio), "by_status": by(studio, "status"), "validated": validated},
        "datasets": list(cl.DATASETS), "architectures": list(cl.ARCHITECTURES),
        "recent": [studio_record(r)
                   for r in list_run_ids()[:8]],
    }
