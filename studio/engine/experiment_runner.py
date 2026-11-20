"""Execute one Studio run in its own process.

    python -m studio.engine.experiment_runner <run_id>

Runs in a subprocess so that cancellation is reliable, a crash cannot take down the
API server, and GPU memory is released when the run ends. The run's configuration
snapshot is verified against its hash before anything is computed.
"""
from __future__ import annotations

import json
import sys
import time
import traceback

from . import config_loader as cl
from . import experiment_registry as reg
from . import experiment_validator as val
from . import legacy_artifacts as la
from . import reproducibility
from .checkpoint_manager import save_final
from .metrics_collector import CancelledRun, MetricsCollector, summarize
from .paths import run_dir


def _log(msg: str) -> None:
    print(f"[{reg.now_iso()}] {msg}", flush=True)


def execute(run_id: str) -> str:
    rd = run_dir(run_id)
    meta = reg.read_meta(run_id)
    if meta["status"] in reg.TERMINAL:
        _log(f"run already {meta['status']}; refusing to re-execute (create a new run instead)")
        return meta["status"]

    snap = val.verify_snapshot(rd)
    if not snap["ok"]:
        reg.update_meta(run_id, status="failed", finished_at=reg.now_iso(), error=snap["reason"])
        _log(f"FAILED: {snap['reason']}")
        return "failed"
    cfg = reg.read_config(run_id)
    errors, warnings = val.validate_config(cfg)
    if errors:
        reg.update_meta(run_id, status="failed", finished_at=reg.now_iso(),
                        error="invalid configuration: " + "; ".join(errors))
        _log("FAILED: invalid configuration")
        return "failed"

    t_start = time.time()
    reg.update_meta(run_id, status="initializing", started_at=reg.now_iso())
    _log(f"initializing {run_id}: {cl.describe(cfg)} · seed {cfg['seed']}")
    for w in warnings:
        _log(f"warning: {w}")
    try:
        import torch
        from . import training_service as ts

        if cfg["device"] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device 'cuda' requested but CUDA is not available "
                               "(not silently falling back to CPU)")
        device, profile = ts.resolve_device(cfg["device"])
        env = reproducibility.snapshot()
        (rd / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
        ts.mnist()                                   # load data before declaring 'running'
        device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        reg.update_meta(run_id, status="running", device_name=device_name, device_profile=profile)
        _log(f"running on {device_name}")

        ckpt = {}

        def on_final(model):
            if cfg["checkpoint_policy"] == "final":
                ckpt.update(save_final(model, rd, meta["config_hash"]))
                _log(f"checkpoint saved: {ckpt['path']} sha256={ckpt['sha256'][:12]}…")

        collector = MetricsCollector(rd, cfg["dataset"]["n_tasks"], cfg["optimization"]["steps_per_task"],
                                     on_final_task=on_final)
        t_train = time.time()
        result = ts.run(cfg, device, progress_cb=collector)
        train_s = time.time() - t_train

        me = cl.METHODS[cfg["method"]["name"]]
        summary = summarize(result, collector.records, cfg, me["matched_step"])
        flags = val.result_flags(summary)
        validation = {"status": "no_legacy_counterpart",
                      "detail": "no per-seed legacy record has this exact configuration"}
        ref = la.per_seed_reference(cfg)
        if ref is not None:
            rec, row = ref
            cmp = val.compare_to_legacy(summary, row)
            validation = {**cmp, "against": rec["id"], "artifact": rec["artifact"], "seed": cfg["seed"],
                          "legacy_values": {k: row.get(k) for k in val.REPRO_METRICS}}
        steps_done = summary["n_tasks_completed"] * cfg["optimization"]["steps_per_task"]
        out = {"run_id": run_id, "config_hash": meta["config_hash"], "summary": summary,
               "flags": flags, "validation": validation, "checkpoint": ckpt or None,
               "timing": {"train_s": round(train_s, 3), "total_s": round(time.time() - t_start, 3),
                          "steps": steps_done}}
        (rd / "result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        reg.update_meta(run_id, status="completed", finished_at=reg.now_iso(),
                        duration_s=round(train_s, 3), steps_done=steps_done,
                        validation=validation, flags=flags)
        _log(f"completed: mean_acc={summary['mean_acc']:.4f} decay={summary['decay']:.4f} "
             f"final_norm={summary['final_weight_norm']} validation={validation['status']}")
        return "completed"
    except CancelledRun:
        done = len(reg.read_events(run_id))
        reg.update_meta(run_id, status="cancelled", finished_at=reg.now_iso(),
                        duration_s=round(time.time() - t_start, 3),
                        error=f"cancelled by user after {done} completed task(s)")
        _log(f"cancelled after {done} task(s); partial per-task events are kept")
        return "cancelled"
    except Exception as e:  # noqa: BLE001 — every failure is recorded verbatim
        tb = traceback.format_exc()
        reg.update_meta(run_id, status="failed", finished_at=reg.now_iso(),
                        duration_s=round(time.time() - t_start, 3), error=f"{type(e).__name__}: {e}")
        _log("FAILED with exception:\n" + tb)
        return "failed"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m studio.engine.experiment_runner <run_id>")
        sys.exit(2)
    status = execute(sys.argv[1])
    sys.exit(0 if status in ("completed", "cancelled") else 1)
