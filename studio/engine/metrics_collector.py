"""Metrics collection for Studio runs.

Consumes the read-only events emitted by ``run_arm`` / ``run_arm_cnn`` via
``progress_cb`` and persists them. Every value written here is either emitted by
the training code or a documented arithmetic transform of emitted values:

* radial component of an applied step   = ‖Δθ'‖ · cos(Δθ', θ)
* tangential component                  = ‖Δθ'‖ · sqrt(1 − cos²(Δθ', θ))
* matched-step deviation (per task)     = max |‖Δθ'‖ − η| / η
* plasticity decay (paper definition)   = max_t a_t − mean(a_t over the last ⌈10 %⌉ tasks)
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import fsutil


class CancelledRun(Exception):
    """Raised from the progress hook when a cancellation was requested."""


# --------------------------------------------------------------------------- #
# Metric definitions                                                           #
# --------------------------------------------------------------------------- #
def peak_to_tail_decay(task_acc: list[float]) -> tuple[float, float, float]:
    """(peak, tail mean, decay) exactly as e20's ``peak_end_decay``."""
    ta = np.asarray(task_acc, dtype=float)
    peak = float(ta.max())
    end = float(ta[-max(1, len(ta) // 10):].mean())
    return peak, end, peak - end


def internal_first_last_decay(task_acc: list[float]) -> float:
    """``run_arm``'s own ``plasticity_decay`` (first 20 % − last 20 %). NOT the paper metric."""
    k = max(1, len(task_acc) // 5)
    return float(np.mean(task_acc[:k]) - np.mean(task_acc[-k:]))


def finite_or_none(x):
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def task_record(ev: dict) -> dict:
    """Per-task record from one 'task' event (all inputs emitted by the training loop)."""
    an = np.asarray(ev["applied_norms"], dtype=float)
    rn = np.asarray(ev["ref_norms"], dtype=float)
    ga = np.asarray(ev["grad_aligns"], dtype=float)
    rc = np.asarray(ev["radial_aligns"], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        dev = np.abs(an - rn) / (rn + 1e-12)
        radial = an * rc
        tang = an * np.sqrt(np.clip(1.0 - rc ** 2, 0.0, None))
    nan_mean = lambda a: finite_or_none(np.nanmean(a)) if a.size and np.isfinite(a).any() else None
    nan_max = lambda a: finite_or_none(np.nanmax(a)) if a.size and np.isfinite(a).any() else None
    return {
        "task": int(ev["task"]),
        "acc": finite_or_none(ev["acc"]),
        "online_acc": finite_or_none(ev.get("online_acc")),
        "weight_norm": finite_or_none(ev["weight_l2"]),
        "dormant": finite_or_none(ev["dormant"]),
        "eff_rank": finite_or_none(ev["eff_rank"]),
        "steps": int(an.size),
        "step_norm_mean": nan_mean(an),            # update magnitude ‖Δθ'‖
        "sgd_norm_mean": nan_mean(rn),             # η = ‖Δθ_SGD‖
        "matched_dev_max": nan_max(dev),
        "grad_align_mean": nan_mean(ga),           # cos(Δθ', Δθ)
        "radial_cos_mean": nan_mean(rc),           # cos(Δθ', θ)
        "radial_component_mean": nan_mean(radial),
        "tangential_component_mean": nan_mean(tang),
        "radial_energy_frac": nan_mean(rc ** 2),   # share of ‖Δθ'‖² spent radially
    }


# --------------------------------------------------------------------------- #
# Collector                                                                    #
# --------------------------------------------------------------------------- #
class MetricsCollector:
    """Turns progress events into files inside a run directory.

    events.jsonl   one JSON line per finished task (+ a final summary line)
    progress.json  latest step/task position (atomically replaced, throttled)
    CANCEL         if this file appears, the next hook call raises CancelledRun
    """

    def __init__(self, run_path: Path, n_tasks: int, steps_per_task: int,
                 on_final_task: Callable | None = None, progress_interval: float = 0.5,
                 cancel_check_every: int = 5):
        self.run_path = Path(run_path)
        self.n_tasks, self.steps_per_task = n_tasks, steps_per_task
        self.events_path = self.run_path / "events.jsonl"
        self.progress_path = self.run_path / "progress.json"
        self.cancel_path = self.run_path / "CANCEL"
        self.on_final_task = on_final_task
        self.progress_interval = progress_interval
        self.cancel_check_every = cancel_check_every
        self.t0 = time.time()
        self._last_write = 0.0
        self._step_counter = 0
        self.accs: list[float] = []
        self.records: list[dict] = []
        self._gpu = None
        try:
            import torch
            self._gpu = torch.cuda if torch.cuda.is_available() else None
        except Exception:  # pragma: no cover
            self._gpu = None

    # -- helpers ----------------------------------------------------------------
    def _write_progress(self, task: int, step: int, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_write < self.progress_interval:
            return
        self._last_write = now
        state = {"task": task, "n_tasks": self.n_tasks, "step": step,
                 "steps_per_task": self.steps_per_task,
                 "global_step": task * self.steps_per_task + step,
                 "total_steps": self.n_tasks * self.steps_per_task,
                 "elapsed_s": round(now - self.t0, 3), "tasks_done": len(self.records),
                 "updated_at": now}
        if self._gpu is not None:
            state["gpu_mem_allocated_mb"] = round(self._gpu.memory_allocated() / 2**20, 1)
            state["gpu_mem_peak_mb"] = round(self._gpu.max_memory_allocated() / 2**20, 1)
        # Advisory file: a reader may hold it open (Windows); never let that fail the run.
        fsutil.atomic_write(self.progress_path, json.dumps(state), attempts=5)

    def _check_cancel(self) -> None:
        if self.cancel_path.exists():
            raise CancelledRun("cancellation requested")

    # -- the hook passed to run_arm ---------------------------------------------
    def __call__(self, ev: dict) -> None:
        if ev["type"] == "step":
            self._step_counter += 1
            if self._step_counter % self.cancel_check_every == 0:
                self._check_cancel()
            self._write_progress(ev["task"], ev["step"])
            return
        if ev["type"] == "task":
            rec = task_record(ev)
            self.accs.append(rec["acc"] if rec["acc"] is not None else float("nan"))
            peak, end, dec = peak_to_tail_decay(self.accs)
            rec["decay_running_estimate"] = finite_or_none(dec)   # labelled estimate
            rec["elapsed_s"] = round(time.time() - self.t0, 3)
            if self._gpu is not None:
                rec["gpu_mem_peak_mb"] = round(self._gpu.max_memory_allocated() / 2**20, 1)
            self.records.append(rec)
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"type": "task", **rec}) + "\n")
            self._write_progress(rec["task"], self.steps_per_task, force=True)
            if rec["task"] == self.n_tasks - 1 and self.on_final_task is not None:
                self.on_final_task(ev["model"])
            self._check_cancel()


def summarize(result: dict, records: list[dict], cfg: dict, matched_step: str) -> dict:
    """Run-level summary in the same metric vocabulary as the legacy artifacts."""
    acc = result["task_test_acc"]
    peak, end, decay = peak_to_tail_decay(acc)
    s = {
        "mean_acc": finite_or_none(np.mean(acc)),
        "peak_acc": finite_or_none(peak),
        "end_acc": finite_or_none(end),
        "decay": finite_or_none(decay),                                   # paper metric
        "decay_first_last_20pct": finite_or_none(internal_first_last_decay(acc)),
        "final_weight_norm": finite_or_none(result["diag_weight_l2"][-1]),
        "initial_weight_norm_task0": finite_or_none(result["diag_weight_l2"][0]),
        "final_dormant": finite_or_none(result["diag_dormant"][-1]),
        "final_eff_rank": finite_or_none(result["diag_eff_rank"][-1]),
        "matched_max_dev": finite_or_none(result.get("matched_max_dev")),
        "matched_step": matched_step,
        "grad_align_mean": finite_or_none(result.get("grad_align_mean")),
        "radial_align_mean": finite_or_none(result.get("radial_align_mean")),
        "n_tasks_completed": len(acc),
        "weight_norm_definition": ("backbone (hidden layers incl. biases, excl. head)"
                                   if cfg["architecture"]["name"] == "mlp" else "all parameters"),
    }
    for k in ("theta0_sq", "thetaT_sq", "sum_applied_sq"):      # MLP Cor. 1 audit fields
        if k in result:
            s[k] = finite_or_none(result[k])
    if "theta0_sq" in s and s.get("thetaT_sq") is not None and s.get("sum_applied_sq"):
        delta = s["thetaT_sq"] - s["theta0_sq"]
        s["cor1_relative_error"] = (abs(delta - s["sum_applied_sq"]) / abs(delta)
                                    if delta else None)
    s["per_task"] = {
        "acc": [finite_or_none(x) for x in acc],
        "weight_norm": [finite_or_none(x) for x in result["diag_weight_l2"]],
        "dormant": [finite_or_none(x) for x in result["diag_dormant"]],
        "eff_rank": [finite_or_none(x) for x in result["diag_eff_rank"]],
        "step_norm_mean": [r["step_norm_mean"] for r in records],
        "matched_dev_max": [r["matched_dev_max"] for r in records],
        "grad_align_mean": [r["grad_align_mean"] for r in records],
        "radial_component_mean": [r["radial_component_mean"] for r in records],
        "tangential_component_mean": [r["tangential_component_mean"] for r in records],
        "radial_energy_frac": [r["radial_energy_frac"] for r in records],
    }
    return s
