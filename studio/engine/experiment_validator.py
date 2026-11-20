"""Validation: configurations, immutable snapshots, results, and reproduction checks."""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import config_loader as cl

# Paper reference protocol per (dataset, architecture) — used only to *warn* when a
# configuration departs from the published setting, never to block it.
PAPER_PROTOCOL = {
    ("permuted_mnist", "mlp"): {"lr": 0.2, "batch": 128, "steps_per_task": 60, "n_tasks": 100,
                                "train_subset": 6000, "depth": 2, "diag_probe_size": 1000},
    ("rotated_mnist", "mlp"): {"lr": 0.2, "batch": 128, "steps_per_task": 60, "n_tasks": 100,
                               "train_subset": 6000, "test_subset": 2000, "hidden": 16, "depth": 2,
                               "diag_probe_size": 1000},
    ("rotated_mnist", "cnn"): {"lr": 0.1, "batch": 128, "steps_per_task": 60, "n_tasks": 100,
                               "train_subset": 6000, "test_subset": 2000, "base": 16, "fc": 64,
                               "diag_probe_size": 1000},
}
PAPER_METHOD_PARAMS = {
    "l2_init": {"l2_init_lambda": 1e-2}, "shrink_pert": {"shrink": 0.7, "perturb": 0.01},
    "cbp": {"cbp_reset_rate": 1e-2}, "dir_control": {"dir_coeff": 1.0},
    "dir_inward": {"dir_coeff": 0.6},
    "adaptive_norm_revive": {"anr_target_norm": 3.0, "anr_dormant_thresh": 1e-3, "anr_mode": "full"},
    "norm_project": {"norm_project_target": 6.0},
}


def _range_errors(block: str, spec: dict, values: dict) -> list[str]:
    errs = []
    for k, s in spec.items():
        v = values.get(k)
        if s["type"] == "choice":
            if v not in s["choices"]:
                errs.append(f"{block}.{k} must be one of {s['choices']} (got {v!r})")
            continue
        if isinstance(v, float) and not math.isfinite(v):
            errs.append(f"{block}.{k} must be finite")
            continue
        if "min" in s and v < s["min"]:
            errs.append(f"{block}.{k} = {v} is below the minimum {s['min']}")
        if "max" in s and v > s["max"]:
            errs.append(f"{block}.{k} = {v} exceeds the maximum {s['max']}")
    return errs


def validate_config(cfg: dict) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for a *normalized* configuration."""
    errors: list[str] = []
    warnings: list[str] = []
    ds, ar, me = cfg["dataset"]["name"], cfg["architecture"]["name"], cfg["method"]["name"]

    if ar not in cl.DATASETS[ds]["archs"]:
        errors.append("The CNN is only defined for Rotated MNIST in this project: permuting pixels "
                      "destroys the spatial structure convolutions rely on, and the repository never "
                      "ran CNN × Permuted MNIST.")
    if ar not in cl.METHODS[me]["archs"]:
        errors.append(f"{cl.METHODS[me]['label']} acts on individual MLP hidden units; "
                      "run_arm_cnn does not implement it (CBP/ANR are MLP-only).")

    errors += _range_errors("dataset", cl.DATASETS[ds]["params"], cfg["dataset"])
    errors += _range_errors("architecture", cl.ARCHITECTURES[ar]["params"], cfg["architecture"])
    errors += _range_errors(f"method.{me}", cl.METHODS[me]["params"], cfg["method"]["params"])
    errors += _range_errors("optimization", cl.OPTIMIZATION, cfg["optimization"])
    errors += _range_errors("evaluation", cl.EVALUATION, cfg["evaluation"])
    if not (0 <= cfg["seed"] <= 2**31 - 1):
        errors.append("seed must be in [0, 2^31 − 1]")
    if cfg["device"] not in cl.DEVICE_CHOICES:
        errors.append(f"device must be one of {cl.DEVICE_CHOICES}")
    if cfg["checkpoint_policy"] not in cl.CHECKPOINT_POLICIES:
        errors.append(f"checkpoint_policy must be one of {cl.CHECKPOINT_POLICIES}")

    test_size = cl.DATASETS[ds]["test_size"] or cfg["dataset"].get("test_subset")
    probe = cfg["evaluation"]["diag_probe_size"]
    if test_size and probe > test_size:
        warnings.append(f"diag_probe_size {probe} exceeds the test set ({test_size}); the code "
                        f"silently uses all {test_size} images.")
    if cfg["dataset"]["n_tasks"] > 100:
        warnings.append("More than 100 tasks is beyond the range studied in the paper.")
    if me == "dir_inward" and cfg["method"]["params"]["dir_coeff"] >= 0.8:
        warnings.append("γ ≥ 0.8 collapsed learning in the paper's radial sweep (dormancy → 100%).")

    # Departures from the published protocol (informational).
    ref = PAPER_PROTOCOL.get((ds, ar))
    if ref:
        flat = {**cfg["dataset"], **cfg["architecture"], **cfg["optimization"], **cfg["evaluation"]}
        diffs = [f"{k}: paper {v} → {flat[k]}" for k, v in ref.items() if k in flat and flat[k] != v]
        if diffs:
            warnings.append("Differs from the paper protocol (" + "; ".join(diffs) + ").")
    pm = PAPER_METHOD_PARAMS.get(me)
    if pm:
        diffs = [f"{k}: paper {v} → {cfg['method']['params'][k]}" for k, v in pm.items()
                 if cfg["method"]["params"].get(k) != v]
        if diffs:
            warnings.append("Method parameters differ from the paper arm (" + "; ".join(diffs) + ").")
    return errors, warnings


def verify_snapshot(run_path: Path) -> dict:
    """Check that a run's configuration snapshot still matches its recorded hash."""
    cfg_p, meta_p = run_path / "config.json", run_path / "meta.json"
    if not cfg_p.exists() or not meta_p.exists():
        return {"ok": False, "reason": "missing config.json or meta.json"}
    from .fsutil import read_json
    cfg, meta = read_json(cfg_p), read_json(meta_p)
    actual = cl.config_hash(cfg)
    ok = actual == meta.get("config_hash")
    return {"ok": ok, "recorded": meta.get("config_hash"), "actual": actual,
            "reason": None if ok else "configuration snapshot was modified after creation"}


def result_flags(summary: dict) -> list[str]:
    """Scientific sanity flags for a finished run (reported, never 'fixed')."""
    flags = []
    for k in ("final_weight_norm", "mean_acc", "final_eff_rank"):
        v = summary.get(k)
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            flags.append(f"{k} is not finite (diverged or undefined)")
    dev = summary.get("matched_max_dev")
    if summary.get("matched_step") == "applicable" and dev is not None and dev > 1e-4:
        flags.append(f"matched-step deviation {dev:.2e} exceeds the 1e-4 tolerance")
    return flags


REPRO_METRICS = ("mean_acc", "decay", "final_weight_norm", "final_dormant", "final_eff_rank")


def compare_to_legacy(summary: dict, legacy_row: dict, tol_exact: float = 1e-9,
                      tol_close: float = 1e-4) -> dict:
    """Compare a Studio run's summary with one per-seed row of a legacy artifact."""
    diffs = {}
    for k in REPRO_METRICS:
        a, b = summary.get(k), legacy_row.get(k)
        if a is None or b is None:
            continue
        diffs[k] = abs(float(a) - float(b))
    if not diffs:
        return {"status": "not_comparable", "diffs": {}}
    worst = max(diffs.values())
    status = ("reproduced_exact" if worst <= tol_exact
              else "reproduced_within_tolerance" if worst <= tol_close else "mismatch")
    return {"status": status, "max_abs_diff": worst, "diffs": diffs,
            "tolerances": {"exact": tol_exact, "close": tol_close}}
