"""Read-only index of the existing experiment artifacts in results/tables/.

Each released (experiment, arm) becomes a registry record. Values are copied
from the artifact; non-finite values become ``None`` (shown as "Not available"),
never 0. Configurations are reconstructed from the constants in the experiment
scripts (verified in docs/EXPERIMENTAL_STUDIO_AUDIT.md §3–§6) so that Studio runs can
be checked against them. Nothing in results/tables/ is ever written.
"""
from __future__ import annotations

import datetime as _dt
import functools
import json
import math
import re
from pathlib import Path

from . import config_loader as cl
from .paths import ROOT, TABLES_DIR

CATEGORIES = {
    "A": "Primary Permuted MNIST experiments",
    "B": "Norm-control strength (λ) sweep",
    "C": "Radial coefficient (γ) sweep",
    "D": "Network width experiments",
    "E": "Task geometry experiments",
    "F": "Adaptive-Norm-Revive ablations",
    "S": "Supplementary diagnostics",
    "P1": "Phase 1 (β₄ regression protocol — not comparable)",
    "X": "Superseded / incomplete",
}

METRIC_MAP = {  # artifact key -> Studio metric key
    "mean_acc": "mean_acc", "decay": "decay", "final_wl2": "final_weight_norm",
    "final_dormant": "final_dormant", "final_eff_rank": "final_eff_rank",
    "matched_max_dev": "matched_max_dev", "grad_align": "grad_align_mean",
    "radial_align": "radial_align_mean", "peak_acc": "peak_acc", "end_acc": "end_acc",
}


def _f(x):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _load(name: str) -> dict | None:
    p = TABLES_DIR / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _mtime(name: str) -> str | None:
    p = TABLES_DIR / name
    if not p.exists():
        return None
    return _dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")


def _metrics(agg: dict) -> dict:
    out = {}
    for src, dst in METRIC_MAP.items():
        if f"{src}_mean" in agg:
            out[dst] = {"mean": _f(agg.get(f"{src}_mean")), "std": _f(agg.get(f"{src}_std"))}
    return out


def _per_seed(rows: list[dict], seeds: list[int]) -> list[dict]:
    out = []
    for seed, r in zip(seeds, rows):
        row = {"seed": seed}
        for src, dst in METRIC_MAP.items():
            if src in r:
                row[dst] = _f(r[src])
        out.append(row)
    return out


def _arm_method(key: str) -> tuple[str, dict]:
    """Map an artifact arm key onto (method, params) of the Studio schema."""
    fixed = {
        "sgd": ("sgd", {}), "l2_init": ("l2_init", {"l2_init_lambda": 1e-2}),
        "shrink_pert": ("shrink_pert", {"shrink": 0.7, "perturb": 0.01}),
        "cbp": ("cbp", {"cbp_reset_rate": 1e-2}),
        "dir_control_c1.0": ("dir_control", {"dir_coeff": 1.0}),
        "dir_control": ("dir_control", {"dir_coeff": 1.0}),
        "dir_inward_g0.6": ("dir_inward", {"dir_coeff": 0.6}),
        "dir_inward": ("dir_inward", {"dir_coeff": 0.6}),
        "adaptive_norm_revive": ("adaptive_norm_revive",
                                 {"anr_target_norm": 3.0, "anr_dormant_thresh": 1e-3, "anr_mode": "full"}),
        "anr_full": ("adaptive_norm_revive",
                     {"anr_target_norm": 3.0, "anr_dormant_thresh": 1e-3, "anr_mode": "full"}),
        "anr_shrink_only": ("adaptive_norm_revive",
                            {"anr_target_norm": 3.0, "anr_dormant_thresh": 1e-3, "anr_mode": "shrink"}),
        "anr_revive_only": ("adaptive_norm_revive",
                            {"anr_target_norm": 3.0, "anr_dormant_thresh": 1e-3, "anr_mode": "revive"}),
        "norm_project": ("norm_project", {"norm_project_target": 6.0}),
        "norm_project_t6": ("norm_project", {"norm_project_target": 6.0}),
        "random_dir": ("random_dir", {}),
    }
    if key in fixed:
        return fixed[key]
    m = re.match(r"angle_c=([+-]?\d+\.\d+)", key)
    if m:
        return "angle", {"cos_target": float(m.group(1))}
    raise KeyError(key)


def _config(dataset: str, arch: dict, method: str, params: dict, lr: float, seed: int) -> dict:
    raw = {"dataset": {"name": dataset, "n_tasks": 100, "train_subset": 6000},
           "architecture": arch, "method": {"name": method, "params": params},
           "optimization": {"lr": lr, "batch": 128, "steps_per_task": 60},
           "evaluation": {"diag_probe_size": 1000}, "seed": seed}
    if dataset == "rotated_mnist":
        raw["dataset"]["test_subset"] = 2000
    return cl.normalize(raw)


MLP16 = {"name": "mlp", "hidden": 16, "depth": 2}
ARM_LABEL = {
    "sgd": "SGD", "l2_init": "L2-init", "shrink_pert": "Shrink & Perturb", "cbp": "Continual Backprop",
    "dir_control_c1.0": "Orthogonal Steering", "dir_control": "Orthogonal Steering",
    "dir_inward_g0.6": "Inward Steering (γ=0.6)", "dir_inward": "Inward Steering (γ=0.6)",
    "adaptive_norm_revive": "Adaptive-Norm-Revive", "anr_full": "ANR (shrink + revive)",
    "anr_shrink_only": "ANR shrink-only", "anr_revive_only": "ANR revive-only",
    "norm_project": "Norm Projection (r=6)", "norm_project_t6": "Norm Projection (r=6)",
    "random_dir": "Random Direction",
}


def _record(*, exp: str, key: str, label: str, category: str, artifact: str, script: str,
            paper_ref: str | None, cfg: dict, seeds: list[int], metrics: dict,
            per_seed: list[dict] | None, group: str, hardware: dict | None,
            notes: list[str] | None = None, status: str | None = None, extra: dict | None = None) -> dict:
    me = cl.METHODS[cfg["method"]["name"]]
    nm = metrics.get("final_weight_norm", {}).get("mean") if metrics else None
    if status is None:
        status = "completed"
        if metrics and "final_weight_norm" in metrics and nm is None:
            status = "completed_diverged"
    rec = {
        "id": f"legacy.{exp}.{key}",
        "source": "legacy_artifact",
        "name": f"{label} — {cl.describe(cfg).split(' · ', 1)[1]}",
        "arm_key": key, "arm_label": label,
        "category": category, "category_label": CATEGORIES[category],
        "experiment": exp, "script": script, "artifact": f"results/tables/{artifact}",
        "paper_ref": paper_ref, "status": status,
        "validation": {"status": "artifact", "detail": "released artifact; not independently "
                                                       "re-run by the Studio"},
        "config": cfg, "protocol_key": cl.protocol_key(cfg),
        "dataset": cfg["dataset"]["name"], "architecture": cfg["architecture"],
        "method": cfg["method"]["name"], "method_label": me["label"], "family": me["family"],
        "matched_step": me["matched_step"], "params": cfg["method"]["params"],
        "seeds": seeds, "n_seeds": len(seeds), "group": group,
        "metrics": metrics, "per_seed": per_seed, "has_per_task": False,
        "per_task_note": "Per-task trajectories were not persisted by this experiment script.",
        "hardware": hardware, "created_at": _mtime(artifact), "notes": notes or [],
    }
    if status == "completed_diverged":
        rec["notes"].append("Final weight norm is non-finite (diverged); excluded from correlations.")
    if extra:
        rec.update(extra)
    return rec


# --------------------------------------------------------------------------- #
def _seven_arm(exp, fname, script, category, group, paper_ref, dataset, arch, lr, agg_key="aggregate",
               per_seed_key=None):
    d = _load(fname)
    if d is None:
        return []
    seeds = d.get("seeds", cl.PAPER_SEEDS)
    out = []
    for key, agg in d[agg_key].items():
        method, params = _arm_method(key)
        cfg = _config(dataset, arch, method, params, lr, seeds[0])
        ps = _per_seed(d[per_seed_key][key], seeds) if per_seed_key and key in d.get(per_seed_key, {}) else None
        out.append(_record(exp=exp, key=key, label=ARM_LABEL.get(key, key), category=category,
                           artifact=fname, script=script, paper_ref=paper_ref, cfg=cfg, seeds=seeds,
                           metrics=_metrics(agg), per_seed=ps, group=group,
                           hardware=d.get("hardware")))
    return out


@functools.lru_cache(maxsize=1)
def build_index() -> tuple[dict, ...]:
    recs: list[dict] = []
    E = "experiments/"
    # A — primary (also the width-16 point of D and the permuted point of E)
    recs += _seven_arm("e20", "e20_summary.json", E + "e20_plasticity_stress.py", "A", "e20",
                       "Table I · Fig. 2 · Table II (h=16)", "permuted_mnist", MLP16, 0.2,
                       per_seed_key="per_seed_raw")
    # D — width
    recs += _seven_arm("e22_arch64", "e22_arch64_summary.json", E + "e22_reviewer_experiments.py", "D",
                       "e22_arch64", "Table II (h=64)", "permuted_mnist",
                       {"name": "mlp", "hidden": 64, "depth": 2}, 0.2)
    recs += _seven_arm("e24_arch128", "e24_arch128_summary.json", E + "e24_generalization.py", "D",
                       "e24_arch128", "Table II (h=128)", "permuted_mnist",
                       {"name": "mlp", "hidden": 128, "depth": 2}, 0.2)
    # E — task geometry
    recs += _seven_arm("e24_rotmnist", "e24_rotmnist_summary.json", E + "e24_generalization.py", "E",
                       "e24_rotmnist", "Table II (rotated MNIST, MLP)", "rotated_mnist", MLP16, 0.2)
    recs += _seven_arm("e25_cnn", "e25_cnn_summary.json", E + "e25_cnn.py", "E", "e25_cnn",
                       "Table II (rotated MNIST, CNN)", "rotated_mnist",
                       {"name": "cnn", "base": 16, "fc": 64}, 0.1, agg_key="arm_aggregate")

    # B — λ sweep
    d = _load("e22_lambda_summary.json")
    if d:
        for lam_s, agg in d["aggregate"].items():
            lam = float(lam_s)
            cfg = _config("permuted_mnist", MLP16, "l2_init", {"l2_init_lambda": lam}, 0.2, d["seeds"][0])
            recs.append(_record(exp="e22_lambda", key=f"lambda={lam_s}", label=f"L2-init λ={lam_s}",
                                category="B", artifact="e22_lambda_summary.json",
                                script=E + "e22_reviewer_experiments.py", paper_ref="Results — λ sweep",
                                cfg=cfg, seeds=d["seeds"], metrics=_metrics(agg), per_seed=None,
                                group="e22_lambda", hardware=None,
                                extra={"sweep": {"param": "l2_init_lambda", "value": lam}},
                                notes=["λ = 0 is functionally plain SGD (no penalty term)."] if lam == 0 else None))
    # C — radial sweep (per-seed raw available)
    d = _load("e27_radial_sweep_summary.json")
    if d:
        for g_s, agg in d["aggregate"].items():
            g = float(g_s)
            cfg = _config("permuted_mnist", MLP16, "dir_inward", {"dir_coeff": g}, 0.2, d["seeds"][0])
            recs.append(_record(exp="e27", key=f"gamma={g_s}", label=f"Inward Steering γ={g_s}",
                                category="C", artifact="e27_radial_sweep_summary.json",
                                script=E + "e27_radial_sweep.py", paper_ref="Fig. 3",
                                cfg=cfg, seeds=d["seeds"], metrics=_metrics(agg),
                                per_seed=_per_seed(d["per_seed_raw"][g_s], d["seeds"]),
                                group="e27_radial", hardware=d.get("hardware"),
                                extra={"sweep": {"param": "dir_coeff", "value": g}},
                                notes=(["γ = 0 is plain SGD (dir_inward_step returns the SGD step)."]
                                       if g == 0 else None)))
    # F — ANR ablation
    d = _load("e22_anr_ablation_summary.json")
    if d:
        for key, agg in d["aggregate"].items():
            method, params = _arm_method(key)
            cfg = _config("permuted_mnist", MLP16, method, params, 0.2, d["seeds"][0])
            recs.append(_record(exp="e22_anr", key=key, label=ARM_LABEL.get(key, key), category="F",
                                artifact="e22_anr_ablation_summary.json",
                                script=E + "e22_reviewer_experiments.py",
                                paper_ref="Results — ANR component ablation", cfg=cfg, seeds=d["seeds"],
                                metrics=_metrics(agg), per_seed=None, group="e22_anr", hardware=None))
    # S — supplementary: e23 extra arms, e26, cor1
    d = _load("e23_summary.json")
    if d:
        for key, agg in d["aggregate"].items():
            if key not in ("random_dir", "norm_project_t6") and not key.startswith("angle_c="):
                continue                     # the seven standard arms duplicate e20
            method, params = _arm_method(key)
            cfg = _config("permuted_mnist", MLP16, method, params, 0.2, d["seeds"][0])
            label = ARM_LABEL.get(key) or f"Angle-controlled cos={params.get('cos_target')}"
            recs.append(_record(exp="e23", key=key, label=label, category="S", artifact="e23_summary.json",
                                script=E + "e23_update_geometry.py", paper_ref=None, cfg=cfg,
                                seeds=d["seeds"], metrics=_metrics(agg), per_seed=None, group="e23",
                                hardware=d.get("hardware")))
    d = _load("e26_summary.json")
    if d:
        for key, alt in d["alt_metrics"].items():
            method, params = _arm_method({"dir_control": "dir_control_c1.0",
                                          "dir_inward": "dir_inward_g0.6"}.get(key, key))
            cfg = _config("permuted_mnist", MLP16, method, params, 0.2, d["seeds"][0])
            metrics = {"decay": {"mean": _f(alt.get("decay_peak_tail")), "std": None},
                       "mean_acc": {"mean": _f(alt.get("mean_acc")), "std": None},
                       "last_task_acc": {"mean": _f(alt.get("last_task_acc")), "std": None},
                       "late10_acc": {"mean": _f(alt.get("late10_acc")), "std": None},
                       "slope_per_task": {"mean": _f(alt.get("slope_per_task")), "std": None}}
            recs.append(_record(exp="e26", key=key, label=ARM_LABEL.get(key, key), category="S",
                                artifact="e26_summary.json", script=E + "e26_diagnostics.py",
                                paper_ref=None, cfg=cfg, seeds=d["seeds"], metrics=metrics,
                                per_seed=None, group="e26", hardware=None,
                                extra={"layer_norms": {"labels": d["layer_labels"],
                                                       "init": d["init_layer_norms"],
                                                       "final": d["final_layer_norms"].get(key)}}))
    d = _load("cor1_orthogonal_audit.json")
    if d:
        cfg = _config("permuted_mnist", MLP16, "dir_control", {"dir_coeff": 1.0}, 0.2, d["seeds"][0])
        per = [{"seed": r["seed"], "endpoint_delta": r["endpoint_delta"], "sum_eta_sq": r["sum_eta_sq"],
                "cor1_relative_error": r["rel_err"], "radial_align_mean": r["radial_align_mean"],
                "mean_acc": r["mean_acc"], "final_weight_norm": r["backbone_norm_final"],
                "initial_global_norm": r["theta0_norm"], "final_global_norm": r["thetaT_norm"]}
               for r in d["per_seed"]]
        recs.append(_record(exp="cor1", key="orthogonal", label="Cor. 1 audit (orthogonal arm)",
                            category="S", artifact="cor1_orthogonal_audit.json",
                            script="(one-off audit script, not included in the repository)",
                            paper_ref="Results — Cor. 1 audit sentence", cfg=cfg, seeds=d["seeds"],
                            metrics={"cor1_relative_error": {"mean": d["rel_err_mean"], "std": None},
                                     "radial_align_mean": {"mean": d["radial_align_mean_over_seeds"],
                                                           "std": None}},
                            per_seed=per, group="cor1", hardware=d.get("hardware"),
                            extra={"cor1": {"endpoint_delta_mean": d["endpoint_delta_mean"],
                                            "sum_eta_sq_mean": d["sum_eta_sq_mean"],
                                            "rel_err_max": d["rel_err_max"]}}))
    # X — superseded / incomplete
    d = _load("e21_summary.json")
    if d:
        recs.append({"id": "legacy.e21.pilot", "source": "legacy_artifact",
                     "name": "λ-sweep pilot (3 seeds, 5 values)", "category": "X",
                     "category_label": CATEGORIES["X"], "experiment": "e21",
                     "script": E + "e21_norm_strength_sweep.py", "artifact": "results/tables/e21_summary.json",
                     "paper_ref": None, "status": "superseded",
                     "validation": {"status": "artifact", "detail": "superseded by e22-B (5 seeds)"},
                     "dataset": "permuted_mnist", "architecture": MLP16, "method": "l2_init",
                     "method_label": "L2-init", "family": "magnitude", "matched_step": "na",
                     "seeds": d["seeds"], "n_seeds": len(d["seeds"]), "group": "e21", "metrics": {},
                     "per_seed": None, "has_per_task": False, "created_at": _mtime("e21_summary.json"),
                     "notes": ["Superseded pilot; not used by the paper."]})
    if not (TABLES_DIR / "e18_summary.json").exists() and (ROOT / "experiments" / "e18_scale_mechanism.py").exists():
        recs.append({"id": "legacy.e18.scale_mechanism", "source": "legacy_artifact",
                     "name": "Scale/mechanism check (larger CNN, Split-CIFAR-10)", "category": "X",
                     "category_label": CATEGORIES["X"], "experiment": "e18",
                     "script": E + "e18_scale_mechanism.py", "artifact": None, "paper_ref": None,
                     "status": "incomplete",
                     "validation": {"status": "missing", "detail": "script exists; no result artifact"},
                     "dataset": "split_cifar10", "architecture": {"name": "phase1_cnn"}, "method": None,
                     "method_label": "Phase-1 predictive theory + controller", "family": "other",
                     "matched_step": "na", "seeds": [], "n_seeds": 0, "group": "e18", "metrics": {},
                     "per_seed": None, "has_per_task": False, "created_at": None,
                     "notes": ["The only experiment that was never completed (no result artifact). "
                               "Phase 1 protocol; not part of the paper. "
                               "Compute requirement not documented; not launched by the Studio."]})
    # P1 — Phase-1 artifacts (listed for completeness; different protocol)
    for p in sorted(TABLES_DIR.glob("e*_summary.json")):
        m = re.match(r"e(\d+)", p.name)
        if not m or int(m.group(1)) >= 20:
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        verdict = d.get("verdict") or d.get("overall_verdict") or d.get("method_verdict")
        if isinstance(verdict, dict):
            verdict = verdict.get("overall") or json.dumps(verdict)[:80]
        exp = p.stem.replace("_summary", "")
        recs.append({"id": f"legacy.{exp}", "source": "legacy_artifact", "name": f"Phase 1 · {exp}",
                     "category": "P1", "category_label": CATEGORIES["P1"], "experiment": exp,
                     "script": None, "artifact": f"results/tables/{p.name}", "paper_ref": None,
                     "status": "completed", "validation": {"status": "artifact", "detail": "Phase-1 artifact"},
                     "dataset": None, "architecture": None, "method": None, "method_label": "β₄ regression protocol",
                     "family": "other", "matched_step": "na", "seeds": [], "n_seeds": 0, "group": "phase1",
                     "metrics": {}, "per_seed": None, "has_per_task": False,
                     "created_at": _mtime(p.name), "verdict": verdict,
                     "notes": ["Different protocol (future-trainability-damage regression); "
                               "not comparable with Phase-2 records."]})
    return tuple(recs)


def index() -> list[dict]:
    return list(build_index())


def get(record_id: str) -> dict | None:
    for r in build_index():
        if r["id"] == record_id:
            return r
    return None


def per_seed_reference(cfg: dict) -> tuple[dict, dict] | None:
    """Find a legacy per-seed row whose configuration equals ``cfg`` (same protocol,
    method, parameters and seed). Returns (record, row) or None."""
    key, method, params, seed = cl.protocol_key(cfg), cfg["method"]["name"], cfg["method"]["params"], cfg["seed"]
    for r in build_index():
        if (r.get("per_seed") and r.get("protocol_key") == key and r.get("method") == method
                and r.get("params") == params and seed in r.get("seeds", [])
                and r["experiment"] in ("e20", "e27")):
            row = next(x for x in r["per_seed"] if x["seed"] == seed)
            return r, row
    return None
