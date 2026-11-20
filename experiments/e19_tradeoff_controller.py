"""e19 — Stability–plasticity dial + sign-reversed plasticity-preserving direction.

Pivots 1 and 2 executed in the FIXED (task-incremental, multi-head) regime.

  Pivot 1 (D-043): sweep controller coefficient c ∈ {-1,-0.75,...,+1} along the
    state-dependent û_hazard estimated per step from auxiliary probes. Trace the
    (BWT, probe FTD, accuracy) frontier at strictly matched ‖Δθ‖ (max dev ≤ 2%
    by construction). Compare to SGD, EWC, replay, CBP.

  Pivot 2 (D-043): among the c<0 sweep points (steer TOWARD û_hazard — the
    sign-reversed direction that was untested in e17), verify that at least one
    setting strictly reduces final probe FTD vs SGD without collapsing mean acc.

Sanity gate (D-043, W7 fix): SGD baseline must reach:
   mean final acc ≥ E19_TRADEOFF.sanity_mnist_acc (0.60)  on Split-MNIST
   mean final acc ≥ E19_TRADEOFF.sanity_cifar_acc (0.35)  on Split-CIFAR-10
   at least sanity_min_tasks_above_chance (3) of 5 tasks retain >= 0.55 acc.

Pipeline is HALTED (no controller comparison logged as positive) if the sanity
gate fails — this is what makes the null credible.
"""
from __future__ import annotations
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from configs.preregistration import (
    PRIMARY_SEED, TIL_REGIME, E19_TRADEOFF, STRICT_MATCH,
)
from src.seeds import seed_everything
from src.device import pick_device, env_summary
from src.cl_harness import (
    prepare_mnist_til, prepare_cifar10_til,
    build_mnist_model, build_cifar_model,
    build_permuted_probes,
    train_til_sequence, sanity_gate,
    TILAdaptiveDirectionalController,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_TABLES = os.path.join(ROOT, "results", "tables")
RESULTS_FIGS = os.path.join(ROOT, "results", "figures")


def _stringify(o):
    if isinstance(o, dict): return {k: _stringify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_stringify(v) for v in o]
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    return o


# ---------------------------------------------------------------------------
# Dataset drivers
# ---------------------------------------------------------------------------

def _prep_mnist(seed):
    tasks = prepare_mnist_til("Deep Learning Dataset1")
    pool_x = torch.cat([t["train_x"] for t in tasks], dim=0)
    # For probe labels, we use "original class % 2" – but tasks already carry
    # remapped 0/1 labels, so the pool label vector is a per-sample task-local
    # 0/1. That's fine — probes need 2-class labels for the fresh 2-way head.
    pool_y = torch.cat([t["train_y"] for t in tasks], dim=0)
    probes = build_permuted_probes(pool_x, pool_y, TIL_REGIME.n_probes,
                                     seed=seed + 10_000,
                                     subset=TIL_REGIME.probe_subset,
                                     n_classes=2)
    n_tasks = len(tasks)
    make_model = lambda: build_mnist_model(TIL_REGIME.mnist_hidden,
                                             TIL_REGIME.mnist_depth,
                                             n_tasks=n_tasks,
                                             n_classes_per_task=2)
    aux_pool_x = pool_x; aux_pool_y = pool_y
    hp = dict(lr=TIL_REGIME.mnist_lr, epochs=TIL_REGIME.mnist_epochs_per_task,
              batch=TIL_REGIME.mnist_batch)
    sanity = dict(min_mean_acc=E19_TRADEOFF.sanity_mnist_acc,
                   min_tasks_above_chance=E19_TRADEOFF.sanity_min_tasks_above_chance,
                   task_chance=E19_TRADEOFF.sanity_task_chance)
    return tasks, probes, aux_pool_x, aux_pool_y, make_model, hp, sanity


def _prep_cifar10(seed):
    tasks = prepare_cifar10_til("cifar-10")
    pool_x = torch.cat([t["train_x"] for t in tasks], dim=0)
    pool_y = torch.cat([t["train_y"] for t in tasks], dim=0)
    probes = build_permuted_probes(pool_x, pool_y, TIL_REGIME.n_probes,
                                     seed=seed + 10_000,
                                     subset=TIL_REGIME.probe_subset,
                                     n_classes=2)
    n_tasks = len(tasks)
    make_model = lambda: build_cifar_model(TIL_REGIME.cifar_cnn_base,
                                             TIL_REGIME.cifar_cnn_fc,
                                             n_tasks=n_tasks,
                                             n_classes_per_task=2)
    hp = dict(lr=TIL_REGIME.cifar_lr, epochs=TIL_REGIME.cifar_epochs_per_task,
              batch=TIL_REGIME.cifar_batch)
    sanity = dict(min_mean_acc=E19_TRADEOFF.sanity_cifar_acc,
                   min_tasks_above_chance=E19_TRADEOFF.sanity_min_tasks_above_chance,
                   task_chance=E19_TRADEOFF.sanity_task_chance)
    return tasks, probes, pool_x, pool_y, make_model, hp, sanity


DATASET_PREP = {"split_mnist": _prep_mnist, "split_cifar10": _prep_cifar10}


def run_dataset(name: str, seed: int, device: torch.device,
                 baselines_only: bool = False,
                 force_controller: bool = False) -> dict:
    print(f"\n{'=' * 68}\n[E19] dataset={name} seed={seed} device={device}\n{'=' * 68}")
    prep = DATASET_PREP[name]
    tasks, probes, aux_x, aux_y, make_model, hp, sanity = prep(seed)
    held_out = tuple(range(len(tasks)))  # any task index — used only for sem-aux fallback

    seed_everything(seed + 7000)
    init_state = {k: v.clone() for k, v in make_model().state_dict().items()}

    def fresh():
        m = make_model()
        m.load_state_dict({k: v.clone() for k, v in init_state.items()})
        return m

    arms: dict = {}

    # ---- Baselines ----
    for method in E19_TRADEOFF.baselines:
        t0 = time.time()
        print(f"\n[E19:{name} seed={seed}] --- baseline: {method} ---")
        m = fresh()
        r = train_til_sequence(
            m, tasks, method=method,
            lr=hp["lr"], epochs=hp["epochs"], batch=hp["batch"],
            seed=seed, device=device,
            probes=probes,
            probe_steps=TIL_REGIME.probe_steps,
            probe_lr=TIL_REGIME.probe_lr,
            probe_batch=TIL_REGIME.probe_batch,
            ewc_lambda=E19_TRADEOFF.ewc_lambda,
            ewc_fisher_batch=E19_TRADEOFF.ewc_fisher_batch,
            replay_memory_per_task=E19_TRADEOFF.replay_memory_per_task,
            replay_mix_ratio=E19_TRADEOFF.replay_mix_ratio,
            cbp_reset_rate=E19_TRADEOFF.cbp_reset_rate,
            cbp_utility_ema=E19_TRADEOFF.cbp_utility_ema,
            verbose=True,
        )
        arms[method] = r
        print(f"[E19:{name}] {method}: mean_final_acc={r['mean_final_acc']:.3f} "
              f"BWT={r['BWT']:.3f} probeFTD_final={r['per_task_probe_ftd'][-1]:.4f} "
              f"max_dev={r['matched_magnitude']['max_deviation']:.4f}  ({time.time()-t0:.1f}s)")

    # Sanity gate on SGD
    sgd_gate = sanity_gate(arms["sgd"], **sanity)
    print(f"\n[E19:{name}] SGD sanity gate: {sgd_gate}")
    if not sgd_gate["passes"]:
        print(f"[E19:{name}] REGIME FAIL — SGD did not learn. Skipping controller.")

    # ---- Controller sweep (Pivot 1 + Pivot 2) ----
    should_run_controller = (not baselines_only) and (
        sgd_gate["passes"] or force_controller)
    if not baselines_only and not sgd_gate["passes"] and force_controller:
        print(f"[E19:{name}] sanity gate FAILED but --force_controller set — "
              f"running sweep with a caveat.")
    if should_run_controller:
        # A shared controller factory for consistency across c
        # `held_out_classes` is only used for the semantic-aux fallback; for
        # TIL with per-task 2-way labels we rely primarily on permuted-input
        # aux probes.
        for c in E19_TRADEOFF.coefficient_sweep:
            arm_name = f"controller_c={c:+.2f}"
            t0 = time.time()
            print(f"\n[E19:{name} seed={seed}] --- {arm_name} ---")
            ctrl = TILAdaptiveDirectionalController(
                n_permuted_aux=E19_TRADEOFF.n_permuted_aux,
                n_semantic_aux=0,   # rely on permuted-only for TIL (labels are 0/1 per task)
                reestimate_every=E19_TRADEOFF.reestimate_every,
                projection_alpha=1.0,  # unused; two-sided coefficient is passed separately
                top_k=E19_TRADEOFF.top_k_project,
                held_out_classes=None,
                aux_pool_x=None,
                aux_pool_y=None,
                seed=seed,
                batch=hp["batch"],
            )
            m = fresh()
            r = train_til_sequence(
                m, tasks, method="controller",
                lr=hp["lr"], epochs=hp["epochs"], batch=hp["batch"],
                seed=seed, device=device,
                probes=probes,
                probe_steps=TIL_REGIME.probe_steps,
                probe_lr=TIL_REGIME.probe_lr,
                probe_batch=TIL_REGIME.probe_batch,
                controller=ctrl, controller_coeff=float(c),
                verbose=True,
            )
            arms[arm_name] = r
            print(f"[E19:{name}] {arm_name}: mean_final_acc={r['mean_final_acc']:.3f} "
                  f"BWT={r['BWT']:.3f} probeFTD_final={r['per_task_probe_ftd'][-1]:.4f} "
                  f"max_dev={r['matched_magnitude']['max_deviation']:.4f}  ({time.time()-t0:.1f}s)")

    return {
        "dataset": name,
        "seed": seed,
        "sanity_gate": sgd_gate,
        "arms": {k: {
            "mean_final_acc": v["mean_final_acc"],
            "BWT": v["BWT"],
            "per_task_acc": v["per_task_acc"],
            "per_task_probe_ftd": v["per_task_probe_ftd"],
            "final_accs": v["final_accs"],
            "matched_magnitude": v["matched_magnitude"],
            "controller_coeff": v["controller_coeff"],
        } for k, v in arms.items()},
    }


# ---------------------------------------------------------------------------
# Aggregation & verdict
# ---------------------------------------------------------------------------

def aggregate(runs: list) -> dict:
    """Average per-arm across seeds within a dataset; compute Pareto set and
    Pivot 1 / Pivot 2 verdicts."""
    by_ds = {}
    for r in runs:
        by_ds.setdefault(r["dataset"], []).append(r)

    ds_results = {}
    for ds, ds_runs in by_ds.items():
        if not ds_runs[0]["arms"]:
            ds_results[ds] = {"verdict": "SKIPPED_REGIME_FAIL"}
            continue
        arm_names = list(ds_runs[0]["arms"].keys())
        agg = {}
        for a in arm_names:
            pfs, accs, bwts, max_devs = [], [], [], []
            for r in ds_runs:
                if a not in r["arms"]: continue
                v = r["arms"][a]
                pfs.append(v["per_task_probe_ftd"][-1])
                accs.append(v["mean_final_acc"])
                bwts.append(v["BWT"])
                max_devs.append(v["matched_magnitude"]["max_deviation"])
            if not pfs: continue
            agg[a] = {
                "ftd_mean": float(np.mean(pfs)),
                "ftd_std": float(np.std(pfs)),
                "acc_mean": float(np.mean(accs)),
                "acc_std": float(np.std(accs)),
                "bwt_mean": float(np.mean(bwts)),
                "bwt_std": float(np.std(bwts)),
                "max_dev_max": float(np.max(max_devs)),
            }
        # Pivot 2 (sign-reversed FTD reduction vs SGD, at matched mag)
        sgd_ftd = agg.get("sgd", {}).get("ftd_mean", float("inf"))
        sgd_acc = agg.get("sgd", {}).get("acc_mean", 0.0)
        p2 = {"passes": False, "operating_points": []}
        for a, v in agg.items():
            if not a.startswith("controller_c=-"): continue
            coeff = float(a.split("=")[1])
            if v["max_dev_max"] > STRICT_MATCH.per_step_tol: continue
            ftd_red_raw = sgd_ftd - v["ftd_mean"]
            ftd_red_norm = ftd_red_raw / max(sgd_ftd, 1e-12)
            acc_drop = sgd_acc - v["acc_mean"]
            passes = (ftd_red_raw >= E19_TRADEOFF.p2_min_ftd_reduction_raw and
                      ftd_red_norm >= E19_TRADEOFF.p2_min_ftd_reduction_norm and
                      acc_drop <= E19_TRADEOFF.p2_max_acc_drop)
            p2["operating_points"].append({
                "arm": a, "coeff": coeff,
                "ftd_reduction_raw": float(ftd_red_raw),
                "ftd_reduction_norm": float(ftd_red_norm),
                "acc_drop": float(acc_drop),
                "passes": bool(passes),
            })
            if passes: p2["passes"] = True

        # Pivot 1 (Pareto dominates OR matches EWC)
        p1 = {"passes": False, "operating_points": []}
        ewc = agg.get("ewc")
        for a, v in agg.items():
            if not a.startswith("controller_c="): continue
            if v["max_dev_max"] > STRICT_MATCH.per_step_tol: continue
            # dominates SGD
            dom_sgd = (v["ftd_mean"] <= sgd_ftd - 1e-6
                        and v["acc_mean"] >= sgd_acc - 1e-6
                        and (v["ftd_mean"] < sgd_ftd or v["acc_mean"] > sgd_acc))
            dom_ewc = False
            if ewc is not None:
                dom_ewc = (v["ftd_mean"] <= ewc["ftd_mean"] - 1e-6
                            and v["acc_mean"] >= ewc["acc_mean"] - 1e-6
                            and (v["ftd_mean"] < ewc["ftd_mean"]
                                 or v["acc_mean"] > ewc["acc_mean"]))
            if dom_sgd or dom_ewc:
                p1["operating_points"].append({
                    "arm": a, "dominates_sgd": bool(dom_sgd),
                    "dominates_ewc": bool(dom_ewc),
                    "ftd": v["ftd_mean"], "acc": v["acc_mean"], "bwt": v["bwt_mean"],
                })
                if dom_ewc: p1["passes"] = True   # main-track bar

        ds_results[ds] = {
            "per_arm_agg": agg,
            "pivot_1_dominates_ewc": p1,
            "pivot_2_ftd_reduction": p2,
            "verdict": ("POSITIVE" if (p1["passes"] or p2["passes"]) else "NULL"),
        }
    return ds_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_dataset(ds_name: str, agg: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.2))
    if "per_arm_agg" not in agg: return
    for arm, v in agg["per_arm_agg"].items():
        if arm.startswith("controller_"):
            ax.scatter(v["ftd_mean"], v["acc_mean"], s=70,
                        edgecolor="black", linewidth=0.5, alpha=0.9,
                        label=arm, marker="o", color="#3E5C76")
        else:
            marker = {"sgd": "s", "ewc": "D", "replay": "^", "cbp": "P"}.get(arm, "o")
            ax.scatter(v["ftd_mean"], v["acc_mean"], s=100,
                        edgecolor="black", linewidth=0.8, marker=marker,
                        label=arm, color="#B07D48", alpha=0.9)
    ax.set_xlabel("Final probe FTD (lower = better plasticity)")
    ax.set_ylabel("Mean final accuracy (higher = better)")
    ax.set_title(f"E19 {ds_name}: controller vs baselines (TIL, matched ‖Δθ‖)")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                    choices=["split_mnist", "split_cifar10"],
                    default=list(E19_TRADEOFF.datasets))
    p.add_argument("--seeds", nargs="+", type=int,
                    default=[PRIMARY_SEED, PRIMARY_SEED + 1, PRIMARY_SEED + 2])
    p.add_argument("--device", default="auto")
    p.add_argument("--baselines_only", action="store_true",
                    help="Only run baselines (SGD sanity check).")
    p.add_argument("--force_controller", action="store_true",
                    help="Run the controller sweep even if the sanity gate "
                         "fails. Sanity result is still logged.")
    args = p.parse_args()

    device, profile = pick_device(args.device)
    print(f"[E19] device={device} profile={profile}")

    out = {"hardware": env_summary(), "runs": []}
    for ds in args.datasets:
        for s in args.seeds:
            # MNIST on CPU (small MLP); CIFAR on GPU
            dev = torch.device("cpu") if ds == "split_mnist" else device
            r = run_dataset(ds, s, dev,
                             baselines_only=args.baselines_only,
                             force_controller=args.force_controller)
            out["runs"].append(r)

    out["aggregated"] = aggregate(out["runs"])

    # Plots
    for ds, agg in out["aggregated"].items():
        if "per_arm_agg" in agg:
            plot_dataset(ds, agg,
                          os.path.join(RESULTS_FIGS, f"e19_{ds}_pareto.png"))

    any_pos = any(v.get("verdict") == "POSITIVE"
                   for v in out["aggregated"].values())
    out["overall_verdict"] = "POSITIVE_MAIN_TRACK" if any_pos else "DEFINITIVE_NULL"

    os.makedirs(RESULTS_TABLES, exist_ok=True)
    with open(os.path.join(RESULTS_TABLES, "e19_summary.json"), "w") as f:
        json.dump(_stringify(out), f, indent=2)

    print(f"\n[E19] OVERALL VERDICT: {out['overall_verdict']}")
    for ds, agg in out["aggregated"].items():
        v = agg.get("verdict", "?")
        p1_pass = agg.get("pivot_1_dominates_ewc", {}).get("passes", False)
        p2_pass = agg.get("pivot_2_ftd_reduction", {}).get("passes", False)
        print(f"  {ds}: {v}  (Pivot1_beats_EWC={p1_pass}  Pivot2_FTD_reduces={p2_pass})")


if __name__ == "__main__":
    main()
