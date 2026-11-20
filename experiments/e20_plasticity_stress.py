"""e20 — Plasticity-stress regime: mechanism + impossibility + method attempt.

Locked regime (D-044): long permuted-MNIST, hidden=16 MLP, lr=0.2, 60 steps/task,
100 tasks, train_subset=6000. Confirmed to induce ~12-point plasticity decay with
weight-norm growth (4->20) and dormancy (6%->25%).

Arms (all 5 seeds):
  - sgd                          : control
  - l2_init (lambda=1e-2, FIXED) : magnitude method (Kumar 2023) — penalty bug fixed (D-045)
  - shrink_pert (0.7, 0.01)      : magnitude method (Ash & Adams 2020)
  - cbp (reset_rate=1e-2)        : Continual Backprop (Dohare 2024)
  - dir_control (c=1.0)          : ABLATION — matched-mag orthogonal steering (impossibility demo)
  - dir_inward (gamma=0.6)       : ABLATION — matched-mag inward steering (controls norm, kills acc)
  - adaptive_norm_revive (tgt=3) : OUR method attempt (D-046) — per-unit targeted control + revival

Decision rule (pre-committed): if adaptive_norm_revive Pareto-beats the best
magnitude baseline over 5 seeds (higher mean acc AND lower decay), we have a
method (contribution B). Otherwise we STOP and publish contribution A:
mechanism + impossibility + validation. Either branch is honest and complete.
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.datasets import load_mnist
from src.device import pick_device, env_summary
from src.plasticity_stress import PermutedMNIST, run_arm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_TABLES = os.path.join(ROOT, "results", "tables")
RESULTS_FIGS = os.path.join(ROOT, "results", "figures")

# Locked regime
REGIME = dict(hidden=16, depth=2, lr=0.2, epochs=1, batch=128,
              steps_per_task=60, n_tasks=100, train_subset=6000)
SEEDS = [20260824, 20260825, 20260826, 20260827, 20260828]

ARMS = [
    ("sgd",                  dict(method="sgd")),
    ("l2_init",              dict(method="l2_init", l2_init_lambda=1e-2)),
    ("shrink_pert",          dict(method="shrink_pert", shrink=0.7, perturb=0.01)),
    ("cbp",                  dict(method="cbp", cbp_reset_rate=1e-2)),
    ("dir_control_c1.0",     dict(method="dir_control", dir_coeff=1.0)),
    ("dir_inward_g0.6",      dict(method="dir_inward", dir_coeff=0.6)),
    ("adaptive_norm_revive", dict(method="adaptive_norm_revive", anr_target_norm=3.0)),
]

# Which arms are "magnitude baselines" the method must beat
MAGNITUDE_BASELINES = ["l2_init", "shrink_pert"]

# Publication-ready display names (used only for figure labels; keys are internal).
DISPLAY = {
    "sgd": "SGD", "l2_init": "L2-init", "shrink_pert": "Shrink&Perturb",
    "cbp": "CBP", "dir_control_c1.0": "Orthogonal",
    "dir_inward_g0.6": r"Inward ($\gamma=0.6$)", "adaptive_norm_revive": "ANR",
}


def _stringify(o):
    if isinstance(o, dict): return {k: _stringify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_stringify(v) for v in o]
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    return o


def peak_end_decay(task_acc):
    ta = np.array(task_acc)
    peak = float(ta.max())
    end = float(ta[-max(1, len(ta) // 10):].mean())
    return peak, end, peak - end


def main():
    device, profile = pick_device("auto")
    print(f"[E20] device={device} profile={profile}")
    tx, ty, vx, vy = load_mnist("Deep Learning Dataset1")

    raw = {name: [] for name, _ in ARMS}
    curves = {name: [] for name, _ in ARMS}  # per-seed task_test_acc curves

    for seed in SEEDS:
        stream = PermutedMNIST(tx, ty, vx, vy, n_tasks=REGIME["n_tasks"],
                                seed=seed, train_subset=REGIME["train_subset"])
        for name, kw in ARMS:
            t0 = time.time()
            r = run_arm(stream, lr=REGIME["lr"], epochs=REGIME["epochs"],
                        batch=REGIME["batch"], seed=seed, device=device,
                        hidden=REGIME["hidden"], depth=REGIME["depth"],
                        steps_per_task=REGIME["steps_per_task"],
                        verbose=False, **kw)
            peak, end, decay = peak_end_decay(r["task_test_acc"])
            raw[name].append({
                "mean_acc": r["mean_acc"], "peak_acc": peak, "end_acc": end,
                "decay": decay, "final_wl2": r["diag_weight_l2"][-1],
                "final_dormant": r["diag_dormant"][-1],
                "final_eff_rank": r["diag_eff_rank"][-1],
                "matched_max_dev": r["matched_max_dev"],
            })
            curves[name].append(r["task_test_acc"])
            print(f"  seed={seed} {name:22} mean={r['mean_acc']:.3f} "
                  f"decay={decay:+.3f} wl2={r['diag_weight_l2'][-1]:.1f} "
                  f"dorm={r['diag_dormant'][-1]:.3f}  ({time.time()-t0:.1f}s)")

    # Aggregate mean ± std
    agg = {}
    for name, _ in ARMS:
        rows = raw[name]
        agg[name] = {k + "_mean": float(np.mean([r[k] for r in rows]))
                     for k in rows[0]}
        agg[name].update({k + "_std": float(np.std([r[k] for r in rows]))
                          for k in rows[0]})

    # Best magnitude baseline (by mean_acc)
    best_mag = max(MAGNITUDE_BASELINES, key=lambda m: agg[m]["mean_acc_mean"])
    anr = agg["adaptive_norm_revive"]
    bm = agg[best_mag]
    # Pareto: ANR beats best magnitude baseline iff higher mean acc AND lower decay
    pareto_win = (anr["mean_acc_mean"] > bm["mean_acc_mean"] and
                  anr["decay_mean"] < bm["decay_mean"])
    # per-seed reproducibility of the acc comparison
    anr_wins_per_seed = sum(1 for i in range(len(SEEDS))
                             if raw["adaptive_norm_revive"][i]["mean_acc"]
                                > raw[best_mag][i]["mean_acc"])

    method_verdict = ("METHOD_POSITIVE_B" if (pareto_win and anr_wins_per_seed >= 4)
                       else "METHOD_NULL_PUBLISH_A")

    # Plot 1: learning-accuracy curves (mean across seeds)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    palette = {"sgd": "#8a8a8a", "l2_init": "#3E5C76", "shrink_pert": "#B07D48",
               "cbp": "#6B7B5A", "dir_control_c1.0": "#9C4A3C",
               "dir_inward_g0.6": "#c98a3c", "adaptive_norm_revive": "#2f2f2f"}
    for name, _ in ARMS:
        arr = np.array(curves[name])  # (seeds, tasks)
        m = arr.mean(0)
        ax.plot(range(len(m)), m, label=DISPLAY.get(name, name),
                color=palette.get(name), linewidth=1.6)
    ax.set_xlabel("task index (permuted-MNIST sequence)")
    ax.set_ylabel("task test accuracy (learning ability)")
    ax.grid(alpha=0.25, linewidth=0.5); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_FIGS, "e20_plasticity_curves.png"), dpi=300)
    plt.close(fig)

    # Plot 2: mechanism — decay vs final weight norm (the impossibility story)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for name, _ in ARMS:
        ax.scatter(agg[name]["final_wl2_mean"], agg[name]["decay_mean"],
                   s=90, color=palette.get(name), edgecolor="black", linewidth=0.5)
        ax.annotate(DISPLAY.get(name, name),
                    (agg[name]["final_wl2_mean"], agg[name]["decay_mean"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("final weight ‖θ‖ (norm growth)")
    ax.set_ylabel("plasticity decay (peak − end acc)")
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_FIGS, "e20_mechanism.png"), dpi=300)
    plt.close(fig)

    out = {
        "hardware": env_summary(), "regime": REGIME, "seeds": SEEDS,
        "per_seed_raw": raw, "aggregate": agg,
        "best_magnitude_baseline": best_mag,
        "method_pareto_win": bool(pareto_win),
        "method_wins_per_seed": anr_wins_per_seed,
        "method_verdict": method_verdict,
    }
    with open(os.path.join(RESULTS_TABLES, "e20_summary.json"), "w") as f:
        json.dump(_stringify(out), f, indent=2)

    print("\n" + "=" * 70)
    print(f"[E20] method verdict: {method_verdict}")
    print(f"[E20] best magnitude baseline: {best_mag} "
          f"(mean_acc={bm['mean_acc_mean']:.3f}±{bm['mean_acc_std']:.3f}, "
          f"decay={bm['decay_mean']:+.3f})")
    print(f"[E20] adaptive_norm_revive: mean_acc={anr['mean_acc_mean']:.3f}"
          f"±{anr['mean_acc_std']:.3f}, decay={anr['decay_mean']:+.3f} "
          f"(wins {anr_wins_per_seed}/5 seeds, pareto_win={pareto_win})")
    print("\n[E20] full leaderboard (mean_acc desc):")
    for name in sorted(agg, key=lambda n: -agg[n]["mean_acc_mean"]):
        a = agg[name]
        print(f"  {name:22} acc={a['mean_acc_mean']:.3f}±{a['mean_acc_std']:.3f}  "
              f"decay={a['decay_mean']:+.3f}±{a['decay_std']:.3f}  "
              f"wl2={a['final_wl2_mean']:.1f}  dorm={a['final_dormant_mean']:.3f}  "
              f"maxdev={a['matched_max_dev_mean']:.4f}")


if __name__ == "__main__":
    main()
