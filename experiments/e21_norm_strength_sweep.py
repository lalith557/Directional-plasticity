"""e21 - Norm-control-strength sweep (mechanism-strengthening experiment).

Reviewer point #21: vary the strength of a norm-control regularizer and show the
weight norm and the plasticity decay move together. This tests the norm-growth
association by INTERVENING on the norm knob (L2-init lambda) rather than only
observing across heterogeneous methods.

Same locked regime as e20 (D-044). Real data only; no fabricated numbers.
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.datasets import load_mnist
from src.device import pick_device, env_summary
from src.plasticity_stress import PermutedMNIST, run_arm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "results", "tables")
FIGS = os.path.join(ROOT, "results", "figures")

REGIME = dict(hidden=16, depth=2, lr=0.2, epochs=1, batch=128,
              steps_per_task=60, n_tasks=100, train_subset=6000)
SEEDS = [20260824, 20260825, 20260826]
LAMBDAS = [0.0, 1e-3, 3e-3, 1e-2, 3e-2]


def peak_end_decay(ta):
    ta = np.array(ta); return float(ta.max() - ta[-max(1, len(ta)//10):].mean())


def main():
    device, profile = pick_device("auto")
    print(f"[E21] device={device} profile={profile}")
    tx, ty, vx, vy = load_mnist("Deep Learning Dataset1")
    rows = {f"{lam:g}": [] for lam in LAMBDAS}
    for seed in SEEDS:
        stream = PermutedMNIST(tx, ty, vx, vy, n_tasks=REGIME["n_tasks"],
                                seed=seed, train_subset=REGIME["train_subset"])
        for lam in LAMBDAS:
            r = run_arm(stream, method="l2_init", l2_init_lambda=lam,
                        lr=REGIME["lr"], epochs=REGIME["epochs"], batch=REGIME["batch"],
                        seed=seed, device=device, hidden=REGIME["hidden"],
                        depth=REGIME["depth"], steps_per_task=REGIME["steps_per_task"],
                        verbose=False)
            rows[f"{lam:g}"].append({
                "mean_acc": r["mean_acc"], "decay": peak_end_decay(r["task_test_acc"]),
                "final_wl2": r["diag_weight_l2"][-1], "final_dormant": r["diag_dormant"][-1],
            })
            print(f"  seed={seed} lambda={lam:g}: acc={r['mean_acc']:.3f} "
                  f"decay={rows[f'{lam:g}'][-1]['decay']:+.3f} wl2={r['diag_weight_l2'][-1]:.1f}")
    agg = {}
    for k, rr in rows.items():
        agg[k] = {m + "_mean": float(np.mean([x[m] for x in rr])) for m in rr[0]}
        agg[k].update({m + "_std": float(np.std([x[m] for x in rr])) for m in rr[0]})

    lams = [f"{l:g}" for l in LAMBDAS]
    wl2 = [agg[k]["final_wl2_mean"] for k in lams]
    dec = [agg[k]["decay_mean"] for k in lams]
    fig, ax1 = plt.subplots(figsize=(7.4, 5.0))
    c1, c2 = "#3E5C76", "#9C4A3C"
    ax1.plot(wl2, dec, "o-", color="#23201B", linewidth=1.6, markersize=7)
    for k, x, y in zip(lams, wl2, dec):
        ax1.annotate(f"$\\lambda$={k}", (x, y), fontsize=9, xytext=(6, 5),
                     textcoords="offset points", color="#6B6660")
    ax1.set_xlabel("final weight norm  ||theta||  (controlled by L2-init strength)")
    ax1.set_ylabel("plasticity decay (peak - end)")
    ax1.set_title("E21: stronger norm control -> lower norm -> lower plasticity decay")
    ax1.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "e21_norm_strength.png"), dpi=140)
    plt.close(fig)

    out = {"hardware": env_summary(), "regime": REGIME, "seeds": SEEDS,
           "lambdas": LAMBDAS, "aggregate": agg}
    with open(os.path.join(TABLES, "e21_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n[E21] lambda -> (norm, decay, acc):")
    for k in lams:
        a = agg[k]
        print(f"  lambda={k:>5}: wl2={a['final_wl2_mean']:5.1f}  "
              f"decay={a['decay_mean']:+.3f}+-{a['decay_std']:.3f}  "
              f"acc={a['mean_acc_mean']:.3f}+-{a['mean_acc_std']:.3f}")


if __name__ == "__main__":
    main()
