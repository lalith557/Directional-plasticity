"""E27 — Radial-coefficient sweep (inward directional allocation) at matched step.

Sweeps the inward radial coefficient gamma of the EXISTING dir_inward rule
    Delta' = normalize(Delta - gamma*||Delta||*theta_hat) * ||Delta||   (matched magnitude)
across gamma in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}. gamma=0 -> plain SGD; larger gamma
-> stronger inward (norm-reducing) radial allocation. This converts the two
directional endpoints (orthogonal / inward) into a controlled radial sweep, showing
the geometry->norm->accuracy trade-off (Prop. 1 / Cor. 2).

IDENTICAL protocol to e20 (width-16 permuted MNIST, 5 seeds). No fabrication:
every number is produced by run_arm on the released code/data.
Outputs: results/tables/e27_radial_sweep_summary.json
         results/figures/e27_radial_sweep.png  (+ paper/figs/radial_sweep.png)
"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.device import pick_device, env_summary
from src.datasets import load_mnist
from src.plasticity_stress import PermutedMNIST, run_arm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "results", "tables")
FIGS = os.path.join(ROOT, "results", "figures")
PAPER_FIGS = os.path.join(ROOT, "paper", "figs")

REGIME = dict(hidden=16, depth=2, lr=0.2, epochs=1, batch=128,
              steps_per_task=60, n_tasks=100, train_subset=6000)
SEEDS = [20260824, 20260825, 20260826, 20260827, 20260828]
GAMMAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def peak_end_decay(task_acc):
    ta = np.array(task_acc)
    peak = float(ta.max())
    end = float(ta[-max(1, len(ta) // 10):].mean())
    return peak, end, peak - end


def main():
    device, profile = pick_device("auto")
    print(f"[E27] device={device} profile={profile}")
    tx, ty, vx, vy = load_mnist("Deep Learning Dataset1")

    raw = {f"{g:g}": [] for g in GAMMAS}
    for seed in SEEDS:
        stream = PermutedMNIST(tx, ty, vx, vy, n_tasks=REGIME["n_tasks"],
                               seed=seed, train_subset=REGIME["train_subset"])
        for g in GAMMAS:
            t0 = time.time()
            r = run_arm(stream, method="dir_inward", lr=REGIME["lr"],
                        epochs=REGIME["epochs"], batch=REGIME["batch"], seed=seed,
                        device=device, hidden=REGIME["hidden"], depth=REGIME["depth"],
                        steps_per_task=REGIME["steps_per_task"], dir_coeff=g, verbose=False)
            peak, end, decay = peak_end_decay(r["task_test_acc"])
            raw[f"{g:g}"].append({
                "mean_acc": float(r["mean_acc"]), "peak_acc": peak, "end_acc": end,
                "decay": decay, "final_wl2": float(r["diag_weight_l2"][-1]),
                "final_dormant": float(r["diag_dormant"][-1]),
                "final_eff_rank": float(r["diag_eff_rank"][-1]),
                "radial_align_mean": float(r.get("radial_align_mean", float("nan"))),
                "matched_max_dev": float(r["matched_max_dev"]),
            })
            print(f"  seed={seed} gamma={g:.1f} acc={r['mean_acc']:.3f} "
                  f"decay={decay:+.3f} wl2={r['diag_weight_l2'][-1]:.1f} "
                  f"dorm={r['diag_dormant'][-1]:.3f} ({time.time()-t0:.1f}s)")

    agg = {}
    for k, rows in raw.items():
        agg[k] = {m + "_mean": float(np.mean([r[m] for r in rows])) for m in rows[0]}
        agg[k].update({m + "_std": float(np.std([r[m] for r in rows])) for m in rows[0]})

    out = {"hardware": env_summary(), "regime": REGIME, "seeds": SEEDS,
           "gammas": GAMMAS, "method": "dir_inward",
           "note": "radial-coefficient sweep; gamma=0 is plain SGD; matched step",
           "per_seed_raw": raw, "aggregate": agg}
    os.makedirs(TABLES, exist_ok=True)
    json.dump(out, open(os.path.join(TABLES, "e27_radial_sweep_summary.json"), "w"), indent=2)
    print("wrote e27_radial_sweep_summary.json")

    # ---- figure: gamma vs accuracy (left) and final norm (right, log) ----
    gs = GAMMAS
    acc = [agg[f"{g:g}"]["mean_acc_mean"] for g in gs]
    acc_s = [agg[f"{g:g}"]["mean_acc_std"] for g in gs]
    nrm = [agg[f"{g:g}"]["final_wl2_mean"] for g in gs]
    nrm_s = [agg[f"{g:g}"]["final_wl2_std"] for g in gs]

    fig, axL = plt.subplots(figsize=(7.0, 4.4))
    c_acc, c_norm = "#3E5C76", "#9C4A3C"
    axL.errorbar(gs, acc, yerr=acc_s, fmt="o-", color=c_acc, capsize=3, lw=1.8, ms=6,
                 label="mean accuracy")
    axL.set_xlabel(r"inward radial coefficient $\gamma$ ($\gamma{=}0$: SGD)")
    axL.set_ylabel("mean accuracy", color=c_acc)
    axL.tick_params(axis="y", labelcolor=c_acc)
    axL.grid(alpha=0.25, lw=0.5)
    axR = axL.twinx()
    axR.errorbar(gs, nrm, yerr=nrm_s, fmt="s--", color=c_norm, capsize=3, lw=1.8, ms=6,
                 label=r"final $\|\theta\|$")
    axR.set_yscale("log")
    axR.set_ylabel(r"final weight norm $\|\theta\|$ (log)", color=c_norm)
    axR.tick_params(axis="y", labelcolor=c_norm)
    fig.tight_layout()
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, "e27_radial_sweep.png"), dpi=300)
    if os.path.isdir(PAPER_FIGS):
        fig.savefig(os.path.join(PAPER_FIGS, "radial_sweep.png"), dpi=300)
    plt.close(fig)
    print("wrote e27_radial_sweep.png")

    print("\n=== radial sweep summary (mean over 5 seeds) ===")
    for g in gs:
        a = agg[f"{g:g}"]
        print(f" gamma={g:.1f}  acc={a['mean_acc_mean']:.3f}  decay={a['decay_mean']:.3f}  "
              f"norm={a['final_wl2_mean']:.2f}  dormant={a['final_dormant_mean']:.3f}  "
              f"radial_align={a['radial_align_mean_mean']:+.3f}")


if __name__ == "__main__":
    main()
