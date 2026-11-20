"""e24 - Generalization experiments (ICLR round; all real data).

  A. Rotated-MNIST second benchmark (7 arms x 5 seeds, width-16). Complements
     permuted MNIST: input geometry changes smoothly rather than destroying
     spatial structure.
  B. Wider architecture 784-128-128-10 on permuted MNIST (7 arms x 5 seeds),
     complementing the width-16 and width-64 (e22) results.

Outputs:
  results/tables/e24_rotmnist_summary.json  + figs/e24_rotmnist_curves.png
  results/tables/e24_arch128_summary.json   + figs/e24_arch128_curves.png
"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.datasets import load_mnist
from src.device import pick_device, env_summary
from src.plasticity_stress import PermutedMNIST, RotatedMNIST, run_arm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "results", "tables"); FIGS = os.path.join(ROOT, "results", "figures")
RUN = dict(lr=0.2, epochs=1, batch=128, steps_per_task=60, depth=2)
SEEDS = [20260824, 20260825, 20260826, 20260827, 20260828]
ARMS = [
    ("sgd", dict(method="sgd")),
    ("l2_init", dict(method="l2_init", l2_init_lambda=1e-2)),
    ("shrink_pert", dict(method="shrink_pert", shrink=0.7, perturb=0.01)),
    ("cbp", dict(method="cbp", cbp_reset_rate=1e-2)),
    ("dir_control_c1.0", dict(method="dir_control", dir_coeff=1.0)),
    ("dir_inward_g0.6", dict(method="dir_inward", dir_coeff=0.6)),
    ("adaptive_norm_revive", dict(method="adaptive_norm_revive", anr_target_norm=3.0)),
]
PAL = {"sgd": "#8a8a8a", "l2_init": "#3E5C76", "shrink_pert": "#B07D48", "cbp": "#6B7B5A",
       "dir_control_c1.0": "#9C4A3C", "dir_inward_g0.6": "#c98a3c", "adaptive_norm_revive": "#2f2f2f"}
# Publication-ready display names (figure labels only; internal keys unchanged).
DISPLAY = {"sgd": "SGD", "l2_init": "L2-init", "shrink_pert": "Shrink&Perturb", "cbp": "CBP",
           "dir_control_c1.0": "Directional-orthogonal", "dir_inward_g0.6": "Directional-inward",
           "adaptive_norm_revive": "ANR"}
TITLES = {"rotmnist": "Rotated MNIST, MLP width 16 (mean of 5 seeds)",
          "arch128": "Permuted MNIST, MLP 784-128-128-10 (mean of 5 seeds)"}


def pe(ta): ta = np.array(ta); return float(ta.max() - ta[-max(1, len(ta)//10):].mean())
def agg(rows):
    ks = rows[0].keys()
    a = {k+"_mean": float(np.mean([r[k] for r in rows])) for k in ks}
    a.update({k+"_std": float(np.std([r[k] for r in rows])) for k in ks}); return a
def pear(x, y):
    x = np.array(x)-np.mean(x); y = np.array(y)-np.mean(y)
    return float((x*y).sum()/(np.sqrt((x*x).sum()*(y*y).sum())+1e-12))


def run_block(stream_fn, hidden, tag, device):
    raw = {n: [] for n, _ in ARMS}; curves = {n: [] for n, _ in ARMS}
    for seed in SEEDS:
        stream = stream_fn(seed)
        for name, kw in ARMS:
            t0 = time.time()
            r = run_arm(stream, hidden=hidden, seed=seed, device=device, verbose=False, **RUN, **kw)
            raw[name].append({"mean_acc": r["mean_acc"], "decay": pe(r["task_test_acc"]),
                              "final_wl2": r["diag_weight_l2"][-1], "final_dormant": r["diag_dormant"][-1],
                              "final_eff_rank": r["diag_eff_rank"][-1], "matched_max_dev": r["matched_max_dev"]})
            curves[name].append(r["task_test_acc"])
            print(f"  [{tag}] seed={seed} {name:22} acc={r['mean_acc']:.3f} "
                  f"decay={raw[name][-1]['decay']:+.3f} wl2={r['diag_weight_l2'][-1]:.1f} ({time.time()-t0:.0f}s)")
    A = {n: agg(raw[n]) for n, _ in ARMS}
    rr = pear([A[n]["final_wl2_mean"] for n in A], [A[n]["decay_mean"] for n in A])
    fig, ax = plt.subplots(figsize=(9, 5))
    for n, _ in ARMS:
        m = np.array(curves[n]).mean(0); ax.plot(range(len(m)), m, label=DISPLAY.get(n, n), color=PAL.get(n), linewidth=1.5)
    ax.set_xlabel("task index"); ax.set_ylabel("task test accuracy")
    ax.set_title(TITLES.get(tag, "")); ax.grid(alpha=0.25, lw=0.5); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, f"e24_{tag}_curves.png"), dpi=300); plt.close(fig)
    return A, rr


def main():
    device, profile = pick_device("auto")
    print(f"[E24] device={device} profile={profile}")
    tx, ty, vx, vy = load_mnist("Deep Learning Dataset1")

    print("\n=== A. Rotated MNIST (width 16) ===")
    A_rot, r_rot = run_block(
        lambda s: RotatedMNIST(tx, ty, vx, vy, n_tasks=100, seed=s, train_subset=6000, test_subset=2000),
        hidden=16, tag="rotmnist", device=device)
    json.dump({"hardware": env_summary(), "benchmark": "rotated_mnist", "seeds": SEEDS,
               "aggregate": A_rot, "pearson_norm_decay_narms": r_rot},
              open(os.path.join(TABLES, "e24_rotmnist_summary.json"), "w"), indent=2)

    print("\n=== B. Wider architecture 784-128-128-10 (permuted MNIST) ===")
    A_128, r_128 = run_block(
        lambda s: PermutedMNIST(tx, ty, vx, vy, n_tasks=100, seed=s, train_subset=6000),
        hidden=128, tag="arch128", device=device)
    json.dump({"hardware": env_summary(), "arch": "784-128-128-10", "seeds": SEEDS,
               "aggregate": A_128, "pearson_norm_decay_narms": r_128},
              open(os.path.join(TABLES, "e24_arch128_summary.json"), "w"), indent=2)

    for tag, A, rr in [("rotmnist", A_rot, r_rot), ("arch128", A_128, r_128)]:
        print(f"\n[E24-{tag}] leaderboard (acc desc), Pearson(norm,decay)={rr:.2f}:")
        for n in sorted(A, key=lambda k: -A[k]["mean_acc_mean"]):
            a = A[n]; print(f"  {n:22} acc={a['mean_acc_mean']:.3f}+-{a['mean_acc_std']:.3f} "
                            f"decay={a['decay_mean']:+.3f} wl2={a['final_wl2_mean']:.1f}")


if __name__ == "__main__":
    main()
