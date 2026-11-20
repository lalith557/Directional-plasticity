"""e25 - Second architecture: small CNN on Rotated MNIST (all real data).

Tests whether the two central findings generalize architecturally to a
convolutional network: (A) the matched-magnitude angle finding (off-gradient
matched steps hurt and inflate the norm), and (B) the arm-level norm-decay
relationship. Rotated MNIST is used because it preserves spatial structure
(permuted MNIST would make convolution meaningless).
"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.datasets import load_mnist
from src.device import pick_device, env_summary
from src.plasticity_stress import RotatedMNIST, run_arm_cnn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "results", "tables"); FIG = os.path.join(ROOT, "results", "figures")
RUN = dict(lr=0.1, epochs=1, batch=128, steps_per_task=60, base=16, fc=64)
SEEDS = [20260824, 20260825, 20260826]
ARMS = [("sgd", {}), ("l2_init", {"l2_init_lambda": 1e-2}),
        ("shrink_pert", {"shrink": 0.7, "perturb": 0.01}),
        ("norm_project", {"norm_project_target": 6.0}),
        ("dir_control", {"dir_coeff": 1.0}), ("dir_inward", {"dir_coeff": 0.6}),
        ("random_dir", {})]
COS = [1.0, 0.75, 0.5, 0.25, 0.1]


def pe(ta): ta = np.array(ta); return float(ta.max() - ta[-max(1, len(ta)//10):].mean())
def agg(rows):
    ks = rows[0].keys()
    a = {k+"_mean": float(np.mean([r[k] for r in rows])) for k in ks}
    a.update({k+"_std": float(np.std([r[k] for r in rows])) for k in ks}); return a
def pear(x, y):
    x = np.array(x)-np.mean(x); y = np.array(y)-np.mean(y)
    return float((x*y).sum()/(np.sqrt((x*x).sum()*(y*y).sum())+1e-12))


def main():
    device, profile = pick_device("auto"); print(f"[E25] device={device} {profile}")
    tx, ty, vx, vy = load_mnist("Deep Learning Dataset1")
    sfn = lambda s: RotatedMNIST(tx, ty, vx, vy, n_tasks=100, seed=s,
                                 train_subset=6000, test_subset=2000)
    # ---- 7-arm comparison ----
    raw = {n: [] for n, _ in ARMS}
    for seed in SEEDS:
        st = sfn(seed)
        for name, kw in ARMS:
            t0 = time.time()
            r = run_arm_cnn(st, name, seed=seed, device=device, **RUN, **kw)
            raw[name].append({"mean_acc": r["mean_acc"], "decay": pe(r["task_test_acc"]),
                              "final_wl2": r["diag_weight_l2"][-1], "final_dormant": r["diag_dormant"][-1],
                              "final_eff_rank": r["diag_eff_rank"][-1], "grad_align": r["grad_align_mean"],
                              "matched_max_dev": r["matched_max_dev"]})
            print(f"  seed={seed} {name:14} acc={r['mean_acc']:.3f} decay={raw[name][-1]['decay']:+.3f} "
                  f"wl2={r['diag_weight_l2'][-1]:.1f} ({time.time()-t0:.0f}s)")
    A = {n: agg(raw[n]) for n, _ in ARMS}
    # norm-decay correlation across the 7 arms (use finite arms only)
    keys = [n for n, _ in ARMS if np.isfinite(A[n]["final_wl2_mean"])]
    rnd = pear([A[n]["final_wl2_mean"] for n in keys], [A[n]["decay_mean"] for n in keys])
    # ---- CNN angle sweep ----
    ang = {f"{c:g}": [] for c in COS}
    for seed in SEEDS:
        st = sfn(seed)
        for c in COS:
            r = run_arm_cnn(st, "angle", cos_target=c, seed=seed, device=device, **RUN)
            ang[f"{c:g}"].append({"mean_acc": r["mean_acc"], "final_wl2": r["diag_weight_l2"][-1],
                                  "grad_align": r["grad_align_mean"]})
            print(f"  [angle] seed={seed} cos={c:g} acc={r['mean_acc']:.3f} wl2={r['diag_weight_l2'][-1]:.1f}")
    angA = {k: agg(v) for k, v in ang.items()}
    out = {"hardware": env_summary(), "arch": "SmallCNN(16,32,fc64)", "benchmark": "rotated_mnist",
           "seeds": SEEDS, "arm_aggregate": A, "pearson_norm_decay_narms": rnd,
           "n_arms": len(keys), "angle_cos": COS, "angle_aggregate": angA}
    json.dump(out, open(os.path.join(TAB, "e25_cnn_summary.json"), "w"), indent=2)
    # figure: CNN angle sweep (acc + norm vs achieved cos)
    cs = [float(k) for k in ang]
    acc = [angA[f"{c:g}"]["mean_acc_mean"] for c in COS]
    wl = [angA[f"{c:g}"]["final_wl2_mean"] for c in COS]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(cs, acc, "o-", color="#3E5C76"); ax.set_xlabel("cos(applied, SGD)")
    ax.set_ylabel("mean accuracy", color="#3E5C76")
    ax2 = ax.twinx(); ax2.plot(cs, wl, "s--", color="#B07D48"); ax2.set_ylabel("final ||theta||", color="#B07D48")
    ax.set_title("E25 CNN angle sweep (Rotated MNIST, 3 seeds)"); ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "e25_cnn_angle.png"), dpi=140); plt.close(fig)
    print(f"\n[E25] CNN Rotated-MNIST leaderboard (Pearson(norm,decay) over {len(keys)} arms = {rnd:.3f}):")
    for n in sorted(A, key=lambda k: -A[k]["mean_acc_mean"]):
        a = A[n]; print(f"  {n:14} acc={a['mean_acc_mean']:.3f}+-{a['mean_acc_std']:.3f} "
                        f"decay={a['decay_mean']:+.3f} wl2={a['final_wl2_mean']:.1f} gA={a['grad_align_mean']:+.2f}")
    print("[E25] CNN angle sweep:")
    for c in COS:
        a = angA[f"{c:g}"]; print(f"  cos={c:g}: acc={a['mean_acc_mean']:.3f} wl2={a['final_wl2_mean']:.1f} gA={a['grad_align_mean']:+.2f}")


if __name__ == "__main__":
    main()
