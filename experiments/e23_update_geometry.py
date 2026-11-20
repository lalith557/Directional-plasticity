"""e23 - Update-geometry experiments (ICLR round; all real data).

Addresses the "only two directions were tested" and "inward-steering confound"
criticisms by (1) adding a matched-magnitude RANDOM-direction control, (2) a
controlled cos-similarity ANGLE sweep at matched magnitude, (3) logging the
gradient- and radial-alignment of every applied update, and (4) a direct
norm-projection intervention. Width-16 MLP, locked stress protocol, 5 seeds.

Outputs:
  results/tables/e23_summary.json
  figs/e23_angle_sweep.png       (cos(applied,SGD) vs decay & accuracy)
  figs/e23_alignment.png         (grad-alignment vs decay; radial-align vs norm)
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
from src.plasticity_stress import PermutedMNIST, run_arm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "results", "tables"); FIGS = os.path.join(ROOT, "results", "figures")
RUN = dict(lr=0.2, epochs=1, batch=128, steps_per_task=60, depth=2)
SEEDS = [20260824, 20260825, 20260826, 20260827, 20260828]
# Non-negative cos grid: matched-magnitude alignment sweep. cos<0 (partly
# ascending the loss) diverges trivially and is not informative, so we sweep the
# meaningful [0,1] range where the update retains a non-negative gradient component.
COS = [1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.0]

ARMS = [
    ("sgd", dict(method="sgd")),
    ("l2_init", dict(method="l2_init", l2_init_lambda=1e-2)),
    ("shrink_pert", dict(method="shrink_pert", shrink=0.7, perturb=0.01)),
    ("cbp", dict(method="cbp", cbp_reset_rate=1e-2)),
    ("dir_control_c1.0", dict(method="dir_control", dir_coeff=1.0)),
    ("dir_inward_g0.6", dict(method="dir_inward", dir_coeff=0.6)),
    ("adaptive_norm_revive", dict(method="adaptive_norm_revive", anr_target_norm=3.0)),
    ("random_dir", dict(method="random_dir")),
    ("norm_project_t6", dict(method="norm_project", norm_project_target=6.0)),
] + [(f"angle_c={c:+.2f}", dict(method="angle", cos_target=c)) for c in COS]


def pe(ta): ta = np.array(ta); return float(ta.max() - ta[-max(1, len(ta)//10):].mean())
def agg(rows):
    ks = rows[0].keys()
    a = {k+"_mean": float(np.mean([r[k] for r in rows])) for k in ks}
    a.update({k+"_std": float(np.std([r[k] for r in rows])) for k in ks}); return a
def pearson(x, y):
    x = np.array(x)-np.mean(x); y = np.array(y)-np.mean(y)
    return float((x*y).sum()/(np.sqrt((x*x).sum()*(y*y).sum())+1e-12))


def main():
    device, profile = pick_device("auto")
    print(f"[E23] device={device} profile={profile}")
    tx, ty, vx, vy = load_mnist("Deep Learning Dataset1")
    raw = {n: [] for n, _ in ARMS}
    for seed in SEEDS:
        stream = PermutedMNIST(tx, ty, vx, vy, n_tasks=100, seed=seed, train_subset=6000)
        for name, kw in ARMS:
            t0 = time.time()
            r = run_arm(stream, hidden=16, seed=seed, device=device, verbose=False, **RUN, **kw)
            raw[name].append({
                "mean_acc": r["mean_acc"], "decay": pe(r["task_test_acc"]),
                "final_wl2": r["diag_weight_l2"][-1], "final_dormant": r["diag_dormant"][-1],
                "final_eff_rank": r["diag_eff_rank"][-1],
                "grad_align": r["grad_align_mean"], "radial_align": r["radial_align_mean"],
                "matched_max_dev": r["matched_max_dev"]})
            print(f"  seed={seed} {name:20} acc={r['mean_acc']:.3f} decay={raw[name][-1]['decay']:+.3f} "
                  f"gA={r['grad_align_mean']:+.2f} rA={r['radial_align_mean']:+.2f} "
                  f"wl2={r['diag_weight_l2'][-1]:.1f} ({time.time()-t0:.0f}s)")
    A = {n: agg(raw[n]) for n, _ in ARMS}

    # Fig: angle sweep
    cs = COS
    accs = [A[f"angle_c={c:+.2f}"]["mean_acc_mean"] for c in cs]
    accs_sd = [A[f"angle_c={c:+.2f}"]["mean_acc_std"] for c in cs]
    decs = [A[f"angle_c={c:+.2f}"]["decay_mean"] for c in cs]
    fig, ax = plt.subplots(figsize=(7.4, 5))
    ax.errorbar(cs, accs, yerr=accs_sd, fmt="o-", color="#3E5C76", capsize=3, label="mean accuracy")
    ax.set_xlabel("cos(applied update, SGD update)  at matched step size")
    ax.set_ylabel("mean accuracy", color="#3E5C76")
    ax2 = ax.twinx(); ax2.plot(cs, decs, "s--", color="#B07D48", label="plasticity decay")
    ax2.set_ylabel("plasticity decay", color="#B07D48")
    ax.set_title("E23: matched-magnitude angle sweep (5 seeds)")
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "e23_angle_sweep.png"), dpi=140); plt.close(fig)

    # Fig: alignment scatter across all arms
    names = [n for n, _ in ARMS]
    gA = [A[n]["grad_align_mean"] for n in names]; dec = [A[n]["decay_mean"] for n in names]
    rA = [A[n]["radial_align_mean"] for n in names]; wl = [A[n]["final_wl2_mean"] for n in names]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    a1.scatter(gA, dec, s=60, color="#3E5C76", edgecolor="black", linewidth=0.4)
    a1.set_xlabel("mean gradient alignment cos(applied, -grad)"); a1.set_ylabel("plasticity decay")
    a1.set_title(f"decay vs grad-alignment (r={pearson(gA,dec):.2f})"); a1.grid(alpha=0.25, lw=0.5)
    a2.scatter(rA, wl, s=60, color="#B07D48", edgecolor="black", linewidth=0.4)
    a2.set_xlabel("mean radial alignment cos(applied, theta)"); a2.set_ylabel("final weight norm")
    a2.set_title(f"final-norm vs radial-alignment (r={pearson(rA,wl):.2f})"); a2.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "e23_alignment.png"), dpi=140); plt.close(fig)

    out = {"hardware": env_summary(), "seeds": SEEDS, "cos_grid": COS, "aggregate": A,
           "pearson_gradalign_decay": pearson(gA, dec),
           "pearson_radialalign_norm": pearson(rA, wl)}
    json.dump(out, open(os.path.join(TABLES, "e23_summary.json"), "w"), indent=2)
    print("\n[E23] angle sweep (cos -> acc, decay):")
    for c in cs:
        a = A[f"angle_c={c:+.2f}"]
        print(f"  cos={c:+.2f}: acc={a['mean_acc_mean']:.3f}+-{a['mean_acc_std']:.3f} "
              f"decay={a['decay_mean']:+.3f} wl2={a['final_wl2_mean']:.1f} rA={a['radial_align_mean']:+.2f}")
    print(f"\n[E23] random_dir: acc={A['random_dir']['mean_acc_mean']:.3f} decay={A['random_dir']['decay_mean']:+.3f} "
          f"wl2={A['random_dir']['final_wl2_mean']:.1f} gA={A['random_dir']['grad_align_mean']:+.2f}")
    print(f"[E23] norm_project_t6: acc={A['norm_project_t6']['mean_acc_mean']:.3f} "
          f"decay={A['norm_project_t6']['decay_mean']:+.3f} wl2={A['norm_project_t6']['final_wl2_mean']:.1f}")
    print(f"[E23] dir_inward grad_align={A['dir_inward_g0.6']['grad_align_mean']:+.2f} "
          f"(vs sgd {A['sgd']['grad_align_mean']:+.2f})")
    print(f"[E23] Pearson(grad_align,decay)={out['pearson_gradalign_decay']:.2f} "
          f"Pearson(radial_align,norm)={out['pearson_radialalign_norm']:.2f}")


if __name__ == "__main__":
    main()
