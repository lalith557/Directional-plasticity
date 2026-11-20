"""e22 - Second-round reviewer experiments (all real data; no fabrication).

  A. Second architecture: 784-64-64-10 MLP, SAME stress protocol, 7 arms x 5 seeds.
     Tests whether the norm-growth / plasticity-decay picture persists at 4x width.
  B. 5-seed norm-control-strength sweep over a PREDETERMINED geometric grid of
     lambda values (upgrades the earlier 3-seed sweep to 5 seeds, 6 values).
  C. ANR component ablation: shrink-only vs revive-only vs shrink+revive, x 5 seeds,
     with SGD and Shrink&Perturb as references. Shows which component drives ANR.

Writes:
  results/tables/e22_arch64_summary.json  + figs/e22_arch64_curves.png
  results/tables/e22_lambda_summary.json  + figs/e22_lambda_sweep.png
  results/tables/e22_anr_ablation_summary.json + figs/e22_anr_ablation.png
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.datasets import load_mnist
from src.device import pick_device, env_summary
from src.plasticity_stress import PermutedMNIST, run_arm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "results", "tables")
FIGS = os.path.join(ROOT, "results", "figures")

BASE = dict(lr=0.2, epochs=1, batch=128, steps_per_task=60, n_tasks=100,
            train_subset=6000, depth=2)
# run_arm accepts only these (n_tasks/train_subset belong to the data stream):
RUN_BASE = dict(lr=0.2, epochs=1, batch=128, steps_per_task=60, depth=2)
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
# PREDETERMINED geometric lambda grid (fixed a priori, not tuned on outcomes)
LAMBDAS = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

# Publication-ready display names (figure labels only; internal keys unchanged).
DISPLAY = {
    "sgd": "SGD", "l2_init": "L2-init", "shrink_pert": "Shrink&Perturb",
    "cbp": "CBP", "dir_control_c1.0": "Directional-orthogonal",
    "dir_inward_g0.6": "Directional-inward", "adaptive_norm_revive": "ANR",
    "anr_shrink_only": "ANR: shrink", "anr_revive_only": "ANR: revive",
    "anr_full": "ANR: full",
}


def pe_decay(ta):
    ta = np.array(ta); return float(ta.max() - ta[-max(1, len(ta) // 10):].mean())


def agg_over_seeds(rows):
    keys = rows[0].keys()
    a = {k + "_mean": float(np.mean([r[k] for r in rows])) for k in keys}
    a.update({k + "_std": float(np.std([r[k] for r in rows])) for k in keys})
    return a


def run_seven_arms(stream_fn, hidden, device, tag):
    raw = {n: [] for n, _ in ARMS}
    curves = {n: [] for n, _ in ARMS}
    for seed in SEEDS:
        stream = stream_fn(seed)
        for name, kw in ARMS:
            t0 = time.time()
            r = run_arm(stream, hidden=hidden, seed=seed, device=device,
                        verbose=False, **RUN_BASE, **kw)
            raw[name].append({
                "mean_acc": r["mean_acc"], "decay": pe_decay(r["task_test_acc"]),
                "final_wl2": r["diag_weight_l2"][-1],
                "final_dormant": r["diag_dormant"][-1],
                "final_eff_rank": r["diag_eff_rank"][-1],
                "matched_max_dev": r["matched_max_dev"],
            })
            curves[name].append(r["task_test_acc"])
            print(f"  [{tag}] seed={seed} {name:22} acc={r['mean_acc']:.3f} "
                  f"decay={raw[name][-1]['decay']:+.3f} wl2={r['diag_weight_l2'][-1]:.1f} "
                  f"({time.time()-t0:.0f}s)")
    agg = {n: agg_over_seeds(raw[n]) for n, _ in ARMS}
    return raw, curves, agg


def pearson(x, y):
    x = np.array(x) - np.mean(x); y = np.array(y) - np.mean(y)
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-12))


def main():
    device, profile = pick_device("auto")
    print(f"[E22] device={device} profile={profile}")
    tx, ty, vx, vy = load_mnist("Deep Learning Dataset1")

    def stream_fn(seed):
        return PermutedMNIST(tx, ty, vx, vy, n_tasks=BASE["n_tasks"], seed=seed,
                              train_subset=BASE["train_subset"])

    # ---------- A. Second architecture (width 64) ----------
    print("\n=== A. Second architecture: 784-64-64-10 ===")
    rawA, curvesA, aggA = run_seven_arms(stream_fn, hidden=64, device=device, tag="arch64")
    normsA = [aggA[n]["final_wl2_mean"] for n in aggA]
    decaysA = [aggA[n]["decay_mean"] for n in aggA]
    rA = pearson(normsA, decaysA)
    outA = {"hardware": env_summary(), "arch": "784-64-64-10", "seeds": SEEDS,
            "aggregate": aggA, "pearson_norm_decay_narms": rA, "n_arms": len(aggA)}
    json.dump(outA, open(os.path.join(TABLES, "e22_arch64_summary.json"), "w"), indent=2)
    fig, ax = plt.subplots(figsize=(9, 5))
    pal = {"sgd": "#8a8a8a", "l2_init": "#3E5C76", "shrink_pert": "#B07D48",
           "cbp": "#6B7B5A", "dir_control_c1.0": "#9C4A3C",
           "dir_inward_g0.6": "#c98a3c", "adaptive_norm_revive": "#2f2f2f"}
    for n, _ in ARMS:
        m = np.array(curvesA[n]).mean(0)
        ax.plot(range(len(m)), m, label=DISPLAY.get(n, n), color=pal.get(n), linewidth=1.5)
    ax.set_xlabel("task index"); ax.set_ylabel("task test accuracy")
    ax.grid(alpha=0.25, linewidth=0.5); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "e22_arch64_curves.png"), dpi=300); plt.close(fig)
    print(f"[E22-A] Pearson(norm,decay) across arms = {rA:.3f}")

    # ---------- B. 5-seed lambda sweep ----------
    print("\n=== B. 5-seed norm-control-strength sweep ===")
    rowsB = {f"{l:g}": [] for l in LAMBDAS}
    for seed in SEEDS:
        stream = stream_fn(seed)
        for lam in LAMBDAS:
            r = run_arm(stream, hidden=16, seed=seed, device=device, verbose=False,
                        method="l2_init", l2_init_lambda=lam, **RUN_BASE)
            rowsB[f"{lam:g}"].append({
                "mean_acc": r["mean_acc"], "decay": pe_decay(r["task_test_acc"]),
                "final_wl2": r["diag_weight_l2"][-1], "final_dormant": r["diag_dormant"][-1]})
            print(f"  [lambda] seed={seed} lam={lam:g} acc={r['mean_acc']:.3f} "
                  f"decay={rowsB[f'{lam:g}'][-1]['decay']:+.3f} wl2={r['diag_weight_l2'][-1]:.1f}")
    aggB = {k: agg_over_seeds(v) for k, v in rowsB.items()}
    outB = {"seeds": SEEDS, "lambdas": LAMBDAS, "grid": "predetermined geometric",
            "aggregate": aggB}
    json.dump(outB, open(os.path.join(TABLES, "e22_lambda_summary.json"), "w"), indent=2)
    lams = [f"{l:g}" for l in LAMBDAS]
    wl2 = [aggB[k]["final_wl2_mean"] for k in lams]; dec = [aggB[k]["decay_mean"] for k in lams]
    fig, ax = plt.subplots(figsize=(7.4, 5))
    ax.errorbar(wl2, dec, yerr=[aggB[k]["decay_std"] for k in lams], fmt="o-",
                color="#23201B", ecolor="#B07D48", capsize=3, linewidth=1.6, markersize=7)
    for k, x, y in zip(lams, wl2, dec):
        ax.annotate(f"$\\lambda$={k}", (x, y), fontsize=9, xytext=(6, 5),
                    textcoords="offset points", color="#6B6660")
    ax.set_xlabel(r"final weight norm $\|\theta\|$ (set by L2-init strength)")
    ax.set_ylabel("plasticity decay (peak $-$ end)")
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "e22_lambda_sweep.png"), dpi=300); plt.close(fig)

    # ---------- C. ANR component ablation ----------
    print("\n=== C. ANR component ablation ===")
    ABL = [
        ("sgd", dict(method="sgd")),
        ("shrink_pert", dict(method="shrink_pert", shrink=0.7, perturb=0.01)),
        ("anr_shrink_only", dict(method="adaptive_norm_revive", anr_target_norm=3.0, anr_mode="shrink")),
        ("anr_revive_only", dict(method="adaptive_norm_revive", anr_target_norm=3.0, anr_mode="revive")),
        ("anr_full",        dict(method="adaptive_norm_revive", anr_target_norm=3.0, anr_mode="full")),
    ]
    rawC = {n: [] for n, _ in ABL}
    for seed in SEEDS:
        stream = stream_fn(seed)
        for name, kw in ABL:
            r = run_arm(stream, hidden=16, seed=seed, device=device, verbose=False, **RUN_BASE, **kw)
            rawC[name].append({
                "mean_acc": r["mean_acc"], "decay": pe_decay(r["task_test_acc"]),
                "final_wl2": r["diag_weight_l2"][-1], "final_dormant": r["diag_dormant"][-1]})
            print(f"  [anr-abl] seed={seed} {name:16} acc={r['mean_acc']:.3f} "
                  f"decay={rawC[name][-1]['decay']:+.3f} dorm={r['diag_dormant'][-1]:.3f}")
    aggC = {n: agg_over_seeds(rawC[n]) for n, _ in ABL}
    outC = {"seeds": SEEDS, "aggregate": aggC}
    json.dump(outC, open(os.path.join(TABLES, "e22_anr_ablation_summary.json"), "w"), indent=2)
    fig, ax = plt.subplots(figsize=(7.4, 5))
    names = [n for n, _ in ABL]
    xs = np.arange(len(names))
    accs = [aggC[n]["mean_acc_mean"] for n in names]
    errs = [aggC[n]["mean_acc_std"] for n in names]
    ax.bar(xs, accs, yerr=errs, color=["#8a8a8a", "#B07D48", "#3E5C76", "#6B7B5A", "#2f2f2f"],
           capsize=3, width=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([DISPLAY.get(n, n) for n in names], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("mean accuracy (5 seeds)")
    ax.grid(alpha=0.2, axis="y", linewidth=0.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "e22_anr_ablation.png"), dpi=300); plt.close(fig)

    # ---------- console summary ----------
    print("\n" + "=" * 66)
    print("[E22-A] second arch 784-64-64-10 leaderboard (acc desc):")
    for n in sorted(aggA, key=lambda k: -aggA[k]["mean_acc_mean"]):
        a = aggA[n]
        print(f"  {n:22} acc={a['mean_acc_mean']:.3f}+-{a['mean_acc_std']:.3f} "
              f"decay={a['decay_mean']:+.3f} wl2={a['final_wl2_mean']:.1f}")
    print(f"  Pearson(norm,decay) across 7 arms = {rA:.3f}")
    print("\n[E22-B] lambda sweep (5 seeds):")
    for k in lams:
        a = aggB[k]
        print(f"  lam={k:>5}: wl2={a['final_wl2_mean']:5.1f} decay={a['decay_mean']:+.3f}"
              f"+-{a['decay_std']:.3f} acc={a['mean_acc_mean']:.3f}+-{a['mean_acc_std']:.3f}")
    print("\n[E22-C] ANR ablation (5 seeds):")
    for n in names:
        a = aggC[n]
        print(f"  {n:16} acc={a['mean_acc_mean']:.3f}+-{a['mean_acc_std']:.3f} "
              f"decay={a['decay_mean']:+.3f} dorm={a['final_dormant_mean']:.3f}")


if __name__ == "__main__":
    main()
