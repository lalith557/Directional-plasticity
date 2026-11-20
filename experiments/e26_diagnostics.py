"""e26 - Layer-wise norm breakdown (Exp B) + alternative plasticity metrics (Exp C).

All real data, width-16 permuted MNIST, 3 seeds. (i) Which layer(s) drive global
norm growth? (ii) Do the main conclusions survive alternative plasticity metrics
(last-task accuracy, degradation slope, mean accuracy) beyond the peak-to-tail decay?
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src.datasets import load_mnist
from src.plasticity_stress import PermutedMNIST, PlasticMLP, run_arm, _backbone_params

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "results", "tables")
RUN = dict(lr=0.2, epochs=1, batch=128, steps_per_task=60, hidden=16, depth=2)
SEEDS = [20260824, 20260825, 20260826]
ARMS = [("sgd", {}), ("l2_init", {"l2_init_lambda": 1e-2}),
        ("shrink_pert", {"shrink": 0.7, "perturb": 0.01}),
        ("cbp", {"cbp_reset_rate": 1e-2}),
        ("dir_control", {"dir_coeff": 1.0}), ("dir_inward", {"dir_coeff": 0.6}),
        ("adaptive_norm_revive", {"anr_target_norm": 3.0})]


def metrics(ta):
    ta = np.array(ta); n = len(ta); k = max(1, n // 10)
    peak = float(ta.max()); tail = float(ta[-k:].mean())
    slope = float(np.polyfit(np.arange(n), ta, 1)[0])  # per-task accuracy slope
    return {"decay_peak_tail": peak - tail, "mean_acc": float(ta.mean()),
            "last_task_acc": float(ta[-1]), "late10_acc": tail,
            "slope_per_task": slope}


def main():
    tx, ty, vx, vy = load_mnist("Deep Learning Dataset1")
    dev = torch.device("cpu")
    # initial per-layer norms (deterministic-ish; average a few inits)
    init_layers = []
    for s in SEEDS:
        torch.manual_seed(s)
        m = PlasticMLP(784, 16, 2, 10)
        init_layers.append([float(l.weight.data.norm()) for l in m.hidden_layers]
                           + [float(m.head.weight.data.norm())])
    init_layers = np.array(init_layers).mean(0)

    alt = {n: [] for n, _ in ARMS}
    layer_final = {n: [] for n, _ in ARMS}
    for s in SEEDS:
        stream = PermutedMNIST(tx, ty, vx, vy, n_tasks=100, seed=s, train_subset=6000)
        for name, kw in ARMS:
            r = run_arm(stream, name, seed=s, device=dev, verbose=False, **RUN, **kw)
            alt[name].append(metrics(r["task_test_acc"]))
            layer_final[name].append(r["layer_norms_final"])
            print(f"  seed={s} {name:20} mean={alt[name][-1]['mean_acc']:.3f} "
                  f"last={alt[name][-1]['last_task_acc']:.3f} slope={alt[name][-1]['slope_per_task']:+.4f} "
                  f"layers={[round(x,1) for x in r['layer_norms_final']]}")

    def agg(rows): return {k: float(np.mean([x[k] for x in rows])) for k in rows[0]}
    alt_agg = {n: agg(alt[n]) for n, _ in ARMS}
    layer_agg = {n: np.array(layer_final[n]).mean(0).tolist() for n, _ in ARMS}

    out = {"seeds": SEEDS, "init_layer_norms": init_layers.tolist(),
           "layer_labels": ["hidden1", "hidden2", "head"],
           "alt_metrics": alt_agg, "final_layer_norms": layer_agg}
    json.dump(out, open(os.path.join(TAB, "e26_summary.json"), "w"), indent=2)

    print("\n[E26] initial per-layer norms:", [round(x, 2) for x in init_layers])
    print("[E26] final per-layer norms (hidden1, hidden2, head):")
    for n, _ in ARMS:
        print(f"  {n:20} {[round(x,1) for x in layer_agg[n]]}")
    print("\n[E26] alternative metrics (mean over 3 seeds):")
    print(f"  {'arm':20} {'decay':>7} {'meanAcc':>8} {'lastAcc':>8} {'slope':>9}")
    for n in sorted(alt_agg, key=lambda k: -alt_agg[k]['mean_acc']):
        a = alt_agg[n]
        print(f"  {n:20} {a['decay_peak_tail']:+7.3f} {a['mean_acc']:8.3f} {a['last_task_acc']:8.3f} {a['slope_per_task']:+9.4f}")


if __name__ == "__main__":
    main()
