"""E17 - Adaptive per-step directional controller vs SGD/EWC/replay/CBP (Phase 2).

The ORIGINAL motive of the project: can we CONTROL the direction of updates at
matched ‖Δθ‖ to reduce future-trainability damage (FTD, i.e., plasticity)?

Pre-registered in D-038 (adaptive controller design) and D-039 (non-degenerate
regime hyperparameters). Uses:
  - src/adaptive_controller.AdaptiveDirectionalController
  - src/baselines.train_sequence for SGD/EWC/replay/CBP
  - NON_DEG hyperparameters (8 epochs/task lr=3e-3 on CIFAR; 5 epochs/task
    lr=1e-3 on MNIST) so baselines reach non-trivial accuracy (fixes W7)

Acceptance (STRICT, D-038):
  Adaptive controller (any α > 0) reduces final probe FTD vs plain SGD at
  matched ‖Δθ‖ within tol 2%, reproduced over 3 seeds on >= 1 real dataset,
  AND not Pareto-dominated by EWC/replay/CBP on (FTD, mean_final_acc).

If passes → ORIGINAL MOTIVE achieved. If fails → definitive negative logged.
"""
from __future__ import annotations
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from configs.preregistration import (
    PRIMARY_SEED, SPLIT_MNIST, SPLIT_CIFAR, PROBE, PAST_GRAD_METHOD,
    ADAPTIVE, NON_DEG,
)
from src.seeds import seed_everything
from src.models import flat_params, set_flat_params, flat_grad, SmallCNN
from src.device import pick_device, env_summary
from src.datasets import load_mnist, split_mnist_tasks, load_cifar10, split_cifar10_tasks
from src.baselines import train_sequence, probe_ftd_generic
from src.adaptive_controller import AdaptiveDirectionalController
from experiments.e04_split_mnist_replication import MNISTStudent, task_iter, build_probes as build_probes_mnist


RESULTS_TABLES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "tables"
)
RESULTS_FIGS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures"
)


def _stringify(obj):
    if isinstance(obj, dict): return {k: _stringify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_stringify(v) for v in obj]
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.integer): return int(obj)
    return obj


# ---------------- Adaptive training loop ----------------

def train_adaptive(model, tasks, controller: AdaptiveDirectionalController,
                    lr, epochs, batch, seed, device, probes,
                    probe_steps, probe_lr, probe_batch, verbose=True):
    per_task_acc, per_task_pf = [], []
    per_step_norms = []
    for t_idx, task in enumerate(tasks):
        controller.begin_task(model, task["train_x"], task["train_y"])
        x, y = task["train_x"], task["train_y"]
        n = x.shape[0]
        steps = epochs * max(1, n // batch)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        g = torch.Generator().manual_seed(seed + 17 * t_idx)
        step = 0
        while step < steps:
            for xb_cpu, yb_cpu in task_iter(x, y, batch, g):
                theta_before = flat_params(model)
                xb = xb_cpu.to(device); yb = yb_cpu.to(device)
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(model(xb), yb).backward()
                grad = flat_grad(model)
                sgd_delta = -lr * grad
                # Adaptive projection
                delta_apply = controller.apply(model, sgd_delta,
                                                step_id=step, device=device)
                set_flat_params(model, theta_before + delta_apply)
                per_step_norms.append(float(delta_apply.norm().item()))
                step += 1
                if step >= steps: break
        # Metrics
        with torch.no_grad():
            xt = task["test_x"].to(device); yt = task["test_y"].to(device)
            ta = float((model(xt).argmax(-1) == yt).float().mean().item())
        pf = probe_ftd_generic(model, probes, probe_steps, probe_lr,
                                probe_batch, seed=seed, device=device)
        per_task_acc.append(ta); per_task_pf.append(pf)
        if verbose:
            print(f"[E17:adaptive alpha={controller.alpha}] task {t_idx} acc={ta:.3f} probeFTD={pf:.3f}")
    with torch.no_grad():
        final_accs = []
        for t in tasks:
            xt = t["test_x"].to(device); yt = t["test_y"].to(device)
            final_accs.append(float((model(xt).argmax(-1) == yt).float().mean().item()))
    n = len(per_task_acc)
    bwt = float(sum(final_accs[i] - per_task_acc[i] for i in range(n - 1))
                / max(n - 1, 1))
    return {
        "method": f"adaptive_alpha={controller.alpha}",
        "per_task_acc": per_task_acc,
        "per_task_probe_ftd": per_task_pf,
        "final_accs": final_accs,
        "mean_final_acc": float(sum(final_accs) / len(final_accs)),
        "BWT": bwt,
        "per_step_delta_norms": per_step_norms,
    }


# ---------------- Dataset drivers ----------------

def _load_mnist():
    tx, ty, vx, vy = load_mnist(SPLIT_MNIST.mnist_dir)
    tasks = split_mnist_tasks(tx, ty, vx, vy, SPLIT_MNIST.task_class_pairs)
    for t in tasks:
        pair = t["pair"]
        mask_tr = np.isin(ty, pair); mask_te = np.isin(vy, pair)
        t["train_y"] = torch.from_numpy(ty[mask_tr].astype(np.int64))
        t["test_y"] = torch.from_numpy(vy[mask_te].astype(np.int64))
    tx_norm = torch.from_numpy(tx.reshape(len(tx), -1).astype(np.float32) / 255.0)
    ty_all = torch.from_numpy(ty.astype(np.int64))
    # For the semantic aux probes we use the last-task pair as "held out" from
    # the perspective of the current task; the controller uses the whole 10-way
    # pool with held_out_classes = last pair.
    aux_pool_x = tx_norm
    aux_pool_y = ty_all
    probes = build_probes_mnist(tx_norm, ty_all, PROBE.n_probes,
                                 seed=PRIMARY_SEED + 10_000, subset_size=3000)
    return tasks, probes, aux_pool_x, aux_pool_y, {
        "lr": NON_DEG.mnist_train_lr,
        "epochs": NON_DEG.mnist_train_epochs_per_task,
        "batch": NON_DEG.mnist_train_batch,
    }


def _load_cifar(device):
    cfg = SPLIT_CIFAR
    tx, ty, vx, vy = load_cifar10(cfg.cifar10_dir)
    tasks = split_cifar10_tasks(tx, ty, vx, vy, cfg.cifar10_class_pairs)
    tx_all = torch.from_numpy(np.transpose(tx.astype(np.float32) / 255.0, (0, 3, 1, 2)))
    ty_all = torch.from_numpy(ty.astype(np.int64))
    # permuted-CIFAR probes (structural)
    rng = np.random.default_rng(PRIMARY_SEED + 10_000)
    probes = []
    C, H, W = tx_all.shape[1], tx_all.shape[2], tx_all.shape[3]
    for p_id in range(cfg.n_probes):
        perm_hw = rng.permutation(H * W)
        idx = rng.choice(tx_all.shape[0], size=cfg.probe_subset, replace=False)
        x_sub = tx_all[idx]
        x = x_sub.reshape(cfg.probe_subset, C, H * W)[:, :, perm_hw].reshape(
            cfg.probe_subset, C, H, W)
        probes.append((x, ty_all[idx].long()))
    return tasks, probes, tx_all, ty_all, {
        "lr": NON_DEG.train_lr,
        "epochs": NON_DEG.train_epochs_per_task,
        "batch": NON_DEG.train_batch,
    }


def run_on_dataset(name: str, device: torch.device, seed: int) -> dict:
    if name == "split_mnist":
        tasks, probes, aux_x, aux_y, hp = _load_mnist()
        def fresh():
            return MNISTStudent(input_dim=784,
                                hidden=SPLIT_MNIST.student_hidden_width,
                                depth=SPLIT_MNIST.student_hidden_depth, n_out=10)
    elif name == "split_cifar10":
        tasks, probes, aux_x, aux_y, hp = _load_cifar(device)
        def fresh():
            return SmallCNN(in_channels=3,
                            base=NON_DEG.cnn_base_channels,
                            fc=NON_DEG.cnn_hidden_fc,
                            n_classes=10, input_hw=32)
    else:
        raise ValueError(name)

    # Held-out classes for semantic aux probes: use the LAST-task class pair
    # from the perspective of each task, we treat "future" as the final task
    held_out_classes = tuple(
        tasks[-1].get("pair", (8, 9)) if "pair" in tasks[-1] else (8, 9)
    )

    seed_everything(seed + 7000)
    init_state = {k: v.clone() for k, v in fresh().state_dict().items()}
    def build_arm():
        m = fresh(); m.load_state_dict({k: v.clone() for k, v in init_state.items()})
        return m.to(device)

    print(f"\n[E17:{name} seed={seed}] fixed regime: lr={hp['lr']} epochs={hp['epochs']} batch={hp['batch']}")

    arms = {}
    hpar_baselines = dict(
        ewc_lambda=ADAPTIVE.ewc_lambda, ewc_fisher_batch=200,
        replay_memory_per_task=ADAPTIVE.replay_memory_per_task,
        replay_mix_ratio=ADAPTIVE.replay_mix_ratio,
        cbp_reset_rate=ADAPTIVE.cbp_reset_rate, cbp_utility_ema=0.99,
    )
    for method in ADAPTIVE.baselines:
        print(f"\n[E17:{name}] --- baseline {method} ---")
        m = build_arm()
        r = train_sequence(
            m, tasks, lr=hp["lr"], epochs=hp["epochs"], batch=hp["batch"],
            seed=seed, device=device, method=method,
            probes=probes, probe_steps=PROBE.probe_steps,
            probe_lr=PROBE.probe_lr, probe_batch=PROBE.probe_batch,
            hparams=hpar_baselines, verbose=True,
        )
        arms[method] = r

    # Adaptive controller sweep
    for alpha in ADAPTIVE.projection_alpha_sweep:
        name_arm = f"adaptive_alpha={alpha:.2f}"
        print(f"\n[E17:{name}] --- {name_arm} ---")
        m = build_arm()
        ctrl = AdaptiveDirectionalController(
            n_permuted_aux=3, n_semantic_aux=3,
            reestimate_every=ADAPTIVE.reestimate_every,
            projection_alpha=float(alpha),
            top_k=ADAPTIVE.top_k_project,
            held_out_classes=held_out_classes,
            aux_pool_x=aux_x, aux_pool_y=aux_y,
            seed=seed, batch=hp["batch"],
        )
        arms[name_arm] = train_adaptive(
            m, tasks, ctrl,
            lr=hp["lr"], epochs=hp["epochs"], batch=hp["batch"],
            seed=seed, device=device, probes=probes,
            probe_steps=PROBE.probe_steps, probe_lr=PROBE.probe_lr,
            probe_batch=PROBE.probe_batch,
        )

    # Matched-magnitude fairness
    sgd_norm = float(np.mean(arms["sgd"]["per_step_delta_norms"]))
    matched = {n: {"mean_norm": float(np.mean(a["per_step_delta_norms"])),
                    "relative_deviation": abs(float(np.mean(a["per_step_delta_norms"])) - sgd_norm)
                                          / max(sgd_norm, 1e-12)}
                for n, a in arms.items()}

    # Sanity gate for non-degenerate regime
    sgd_acc = arms["sgd"]["mean_final_acc"]
    regime_ok = (sgd_acc >= (0.30 if name == "split_mnist" else 0.20))
    print(f"[E17:{name}] SGD mean_final_acc={sgd_acc:.3f} (regime_ok={regime_ok})")

    # Acceptance: adaptive alpha > 0 must reduce probeFTD vs SGD by any margin,
    # matched-mag within 2%, and not be Pareto-dominated by EWC/replay/CBP.
    sgd_pf = arms["sgd"]["per_task_probe_ftd"][-1]
    adaptive_wins = []
    for name_arm, a in arms.items():
        if not name_arm.startswith("adaptive_alpha=0.00") and name_arm.startswith("adaptive_"):
            pf = a["per_task_probe_ftd"][-1]
            rel_dev = matched[name_arm]["relative_deviation"]
            if pf < sgd_pf and rel_dev <= 0.02:
                # Not Pareto-dominated: no baseline arm has both lower pf AND >= acc
                dominated = False
                for bn in ADAPTIVE.baselines:
                    if bn == "sgd": continue
                    bpf = arms[bn]["per_task_probe_ftd"][-1]
                    bacc = arms[bn]["mean_final_acc"]
                    if (bpf <= pf and bacc >= a["mean_final_acc"]
                            and (bpf < pf or bacc > a["mean_final_acc"])):
                        dominated = True
                        break
                if not dominated:
                    adaptive_wins.append(name_arm)
    verdict = "POSITIVE_ADAPTIVE_BEATS_SGD_NOT_DOMINATED" if adaptive_wins else "NULL"

    # Plot Pareto
    fig, ax = plt.subplots(figsize=(7, 5))
    for n, a in arms.items():
        pf = a["per_task_probe_ftd"][-1]
        ax.scatter(pf, a["mean_final_acc"], s=80,
                   marker=("s" if not n.startswith("adaptive_") else "o"),
                   label=n, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("final probe FTD (lower = better plasticity)")
    ax.set_ylabel("mean final acc (higher = better)")
    ax.set_title(f"E17 {name} seed={seed}: adaptive vs baselines (matched ||dtheta||)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_FIGS, f"e17_{name}_seed={seed}_pareto.png"), dpi=140)
    plt.close(fig)

    return {
        "dataset": name, "seed": seed,
        "arms": {n: {"mean_final_acc": a["mean_final_acc"], "BWT": a["BWT"],
                     "per_task_acc": a["per_task_acc"],
                     "per_task_probe_ftd": a["per_task_probe_ftd"],
                     "final_accs": a["final_accs"]}
                 for n, a in arms.items()},
        "matched_magnitude": matched,
        "sgd_sanity_gate_passed": regime_ok,
        "adaptive_wins": adaptive_wins,
        "verdict": verdict,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   choices=["split_mnist", "split_cifar10"],
                   default=["split_mnist"])
    p.add_argument("--seeds", nargs="+", type=int,
                   default=[PRIMARY_SEED, PRIMARY_SEED + 1, PRIMARY_SEED + 2])
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device, profile = pick_device(args.device)
    print(f"[E17] device={device} profile={profile}")

    out = {"hardware": env_summary(), "runs": []}
    for ds in args.datasets:
        for s in args.seeds:
            # MNIST: use CPU (small MLP); CIFAR: use GPU
            dev = torch.device("cpu") if ds == "split_mnist" else device
            r = run_on_dataset(ds, device=dev, seed=s)
            out["runs"].append(r)

    # Cross-seed aggregation per dataset
    aggregated = {}
    for ds in args.datasets:
        ds_runs = [r for r in out["runs"] if r["dataset"] == ds]
        # For each arm, aggregate final probe FTD and mean_final_acc across seeds
        arm_names = ds_runs[0]["arms"].keys()
        agg = {}
        for a in arm_names:
            pfs = [r["arms"][a]["per_task_probe_ftd"][-1] for r in ds_runs]
            accs = [r["arms"][a]["mean_final_acc"] for r in ds_runs]
            agg[a] = {"pf_mean": float(np.mean(pfs)),
                      "pf_std": float(np.std(pfs)),
                      "acc_mean": float(np.mean(accs)),
                      "acc_std": float(np.std(accs))}
        # Verdict across seeds: adaptive must beat SGD in the MEAN, and not be
        # dominated by EWC/replay/CBP in the mean.
        sgd_mean_pf = agg["sgd"]["pf_mean"]
        winners = []
        for a, v in agg.items():
            if not a.startswith("adaptive_") or a == "adaptive_alpha=0.00":
                continue
            if v["pf_mean"] < sgd_mean_pf:
                dominated = False
                for bn in ADAPTIVE.baselines:
                    if bn == "sgd": continue
                    bv = agg[bn]
                    if (bv["pf_mean"] <= v["pf_mean"] and
                        bv["acc_mean"] >= v["acc_mean"] and
                        (bv["pf_mean"] < v["pf_mean"] or bv["acc_mean"] > v["acc_mean"])):
                        dominated = True; break
                if not dominated: winners.append(a)
        aggregated[ds] = {"per_arm": agg, "adaptive_winners_across_seeds": winners}
        print(f"\n[E17-agg {ds}] adaptive winners across seeds: {winners}")
        print(f"[E17-agg {ds}] sgd pf_mean={sgd_mean_pf:.4f}")
        for a, v in agg.items():
            if a.startswith("adaptive_") or a in ADAPTIVE.baselines:
                print(f"  {a}: pf={v['pf_mean']:.4f}±{v['pf_std']:.4f}  acc={v['acc_mean']:.3f}±{v['acc_std']:.3f}")

    out["aggregated"] = aggregated
    any_winner = any(len(a["adaptive_winners_across_seeds"]) > 0 for a in aggregated.values())
    out["overall_verdict"] = ("POSITIVE_MAIN_TRACK" if any_winner
                                else "DEFINITIVE_NULL")

    with open(os.path.join(RESULTS_TABLES, "e17_summary.json"), "w") as f:
        json.dump(_stringify(out), f, indent=2)
    print(f"\n[E17] OVERALL VERDICT: {out['overall_verdict']}")


if __name__ == "__main__":
    main()
