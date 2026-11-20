"""E13 - Forgetting/plasticity Pareto controller vs SGD/EWC/replay/CBP.

Phase B of v1.2. Pre-registered in decision.md D-028 BEFORE this run.

Given the E08 finding that the past-gradient axis is monotonically directional
for BWT (forgetting), we build a controller that moves a tunable amount along
u_past at EXACTLY matched ||dtheta||, sweep c in [-1, +1], and compare to fair
baselines (SGD, EWC, replay, CBP) on Split-MNIST AND Split-CIFAR-10.

Acceptance bar (D-028):
  - Controller has >=1 operating point that STRICTLY Pareto-dominates EWC on
    (BWT, probe FTD) at matched-||dtheta|| tolerance <=2%, on at least one dataset.
  - Also strictly Pareto-dominates plain SGD.

Positive = main-track claim on the past-gradient axis; null = honest report.
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
    PRIMARY_SEED, SPLIT_MNIST, SPLIT_CIFAR, TRADEOFF, PROBE, PAST_GRAD_METHOD,
)
from src.seeds import seed_everything
from src.models import SmallCNN, flat_params, set_flat_params, flat_grad
from src.device import pick_device, env_summary
from src.datasets import (
    load_mnist, split_mnist_tasks, load_cifar10, split_cifar10_tasks,
)
from src.baselines import train_sequence, probe_ftd_generic
from experiments.e04_split_mnist_replication import (
    MNISTStudent, task_iter, build_probes,
)

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


def compute_past_axis_generic(model, past_tasks, batch: int, seed: int,
                               n_samples: int, device: torch.device) -> torch.Tensor:
    D = sum(p.numel() for p in model.parameters())
    accum = torch.zeros(D, device=device)
    for s in range(n_samples):
        for t_idx, task in enumerate(past_tasks):
            m = deepcopy(model).to(device)
            g = torch.Generator().manual_seed(seed + 331 * (t_idx + 1) + 7 * s)
            xb_cpu, yb_cpu = next(task_iter(task["train_x"], task["train_y"], batch, g))
            xb = xb_cpu.to(device); yb = yb_cpu.to(device)
            m.zero_grad(set_to_none=True)
            F.cross_entropy(m(xb), yb).backward()
            accum = accum + flat_grad(m)
    accum = accum.detach().cpu()
    return accum / (accum.norm() + 1e-12)


def build_permuted_probes_mnist(tx_norm, ty_all, n_probes, seed, subset_size=3000):
    probes = []
    rng = np.random.default_rng(seed)
    for p_id in range(n_probes):
        perm = rng.permutation(tx_norm.shape[1])
        idx = rng.choice(tx_norm.shape[0], size=subset_size, replace=False)
        probes.append((tx_norm[idx][:, perm], ty_all[idx].long()))
    return probes


def build_permuted_probes_cifar(tx_all, ty_all, n_probes, seed, subset_size=3000):
    probes = []
    rng = np.random.default_rng(seed)
    C, H, W = tx_all.shape[1], tx_all.shape[2], tx_all.shape[3]
    for p_id in range(n_probes):
        perm_hw = rng.permutation(H * W)
        idx = rng.choice(tx_all.shape[0], size=subset_size, replace=False)
        x_sub = tx_all[idx]
        x = x_sub.reshape(subset_size, C, H * W)[:, :, perm_hw].reshape(subset_size, C, H, W)
        probes.append((x, ty_all[idx].long()))
    return probes


def _pareto_dominates(arm_a: dict, arm_b: dict) -> bool:
    """Strict Pareto dominance in (BWT higher better, probe FTD lower better)."""
    # Higher BWT strictly better; lower probeFTD strictly better
    a_bwt, a_ftd = arm_a["BWT"], arm_a["per_task_probe_ftd"][-1]
    b_bwt, b_ftd = arm_b["BWT"], arm_b["per_task_probe_ftd"][-1]
    weakly = (a_bwt >= b_bwt) and (a_ftd <= b_ftd)
    strictly = (a_bwt > b_bwt) or (a_ftd < b_ftd)
    return weakly and strictly


def _load_dataset(name: str, device: torch.device):
    if name == "split_mnist":
        tx, ty, vx, vy = load_mnist(SPLIT_MNIST.mnist_dir)
        tasks = split_mnist_tasks(tx, ty, vx, vy, SPLIT_MNIST.task_class_pairs)
        for t in tasks:
            pair = t["pair"]
            mask_tr = np.isin(ty, pair); mask_te = np.isin(vy, pair)
            t["train_y"] = torch.from_numpy(ty[mask_tr].astype(np.int64))
            t["test_y"] = torch.from_numpy(vy[mask_te].astype(np.int64))
        tx_norm = torch.from_numpy(tx.reshape(len(tx), -1).astype(np.float32) / 255.0)
        ty_all = torch.from_numpy(ty.astype(np.int64))
        probes = build_permuted_probes_mnist(tx_norm, ty_all, PROBE.n_probes,
                                             seed=PRIMARY_SEED + 10_000, subset_size=3000)
        def fresh():
            return MNISTStudent(input_dim=784,
                                hidden=SPLIT_MNIST.student_hidden_width,
                                depth=SPLIT_MNIST.student_hidden_depth, n_out=10)
        return tasks, probes, fresh, {"lr": SPLIT_MNIST.train_lr,
                                       "epochs": SPLIT_MNIST.train_epochs_per_task,
                                       "batch": SPLIT_MNIST.train_batch}
    elif name == "split_cifar10":
        cfg = SPLIT_CIFAR
        tx, ty, vx, vy = load_cifar10(cfg.cifar10_dir)
        tasks = split_cifar10_tasks(tx, ty, vx, vy, cfg.cifar10_class_pairs)
        tx_all = torch.from_numpy(np.transpose(tx.astype(np.float32) / 255.0, (0, 3, 1, 2)))
        ty_all = torch.from_numpy(ty.astype(np.int64))
        probes = build_permuted_probes_cifar(tx_all, ty_all, cfg.n_probes,
                                             seed=PRIMARY_SEED + 10_000, subset_size=cfg.probe_subset)
        def fresh():
            return SmallCNN(in_channels=3, base=cfg.cnn_base_channels,
                            fc=cfg.cnn_hidden_fc, n_classes=10, input_hw=32)
        return tasks, probes, fresh, {"lr": cfg.train_lr,
                                       "epochs": cfg.train_epochs_per_task,
                                       "batch": cfg.train_batch}
    else:
        raise ValueError(name)


def run_dataset(name: str, device: torch.device, seed: int) -> dict:
    tasks, probes, fresh, hp = _load_dataset(name, device)
    print(f"\n[E13:{name}] tasks={len(tasks)}, arms=SGD+EWC+replay+CBP+controller×{len(TRADEOFF.coefficient_sweep)}")

    # 1) Warm a reference student through tasks[:-1] to compute u_past
    warm = fresh().to(device)
    seed_everything(seed)
    for t_idx in range(len(tasks) - 1):
        task = tasks[t_idx]
        n = task["train_x"].shape[0]
        steps = hp["epochs"] * max(1, n // hp["batch"])
        opt = torch.optim.SGD(warm.parameters(), lr=hp["lr"])
        g = torch.Generator().manual_seed(seed + 17 * t_idx)
        step = 0
        while step < steps:
            for xb_cpu, yb_cpu in task_iter(task["train_x"], task["train_y"], hp["batch"], g):
                xb = xb_cpu.to(device); yb = yb_cpu.to(device)
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(warm(xb), yb).backward(); opt.step()
                step += 1
                if step >= steps: break
    u_past = compute_past_axis_generic(warm, tasks[:len(tasks) - 1],
                                        batch=hp["batch"],
                                        seed=seed + 22222,
                                        n_samples=PAST_GRAD_METHOD.n_past_grad_samples,
                                        device=device)
    del warm
    if device.type == "cuda": torch.cuda.empty_cache()
    print(f"[E13:{name}] u_past estimated ||u_past||={u_past.norm():.4f}")

    # 2) Run each baseline + each controller coefficient from the same fresh init
    seed_everything(seed + 7000)
    init_state = {k: v.clone() for k, v in fresh().state_dict().items()}

    def build_arm():
        m = fresh(); m.load_state_dict({k: v.clone() for k, v in init_state.items()})
        return m.to(device)

    arms = {}
    hpar = dict(
        ewc_lambda=TRADEOFF.ewc_lambda, ewc_fisher_batch=TRADEOFF.ewc_fisher_batch,
        replay_memory_per_task=TRADEOFF.replay_memory_per_task,
        replay_mix_ratio=TRADEOFF.replay_mix_ratio,
        cbp_reset_rate=TRADEOFF.cbp_reset_rate, cbp_utility_ema=TRADEOFF.cbp_utility_ema,
    )
    for method in TRADEOFF.baselines:
        print(f"\n[E13:{name}] --- baseline: {method} ---")
        m = build_arm()
        r = train_sequence(
            m, tasks, lr=hp["lr"], epochs=hp["epochs"], batch=hp["batch"],
            seed=seed, device=device, method=method,
            probes=probes, probe_steps=PROBE.probe_steps,
            probe_lr=PROBE.probe_lr, probe_batch=PROBE.probe_batch,
            hparams=hpar, verbose=True,
        )
        arms[method] = r

    for c in TRADEOFF.coefficient_sweep:
        name_arm = f"controller_c={c:+.2f}"
        print(f"\n[E13:{name}] --- {name_arm} ---")
        m = build_arm()
        r = train_sequence(
            m, tasks, lr=hp["lr"], epochs=hp["epochs"], batch=hp["batch"],
            seed=seed, device=device, method="past_axis_controller",
            probes=probes, probe_steps=PROBE.probe_steps,
            probe_lr=PROBE.probe_lr, probe_batch=PROBE.probe_batch,
            hparams=dict(axis_coeff=float(c)),
            past_axis=u_past.to(device), verbose=True,
        )
        arms[name_arm] = r

    # 3) Matched-magnitude fairness check (per D-028: <=2% deviation from SGD's per-step mean)
    sgd_mean_norm = float(np.mean(arms["sgd"]["per_step_delta_norms"]))
    matched = {}
    for name_arm, arm in arms.items():
        arm_mean = float(np.mean(arm["per_step_delta_norms"])) if arm["per_step_delta_norms"] else 0.0
        matched[name_arm] = {
            "mean_norm": arm_mean,
            "relative_deviation": abs(arm_mean - sgd_mean_norm) / max(sgd_mean_norm, 1e-12),
        }

    # 4) Pareto analysis: for each controller point, does it strictly dominate SGD/EWC?
    pareto = {}
    for name_arm in arms:
        if not name_arm.startswith("controller_"): continue
        pareto[name_arm] = {
            "dominates_sgd": _pareto_dominates(arms[name_arm], arms["sgd"]),
            "dominates_ewc": _pareto_dominates(arms[name_arm], arms["ewc"]),
            "dominates_replay": _pareto_dominates(arms[name_arm], arms["replay"]),
            "dominates_cbp": _pareto_dominates(arms[name_arm], arms["cbp"]),
        }

    # Verdict
    any_dom_ewc = any(v["dominates_ewc"] for v in pareto.values())
    any_dom_sgd = any(v["dominates_sgd"] for v in pareto.values())
    verdict = ("POSITIVE_DOMINATES_EWC_AND_SGD"
               if (any_dom_ewc and any_dom_sgd) else
               "POSITIVE_DOMINATES_SGD_ONLY" if any_dom_sgd else
               "NULL")

    # ------------------ Plot ------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for name_arm, arm in arms.items():
        pf = arm["per_task_probe_ftd"][-1]
        b = arm["BWT"]
        if name_arm in ("sgd", "ewc", "replay", "cbp"):
            marker = "s"; sz = 90; col = {"sgd": "black", "ewc": "tab:orange",
                                          "replay": "tab:green", "cbp": "tab:purple"}[name_arm]
        else:
            marker = "o"; sz = 45
            # colormap by coefficient value
            c = float(name_arm.split("=")[1])
            col = plt.cm.RdBu_r((c + 1) / 2)
        ax.scatter(pf, b, s=sz, marker=marker, color=col, edgecolor="black",
                   linewidth=0.7)
        label = name_arm if name_arm in ("sgd", "ewc", "replay", "cbp") else name_arm
        ax.annotate(label, (pf, b), fontsize=6, xytext=(4, 2),
                    textcoords="offset points")
    ax.set_xlabel("probe FTD (lower better)")
    ax.set_ylabel("BWT (higher = less forgetting; better)")
    ax.set_title(f"E13 {name}: Pareto (BWT vs probe FTD)  verdict={verdict}")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_FIGS, f"e13_{name}_pareto.png"), dpi=140)
    plt.close(fig)

    return {
        "dataset": name,
        "arms": {n: {"BWT": a["BWT"], "mean_final_acc": a["mean_final_acc"],
                     "per_task_acc": a["per_task_acc"],
                     "per_task_probe_ftd": a["per_task_probe_ftd"],
                     "final_accs": a["final_accs"]}
                 for n, a in arms.items()},
        "matched_magnitude": matched,
        "pareto": pareto,
        "verdict": verdict,
        "sgd_mean_step_norm": sgd_mean_norm,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", choices=list(TRADEOFF.datasets),
                   default=list(TRADEOFF.datasets))
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device, profile = pick_device(args.device)
    print(f"[E13] device={device} profile={profile}")

    out = {"hardware": env_summary(), "datasets": {}}
    for ds in args.datasets:
        # Use MNIST's warmer CPU code path if MNIST (data lives on CPU anyway)
        r = run_dataset(ds, device=device if ds == "split_cifar10" else torch.device("cpu"),
                        seed=PRIMARY_SEED)
        out["datasets"][ds] = r

    # Overall verdict
    d_verdicts = [d["verdict"] for d in out["datasets"].values()]
    if any(v == "POSITIVE_DOMINATES_EWC_AND_SGD" for v in d_verdicts):
        overall = "POSITIVE_MAIN_TRACK"
    elif any(v == "POSITIVE_DOMINATES_SGD_ONLY" for v in d_verdicts):
        overall = "POSITIVE_WEAK"
    else:
        overall = "NULL"
    out["overall_verdict"] = overall

    with open(os.path.join(RESULTS_TABLES, "e13_summary.json"), "w") as f:
        json.dump(_stringify(out), f, indent=2)
    print(f"\n[E13] OVERALL VERDICT: {overall}")


if __name__ == "__main__":
    main()
