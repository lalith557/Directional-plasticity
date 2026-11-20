"""E08 - Past-gradient projection method (Phase 1, pre-registered D-020).

Motivation: E06/N1 established that under caliper matching, the ONLY directional
descriptor that survives the joint regression on Split-MNIST is `d_dir_past` -
alignment with the summed past-task gradient. E05/E06/N3's method attempts
(projection along +g_ref and empirical PCA) both failed to beat SGD at matched
||dtheta||. The unmarked lead from the paper's own data is: **build the method
around the past-gradient axis**, and test both projecting AWAY from it and TOWARD
it - the sign-inverted control converts a correlation into a causal-flavored
intervention test.

Arms on Split-MNIST (matched ||dtheta|| by construction via exact_renorm):
  A0 SGD                     - baseline
  A_past_avoid               - project OUT the past-gradient component
  A_past_toward              - keep only the past-gradient component, then renorm
  A_ewc                      - EWC anti-forgetting baseline (fair comparison)

If A_past_avoid beats SGD on plasticity AND A_past_toward hurts plasticity, that's
directional causation. If both help/hurt, that's a magnitude/anti-forgetting story
in disguise and this is another honest null.
"""
from __future__ import annotations
import sys, os, json
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
    PRIMARY_SEED, SPLIT_MNIST, PROBE, PAST_GRAD_METHOD, KILL,
)
from src.seeds import seed_everything
from src.models import flat_params, set_flat_params, flat_grad
from src.device import pick_device, env_summary
from src.datasets import load_mnist, split_mnist_tasks
from src.constraint import project_out
from experiments.e04_split_mnist_replication import (
    MNISTStudent, task_iter, eval_loss, build_probes, probe_ftd,
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


def eval_acc(model, x, y):
    with torch.no_grad():
        pred = model(x).argmax(dim=-1)
        return float((pred == y).float().mean().item())


def compute_past_axis(model, past_tasks, batch: int, seed: int,
                      n_samples: int = 3) -> torch.Tensor:
    """Averaged summed past-task gradient direction (unit vector)."""
    D = sum(p.numel() for p in model.parameters())
    accum = torch.zeros(D)
    for s in range(n_samples):
        for t_idx, task in enumerate(past_tasks):
            m = deepcopy(model)
            g = torch.Generator().manual_seed(seed + 331 * (t_idx + 1) + 7 * s)
            xb, yb = next(task_iter(task["train_x"], task["train_y"], batch, g))
            m.zero_grad(set_to_none=True)
            F.cross_entropy(m(xb), yb).backward()
            accum += flat_grad(m)
    return accum / (accum.norm() + 1e-12)


def compute_ewc_fisher(model, past_tasks, batch: int, seed: int) -> torch.Tensor:
    """Per-parameter Fisher = mean of squared gradients on past-task samples."""
    D = sum(p.numel() for p in model.parameters())
    fisher = torch.zeros(D)
    count = 0
    for t_idx, task in enumerate(past_tasks):
        m = deepcopy(model)
        g = torch.Generator().manual_seed(seed + 999 * (t_idx + 1))
        xb, yb = next(task_iter(task["train_x"], task["train_y"], batch, g))
        m.zero_grad(set_to_none=True)
        # Use per-example gradient of predicted label log-prob
        logits = m(xb)
        probs = F.softmax(logits, dim=-1)
        # log-likelihood of PREDICTED labels (empirical Fisher approximation)
        pred = logits.argmax(dim=-1)
        loss = F.nll_loss(F.log_softmax(logits, dim=-1), pred)
        loss.backward()
        fisher += flat_grad(m).pow(2)
        count += 1
    return fisher / max(count, 1)


def add_ewc_grad_to_model(model, fisher: torch.Tensor, theta_ref: torch.Tensor,
                          lam: float) -> None:
    """Add lam * F * (theta - theta_ref) to the current .grad of each param."""
    off = 0
    for p in model.parameters():
        n = p.numel()
        f_i = fisher[off:off + n].view_as(p)
        r_i = theta_ref[off:off + n].view_as(p)
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        p.grad.data.add_(lam * f_i * (p.data - r_i))
        off += n


def train_arm(model, tasks, lr, epochs, batch, seed,
              projection_mode: str, u_bad: torch.Tensor,
              probes, probe_cfg,
              ewc_fisher=None, ewc_theta_ref=None, ewc_lambda: float = 0.0,
              verbose: bool = True):
    """Sequentially train on all tasks. Return per-task loss/acc/probeFTD."""
    per_task_loss, per_task_acc, per_task_ftd = [], [], []
    # NB: u_bad is a unit vector on CPU-shaped like flat_params(model)
    for t_idx, task in enumerate(tasks):
        x, y = task["train_x"], task["train_y"]
        n = x.shape[0]
        steps = epochs * max(1, n // batch)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        g = torch.Generator().manual_seed(seed + 17 * t_idx)
        step = 0
        while step < steps:
            for xb, yb in task_iter(x, y, batch, g):
                theta_before = flat_params(model)
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(model(xb), yb).backward()
                # EWC penalty as an added grad
                if ewc_fisher is not None and ewc_theta_ref is not None and ewc_lambda > 0:
                    add_ewc_grad_to_model(model, ewc_fisher, ewc_theta_ref, ewc_lambda)
                grad = flat_grad(model)
                sgd_delta = -lr * grad
                orig_norm = sgd_delta.norm() + 1e-12
                if projection_mode == "none" or u_bad is None:
                    apply_delta = sgd_delta
                elif projection_mode == "avoid":
                    apply_delta = project_out(sgd_delta, u_bad, exact_renorm=True)
                elif projection_mode == "toward":
                    # keep only component along u_bad, then rescale to orig_norm
                    u = u_bad / (u_bad.norm() + 1e-12)
                    comp = (sgd_delta @ u) * u
                    apply_delta = comp * (orig_norm / (comp.norm() + 1e-12))
                else:
                    raise ValueError(projection_mode)
                set_flat_params(model, theta_before + apply_delta)
                step += 1
                if step >= steps: break
        tl = eval_loss(model, task["test_x"], task["test_y"])
        ta = eval_acc(model, task["test_x"], task["test_y"])
        pf = probe_ftd(model, probes, probe_cfg.probe_steps, probe_cfg.probe_lr,
                       probe_cfg.probe_batch, seed=seed)
        per_task_loss.append(tl); per_task_acc.append(ta); per_task_ftd.append(pf)
        if verbose:
            print(f"[E08:{projection_mode}] task {t_idx} pair={task['pair']} "
                  f"loss={tl:.3f} acc={ta:.3f} probeFTD={pf:.3f}")
    final_losses = [eval_loss(model, t["test_x"], t["test_y"]) for t in tasks]
    final_accs = [eval_acc(model, t["test_x"], t["test_y"]) for t in tasks]
    n = len(per_task_acc)
    bwt = float(sum(final_accs[i] - per_task_acc[i] for i in range(n - 1)) / max(n - 1, 1))
    return {"per_task_loss": per_task_loss, "per_task_acc": per_task_acc,
            "per_task_probe_ftd": per_task_ftd,
            "final_losses_all_tasks": final_losses,
            "final_accs_all_tasks": final_accs,
            "mean_final_acc": float(sum(final_accs) / len(final_accs)),
            "BWT": bwt}


def main() -> None:
    device, profile = pick_device("cpu")  # this MLP is faster on CPU than GPU-transfer
    print(f"[E08] device={device} profile={profile}")
    seed_everything(PRIMARY_SEED)

    tx, ty, vx, vy = load_mnist(SPLIT_MNIST.mnist_dir)
    tasks = split_mnist_tasks(tx, ty, vx, vy, SPLIT_MNIST.task_class_pairs)
    for task in tasks:
        pair = task["pair"]
        mask_tr = np.isin(ty, pair); mask_te = np.isin(vy, pair)
        task["train_y"] = torch.from_numpy(ty[mask_tr].astype(np.int64))
        task["test_y"] = torch.from_numpy(vy[mask_te].astype(np.int64))
    tx_norm = torch.from_numpy(tx.reshape(len(tx), -1).astype(np.float32) / 255.0)
    ty_all = torch.from_numpy(ty.astype(np.int64))
    probes = build_probes(tx_norm, ty_all, PROBE.n_probes,
                          seed=PRIMARY_SEED + 10_000, subset_size=3000)

    # Fixed init shared across arms
    seed_everything(PRIMARY_SEED + 7000)
    init_state = MNISTStudent(input_dim=784,
                              hidden=SPLIT_MNIST.student_hidden_width,
                              depth=SPLIT_MNIST.student_hidden_depth, n_out=10).state_dict()

    def fresh():
        m = MNISTStudent(input_dim=784,
                         hidden=SPLIT_MNIST.student_hidden_width,
                         depth=SPLIT_MNIST.student_hidden_depth, n_out=10)
        m.load_state_dict({k: v.clone() for k, v in init_state.items()})
        return m

    # Warm the reference student through tasks 0..3 (identical to E04 protocol)
    warm = fresh()
    for t_idx in range(4):
        task = tasks[t_idx]
        n = task["train_x"].shape[0]
        steps = SPLIT_MNIST.train_epochs_per_task * max(1, n // SPLIT_MNIST.train_batch)
        g = torch.Generator().manual_seed(PRIMARY_SEED + 17 * t_idx)
        opt = torch.optim.SGD(warm.parameters(), lr=SPLIT_MNIST.train_lr)
        step = 0
        while step < steps:
            for xb, yb in task_iter(task["train_x"], task["train_y"],
                                    SPLIT_MNIST.train_batch, g):
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(warm(xb), yb).backward(); opt.step()
                step += 1
                if step >= steps: break

    # Compute u_past from the warmed student (this is the "harmful" axis from E06/N1)
    u_past = compute_past_axis(warm, tasks[:4], batch=SPLIT_MNIST.train_batch,
                               seed=PRIMARY_SEED + 22222,
                               n_samples=PAST_GRAD_METHOD.n_past_grad_samples)
    print(f"[E08] u_past estimated ||u_past||={u_past.norm():.4f}")

    # EWC ingredients from tasks 0..3
    fisher = compute_ewc_fisher(warm, tasks[:4], batch=PAST_GRAD_METHOD.ewc_fisher_batch,
                                seed=PRIMARY_SEED + 33333)
    theta_ref = flat_params(warm).detach().clone()
    print(f"[E08] EWC fisher computed (mean={fisher.mean():.3e}, "
          f"lambda={PAST_GRAD_METHOD.ewc_lambda})")

    # Run 4 arms from the SAME fresh init (train all 5 tasks sequentially)
    arms = {}
    for name, mode, u, ewc in [
        ("A0_sgd",           "none",    None,   False),
        ("A_past_avoid",     "avoid",   u_past, False),
        ("A_past_toward",    "toward",  u_past, False),
        ("A_ewc",            "none",    None,   True),
    ]:
        print(f"\n[E08] --- {name} ---")
        model = fresh()
        arms[name] = train_arm(
            model, tasks, lr=SPLIT_MNIST.train_lr,
            epochs=SPLIT_MNIST.train_epochs_per_task,
            batch=SPLIT_MNIST.train_batch, seed=PRIMARY_SEED,
            projection_mode=mode, u_bad=u,
            probes=probes, probe_cfg=PROBE,
            ewc_fisher=(fisher if ewc else None),
            ewc_theta_ref=(theta_ref if ewc else None),
            ewc_lambda=(PAST_GRAD_METHOD.ewc_lambda if ewc else 0.0),
        )

    # ---- Summarize ----
    for name, r in arms.items():
        print(f"[E08] {name}: mean_final_acc={r['mean_final_acc']:.4f}  "
              f"BWT={r['BWT']:+.4f}  final_probeFTD={r['per_task_probe_ftd'][-1]:.4f}")

    # ---- Fairness (kill #6): matched-magnitude by construction ----
    fairness_note = ("exact_renorm=True preserves ||delta|| for A_past_avoid and "
                     "A_past_toward. EWC arm modifies gradients pre-step, then plain "
                     "SGD step (same lr as A0). Kill #6 satisfied by construction for "
                     "the projection arms; the EWC arm may differ in effective ||delta||.")

    # ---- Plots ----
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, r in arms.items():
        ax.plot(r["per_task_probe_ftd"], marker="o", label=name)
    ax.set_xlabel("task index"); ax.set_ylabel("probe FTD (mean loss)")
    ax.set_title("E08 - probe FTD across Split-MNIST tasks")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_FIGS, "e08_probe_ftd_over_tasks.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    names = list(arms.keys()); accs = [arms[n]["mean_final_acc"] for n in names]
    bwts = [arms[n]["BWT"] for n in names]
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, accs, width=0.4, label="mean final acc")
    ax.bar(xs + 0.2, bwts, width=0.4, label="BWT")
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=8)
    ax.set_title("E08 - mean final acc & BWT per arm")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_FIGS, "e08_accs_bwt.png"), dpi=140)
    plt.close(fig)

    # ---- Causal-flavored intervention verdict ----
    # If A_past_avoid < A0 SGD on probeFTD AND A_past_toward > A0 on probeFTD,
    # AND the difference is meaningful (> a small tolerance), we have a
    # sign-inverted intervention effect on probe learning speed.
    a0_ftd = arms["A0_sgd"]["per_task_probe_ftd"][-1]
    av_ftd = arms["A_past_avoid"]["per_task_probe_ftd"][-1]
    to_ftd = arms["A_past_toward"]["per_task_probe_ftd"][-1]
    tol = 1e-3  # meaningful difference threshold
    directional_causal = (av_ftd + tol < a0_ftd) and (to_ftd > a0_ftd + tol)
    same_direction = ((av_ftd < a0_ftd and to_ftd < a0_ftd)
                      or (av_ftd > a0_ftd and to_ftd > a0_ftd))

    summary = {
        "hardware": env_summary(),
        "arms": {n: {"mean_final_acc": r["mean_final_acc"], "BWT": r["BWT"],
                     "final_probe_ftd": r["per_task_probe_ftd"][-1],
                     "per_task_probe_ftd": r["per_task_probe_ftd"],
                     "per_task_acc": r["per_task_acc"],
                     "final_accs_all_tasks": r["final_accs_all_tasks"]}
                 for n, r in arms.items()},
        "u_past_source": ("mean across tasks 0..3 x n_samples of freshly-drawn "
                          "past-task minibatch gradients, normalized"),
        "fairness": {"note": fairness_note,
                     "matched_magnitude_tolerance": KILL.matched_magnitude_tolerance},
        "causal_intervention": {
            "A0_final_probeFTD": a0_ftd,
            "A_past_avoid_final_probeFTD": av_ftd,
            "A_past_toward_final_probeFTD": to_ftd,
            "sign_inverted_intervention_supports_causation": bool(directional_causal),
            "same_direction_change_suggests_noncausal": bool(same_direction),
        },
    }
    out_json = os.path.join(RESULTS_TABLES, "e08_summary.json")
    with open(out_json, "w") as f:
        json.dump(_stringify(summary), f, indent=2)
    print(f"[E08] wrote {out_json}")


if __name__ == "__main__":
    main()
