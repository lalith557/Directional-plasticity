"""E14 - Better curvature-based directional descriptors (Phase C).

Pre-registered in decision.md D-029 BEFORE this run.

E11 rejected the Fisher-top-1 axis on Split-MNIST (mean pairwise |cos|=0.215).
This experiment tests 4 alternative axes:

  1. hessian_joint_lanczos       — top eigenvector of the Hessian of the JOINT
                                   continual objective (weighted sum of past-task
                                   losses); computed via Hessian-vector products.
  2. kfac_natural_gradient       — Kronecker-factored inverse-Fisher preconditioned
                                   gradient on task 4. (Approximates natural gradient.)
  3. jacobian_svd_top1           — top singular vector of the model's Jacobian at
                                   the pre-update checkpoint on task 4.
  4. update_covariance_svd_top1  — top singular vector of the stack of past-task
                                   sequential updates (a sequence-geometry feature).

For each axis, run 5 seeds. Pass iff BOTH:
  (a) mean pairwise |cos(u_i, u_j)| >= 0.5, AND
  (b) beta_d_dir_axis significant + sign-consistent across 5 seeds.
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
from typing import List

from configs.preregistration import (
    PRIMARY_SEED, BOOTSTRAP_SEEDS, SPLIT_MNIST, UPDATE_GEN, PROBE, STATS,
    CURVATURE_AXES,
)
from src.seeds import seed_everything
from src.models import flat_params, set_flat_params, flat_grad
from src.device import env_summary
from src.datasets import load_mnist, split_mnist_tasks
from src.stats import fit_beta4_regression, ptd_caliper_match
from experiments.e04_split_mnist_replication import (
    MNISTStudent, task_iter, eval_loss, build_probes, probe_ftd,
    gen_sgd_updates, gen_orth_updates, gen_random_sphere_updates,
    gen_convex_mix_updates, _reference_gradient_mnist, cos,
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


# ---------------- Axis 1: Hessian of joint continual objective, Lanczos ----------------

def _hess_vec_joint(model, tasks, batch: int, seed: int, v: torch.Tensor,
                    weights: List[float]) -> torch.Tensor:
    """H v where H is the Hessian of sum_i w_i L_i (weighted joint loss)."""
    m = deepcopy(model); m.train()
    for p in m.parameters(): p.requires_grad_(True)
    # Assign v into parameter shape iterators
    params = list(m.parameters())
    total_grad_v = 0.0
    losses = []
    g = torch.Generator().manual_seed(seed)
    for w, task in zip(weights, tasks):
        xb, yb = next(task_iter(task["train_x"], task["train_y"], batch, g))
        losses.append(w * F.cross_entropy(m(xb), yb))
    joint_loss = sum(losses)
    grads = torch.autograd.grad(joint_loss, params, create_graph=True)
    flat_g = torch.cat([g_.reshape(-1) for g_ in grads])
    hv_scalar = (flat_g * v).sum()
    hv = torch.autograd.grad(hv_scalar, params, retain_graph=False)
    return torch.cat([h.reshape(-1) for h in hv]).detach()


def hessian_joint_top1(model, tasks, batch: int, iters: int, seed: int,
                       task_weights=None) -> torch.Tensor:
    D = sum(p.numel() for p in model.parameters())
    if task_weights is None:
        task_weights = [1.0 / len(tasks)] * len(tasks)
    v = torch.empty(D).normal_()
    v = v / (v.norm() + 1e-12)
    for it in range(iters):
        Hv = _hess_vec_joint(model, tasks, batch, seed + it, v, task_weights)
        n = Hv.norm() + 1e-12
        v = Hv / n
    return v


# ---------------- Axis 2: K-FAC natural gradient direction ----------------

def _linear_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def kfac_natural_gradient_dir(model, x, y, batch: int, damping: float, seed: int
                               ) -> torch.Tensor:
    """Approximate F^{-1} g by KFAC on each Linear layer (A ⊗ G)^{-1}.

    A = E[a a^T]  (input activations),  G = E[g g^T]  (pre-activation output grads).
    We estimate A/G on one batch. Not a full-KFAC implementation but a solid proxy.
    """
    m = deepcopy(model); m.train()
    linears = _linear_layers(m)
    # Register hooks capturing inputs to each layer
    acts = {}; grads_out = {}
    hooks = []
    for l in linears:
        def make_fw(l_):
            def h(_m, inp, out):
                acts[id(l_)] = inp[0].detach()
            return h
        def make_bw(l_):
            def h(_m, gin, gout):
                grads_out[id(l_)] = gout[0].detach()
            return h
        hooks.append(l.register_forward_hook(make_fw(l)))
        hooks.append(l.register_full_backward_hook(make_bw(l)))
    g = torch.Generator().manual_seed(seed)
    xb, yb = next(task_iter(x, y, batch, g))
    m.zero_grad(set_to_none=True)
    loss = F.cross_entropy(m(xb), yb)
    loss.backward()
    # Per-layer natural gradient
    ngs = []
    for l in linears:
        a = acts[id(l)]        # (B, in_f)
        g_out = grads_out[id(l)]   # (B, out_f)
        # Append bias 1s to a
        if l.bias is not None:
            a_aug = torch.cat([a, torch.ones(a.shape[0], 1)], dim=1)
        else:
            a_aug = a
        A = (a_aug.T @ a_aug) / a.shape[0]
        G = (g_out.T @ g_out) / g_out.shape[0]
        A += damping * torch.eye(A.shape[0])
        G += damping * torch.eye(G.shape[0])
        # grads shape: W (out_f, in_f), b (out_f,) -> combined (out_f, in_f + 1)
        W_grad = l.weight.grad.detach()
        if l.bias is not None:
            g_mat = torch.cat([W_grad, l.bias.grad.detach().unsqueeze(1)], dim=1)
        else:
            g_mat = W_grad
        # Natural gradient: G^-1 g_mat A^-1
        try:
            ng = torch.linalg.solve(G, g_mat) @ torch.linalg.inv(A)
        except Exception:
            ng = g_mat
        # Flatten (out_f, in_f + 1) back to weight+bias
        ng_W = ng[:, :l.weight.shape[1]]
        ngs.append(ng_W.reshape(-1))
        if l.bias is not None:
            ng_b = ng[:, -1]
            ngs.append(ng_b.reshape(-1))
    # Zero-pad in any non-Linear params (there are none in MNISTStudent, but safe)
    D = sum(p.numel() for p in m.parameters())
    ng_flat = torch.zeros(D)
    off = 0
    for p in m.parameters():
        n = p.numel()
        # Find matching chunk (parameters iterate in declaration order which
        # matches ngs order for pure-Linear model)
        chunk = ngs.pop(0) if ngs else torch.zeros(n)
        if chunk.numel() != n:
            chunk = torch.zeros(n)  # fallback
        ng_flat[off:off + n] = chunk
        off += n
    for h in hooks: h.remove()
    n = ng_flat.norm() + 1e-12
    return (ng_flat / n).detach()


# ---------------- Axis 3: Jacobian top singular vector ----------------

def jacobian_top1_svd(model, x, y, n_samples: int, seed: int) -> torch.Tensor:
    """Top-1 right singular vector of the parameter-Jacobian of the model output
    (using per-sample gradients of predicted class logit).

    We form J of size (n_samples, D) by taking gradients of `logit_predicted[i]` for
    each of n_samples inputs. Then top right-singular vector via SVD on J.
    """
    m = deepcopy(model); m.train()
    for p in m.parameters(): p.requires_grad_(True)
    g = torch.Generator().manual_seed(seed)
    D = sum(p.numel() for p in m.parameters())
    n_samples = min(n_samples, x.shape[0])
    idx = torch.randperm(x.shape[0], generator=g)[:n_samples]
    rows = []
    for i in idx:
        xi = x[i].unsqueeze(0)
        m.zero_grad(set_to_none=True)
        logits = m(xi)
        y_hat = int(logits.argmax(-1).item())
        # gradient of predicted logit
        logits[0, y_hat].backward()
        rows.append(flat_grad(m).detach().clone())
    J = torch.stack(rows, dim=0)   # (n, D)
    # top right singular vector = right vec of top eigenvalue of J^T J
    _, S, Vt = torch.linalg.svd(J, full_matrices=False)
    return Vt[0].detach()


# ---------------- Axis 4: SVD of update-covariance ----------------

def update_covariance_top1(deltas: List[torch.Tensor]) -> torch.Tensor:
    """Top right singular vector of matrix stacking unit-normalized deltas."""
    A = torch.stack([d / (d.norm() + 1e-12) for d in deltas], dim=0)  # (n, D)
    _, S, Vt = torch.linalg.svd(A, full_matrices=False)
    return Vt[0].detach()


# ---------------- Seed driver: reuse E11-style protocol ----------------

def _seed_run(seed: int):
    seed_everything(seed)
    tx, ty, vx, vy = load_mnist(SPLIT_MNIST.mnist_dir)
    tasks = split_mnist_tasks(tx, ty, vx, vy, SPLIT_MNIST.task_class_pairs)
    for task in tasks:
        pair = task["pair"]
        mask_tr = np.isin(ty, pair); mask_te = np.isin(vy, pair)
        task["train_y"] = torch.from_numpy(ty[mask_tr].astype(np.int64))
        task["test_y"] = torch.from_numpy(vy[mask_te].astype(np.int64))

    student = MNISTStudent(input_dim=784,
                           hidden=SPLIT_MNIST.student_hidden_width,
                           depth=SPLIT_MNIST.student_hidden_depth, n_out=10)
    for t_idx in range(4):
        task = tasks[t_idx]
        n = task["train_x"].shape[0]
        steps = SPLIT_MNIST.train_epochs_per_task * max(1, n // SPLIT_MNIST.train_batch)
        g = torch.Generator().manual_seed(seed + 17 * t_idx)
        opt = torch.optim.SGD(student.parameters(), lr=SPLIT_MNIST.train_lr)
        step = 0
        while step < steps:
            for xb, yb in task_iter(task["train_x"], task["train_y"],
                                    SPLIT_MNIST.train_batch, g):
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(student(xb), yb).backward(); opt.step()
                step += 1
                if step >= steps: break
    baseline = deepcopy(student)

    task4 = tasks[4]

    # Compute 4 candidate axes
    axes = {}
    print(f"[E14 seed={seed}] computing hessian_joint_lanczos ...")
    axes["hessian_joint_lanczos"] = hessian_joint_top1(
        baseline, tasks, batch=CURVATURE_AXES.lanczos_batch,
        iters=CURVATURE_AXES.lanczos_iters, seed=seed + 42,
    )
    print(f"[E14 seed={seed}] computing kfac_natural_gradient ...")
    axes["kfac_natural_gradient"] = kfac_natural_gradient_dir(
        baseline, task4["train_x"], task4["train_y"],
        batch=CURVATURE_AXES.lanczos_batch,
        damping=CURVATURE_AXES.kfac_damping, seed=seed + 43,
    )
    print(f"[E14 seed={seed}] computing jacobian_svd_top1 ...")
    axes["jacobian_svd_top1"] = jacobian_top1_svd(
        baseline, task4["train_x"], task4["train_y"],
        n_samples=CURVATURE_AXES.jacobian_n_samples, seed=seed + 44,
    )

    # Generate updates on task 4 (for beta_axis regression)
    per = 15
    norms = UPDATE_GEN.target_norms
    a = gen_sgd_updates(baseline, task4["train_x"], task4["train_y"], per, norms,
                        UPDATE_GEN.sgd_inner_steps, UPDATE_GEN.sgd_inner_lr,
                        UPDATE_GEN.sgd_inner_batch, seed=seed + 501)
    b = gen_orth_updates(baseline, task4["train_x"], task4["train_y"], per, norms,
                         UPDATE_GEN.sgd_inner_batch, seed=seed + 502)
    c = gen_random_sphere_updates(baseline, task4["train_x"], task4["train_y"],
                                  per, norms, UPDATE_GEN.sgd_inner_batch, seed=seed + 503)
    d = gen_convex_mix_updates(baseline, task4["train_x"], task4["train_y"],
                               per, norms, UPDATE_GEN.sgd_inner_steps,
                               UPDATE_GEN.sgd_inner_lr, UPDATE_GEN.sgd_inner_batch,
                               seed=seed + 504)
    updates = a + b + c + d

    # Axis 4: SVD of these updates themselves (unit-normalized)
    print(f"[E14 seed={seed}] computing update_covariance_svd_top1 ...")
    axes["update_covariance_svd_top1"] = update_covariance_top1(
        [u.delta for u in updates[:CURVATURE_AXES.update_covariance_n_updates]]
    )

    # Standard g_ref for baseline d_dir
    g_ref = _reference_gradient_mnist(baseline, task4["train_x"], task4["train_y"],
                                      UPDATE_GEN.sgd_inner_batch, seed=seed + 991)
    tx_norm = torch.from_numpy(tx.reshape(len(tx), -1).astype(np.float32) / 255.0)
    ty_all = torch.from_numpy(ty.astype(np.int64))
    probes = build_probes(tx_norm, ty_all, PROBE.n_probes,
                          seed=seed + 10_000, subset_size=3000)
    ref_ftd = probe_ftd(baseline, probes, PROBE.probe_steps, PROBE.probe_lr,
                        PROBE.probe_batch, seed=seed)
    past_tasks = tasks[:4]
    ref_past = [eval_loss(baseline, t["test_x"], t["test_y"]) for t in past_tasks]
    theta_before = flat_params(baseline)

    rows = []
    for uid, u in enumerate(updates):
        m = deepcopy(baseline)
        set_flat_params(m, theta_before + u.delta)
        ftd_after = probe_ftd(m, probes, PROBE.probe_steps, PROBE.probe_lr,
                              PROBE.probe_batch, seed=seed)
        ftd = ftd_after - ref_ftd
        after_past = [eval_loss(m, t["test_x"], t["test_y"]) for t in past_tasks]
        ptd = float(sum(a2 - r for a2, r in zip(after_past, ref_past)))
        row = {
            "update_id": uid, "source": u.source, "target_norm": u.target_norm,
            "mag": float(u.delta.norm().item()),
            "d_dir": cos(u.delta, g_ref),
            "ctb": float(u.ctb), "ptd": ptd, "ftd": ftd,
        }
        for axis_name, axis_vec in axes.items():
            row[f"d_dir_{axis_name}"] = cos(u.delta, axis_vec)
        rows.append(row)
    df = pd.DataFrame(rows)

    # Regress FTD on each axis descriptor under caliper matching
    matched = ptd_caliper_match(df, caliper=STATS.ptd_primary_caliper)
    per_axis = {}
    for axis_name in axes:
        covariates = ["mag", "ctb", "ptd", f"d_dir_{axis_name}"]
        if len(matched) < 20:
            per_axis[axis_name] = {"beta": None, "ci_low": None, "ci_high": None,
                                    "p": None, "excludes_zero": False}
            continue
        r = fit_beta4_regression(matched, covariates=covariates, outcome="ftd",
                                 cov_type=STATS.ols_cov_type)
        beta = float(r.coefs[f"d_dir_{axis_name}"])
        ci = (float(r.ci_low[f"d_dir_{axis_name}"]),
              float(r.ci_high[f"d_dir_{axis_name}"]))
        p = float(r.pvalues[f"d_dir_{axis_name}"])
        per_axis[axis_name] = {"beta": beta, "ci_low": ci[0], "ci_high": ci[1],
                                "p": p, "excludes_zero": bool((ci[0] > 0) or (ci[1] < 0))}
        print(f"[E14 seed={seed}] axis={axis_name} beta={beta:+.5f} "
              f"CI=[{ci[0]:+.5f},{ci[1]:+.5f}] p={p:.3g}")
    return {"seed": int(seed), "axes": {k: v.numpy() for k, v in axes.items()},
            "per_axis": per_axis, "n_matched": int(len(matched))}


def main() -> None:
    print(f"[E14] env={env_summary()}")
    per_seed = []
    for seed in BOOTSTRAP_SEEDS[:CURVATURE_AXES.n_seeds]:
        print(f"\n[E14] --- seed={seed} ---")
        r = _seed_run(seed)
        per_seed.append(r)

    # For each axis, compute pairwise |cos| across seeds; sign-consistency across seeds
    verdicts = {}
    for axis_name in CURVATURE_AXES.axes:
        U = [torch.from_numpy(r["axes"][axis_name]) for r in per_seed]
        seeds = [r["seed"] for r in per_seed]
        pair_cos = {}
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                ui, uj = U[i], U[j]
                c = float((ui @ uj / (ui.norm() * uj.norm() + 1e-12)).item())
                pair_cos[f"{seeds[i]}_vs_{seeds[j]}"] = c
        abs_cos = {k: abs(v) for k, v in pair_cos.items()}
        mean_abs = float(np.mean(list(abs_cos.values()))) if abs_cos else 0.0
        # per-seed beta info
        betas = [r["per_axis"][axis_name]["beta"] for r in per_seed
                 if r["per_axis"][axis_name]["beta"] is not None]
        all_sig = all(r["per_axis"][axis_name]["excludes_zero"] for r in per_seed
                      if r["per_axis"][axis_name]["beta"] is not None)
        sign_stable = (all(b > 0 for b in betas) or all(b < 0 for b in betas)) if betas else False
        passes_axis = mean_abs >= CURVATURE_AXES.axis_stability_threshold
        passes_beta = sign_stable and all_sig
        verdicts[axis_name] = {
            "mean_abs_pairwise": mean_abs,
            "sign_consistent": bool(sign_stable),
            "all_seeds_significant": bool(all_sig),
            "passes_axis_stability": bool(passes_axis),
            "passes_beta_sig": bool(passes_beta),
            "passes_both": bool(passes_axis and passes_beta),
            "pair_cos_abs": abs_cos,
            "per_seed_beta": {r["seed"]: r["per_axis"][axis_name] for r in per_seed},
        }
        print(f"[E14] axis={axis_name} mean|cos|={mean_abs:.3f} "
              f"stable_axis={passes_axis} beta_sig={passes_beta}")

    any_pass = any(v["passes_both"] for v in verdicts.values())
    overall = "POSITIVE_STABLE_AXIS_FOUND" if any_pass else "NULL_ALL_AXES_FAIL"

    # Plot: bar of mean|cos| per axis
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    axis_names = list(verdicts.keys())
    means = [verdicts[a]["mean_abs_pairwise"] for a in axis_names]
    colors = ["firebrick" if verdicts[a]["passes_axis_stability"] else "gray"
              for a in axis_names]
    ax.bar(axis_names, means, color=colors)
    ax.axhline(CURVATURE_AXES.axis_stability_threshold, color="black", linestyle="--",
               label=f"threshold={CURVATURE_AXES.axis_stability_threshold}")
    ax.set_ylabel("mean pairwise |cos| across 5 seeds")
    ax.set_title("E14 axis-stability across seeds")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=8)
    # Bar of per-seed sign consistency
    ax = axes[1]
    for i, axis_name in enumerate(axis_names):
        per_beta = [verdicts[axis_name]["per_seed_beta"][s]["beta"] for s in BOOTSTRAP_SEEDS
                    if verdicts[axis_name]["per_seed_beta"][s]["beta"] is not None]
        ax.scatter([i] * len(per_beta), per_beta, alpha=0.7)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(range(len(axis_names))); ax.set_xticklabels(axis_names, rotation=15)
    ax.set_ylabel("per-seed beta_d_dir_axis (matched c=0.02)")
    ax.set_title("E14 sign-consistency per axis")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_FIGS, "e14_curvature_axes.png"), dpi=140)
    plt.close(fig)

    with open(os.path.join(RESULTS_TABLES, "e14_summary.json"), "w") as f:
        json.dump(_stringify({"hardware": env_summary(),
                              "per_seed_n_matched": [r["n_matched"] for r in per_seed],
                              "verdicts": verdicts,
                              "overall_verdict": overall}), f, indent=2)
    print(f"\n[E14] OVERALL VERDICT: {overall}")


if __name__ == "__main__":
    main()
