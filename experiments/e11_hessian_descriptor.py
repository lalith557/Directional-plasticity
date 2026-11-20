"""E11 - Hessian/Fisher-eigenvector directional descriptor (Phase 5, D-022).

Motivation: E06/N2 showed that g_ref, current-task grad, past-task grad, and
empirical PCA of FTD-weighted deltas all give SEED-UNSTABLE harmful axes on
Split-MNIST. The scalar beta_d_dir is reproducible but the axis it points to
is not - which is why projection methods (E05, E06/N3) don't work.

This experiment asks: is there a THEORETICALLY-MOTIVATED axis (top eigenvector
of the empirical Fisher information matrix on task 4) that IS seed-stable? If
yes, that's a genuine controllable geometric object; if no, the "no stable axis"
finding is stronger.

Method: Fisher-vector products via autograd (no explicit N-param x N-param
matrix). Lanczos-style power iteration for top eigenvectors.
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
import torch.nn.functional as F
from copy import deepcopy

from configs.preregistration import (
    PRIMARY_SEED, BOOTSTRAP_SEEDS, SPLIT_MNIST, UPDATE_GEN, PROBE,
    STATS, HessianDescriptorConfig,
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

HESS_CFG = HessianDescriptorConfig()

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


def _fisher_matvec(model, x_batches, v: torch.Tensor) -> torch.Tensor:
    """Empirical Fisher-vector product: E[g g^T] v, using autograd on the
    log-likelihood under model's predicted class (empirical Fisher approximation).

    F v is estimated as (1/B) sum_i g_i (g_i . v).
    """
    result = torch.zeros_like(v)
    total = 0
    for xb, yb in x_batches:
        m = deepcopy(model)
        m.zero_grad(set_to_none=True)
        logits = m(xb)
        # Sample "y_hat" (predicted class) for empirical Fisher
        with torch.no_grad():
            y_hat = logits.argmax(dim=-1)
        # per-example gradient collection is expensive; use total-loss then
        # (F v ~ g (g^T v)) for the batch-mean gradient — Gauss-Newton-style
        loss = F.cross_entropy(logits, y_hat, reduction="mean")
        loss.backward()
        g = flat_grad(m)
        result += g * float((g @ v).item())
        total += 1
    return result / max(total, 1)


def top_k_fisher_eigenvectors(model, x_batches_pool, k: int, iters: int, seed: int):
    """Deflation-based power iteration for top-k Fisher eigenvectors."""
    D = sum(p.numel() for p in model.parameters())
    rng = torch.Generator().manual_seed(seed)
    vecs = []
    for j in range(k):
        v = torch.empty(D).normal_(generator=rng)
        v = v / (v.norm() + 1e-12)
        for it in range(iters):
            Fv = _fisher_matvec(model, x_batches_pool, v)
            # deflate against previously-found eigenvectors
            for u in vecs:
                Fv = Fv - (Fv @ u) * u
            n = Fv.norm() + 1e-12
            v = Fv / n
        vecs.append(v)
    return torch.stack(vecs, dim=1)  # (D, k)


def sample_batches(x, y, batch, n_batches, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_batches):
        for xb, yb in task_iter(x, y, batch, g):
            out.append((xb, yb))
            break
    return out


def _seed_run(seed: int) -> dict:
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

    # Fisher on task 4
    task4 = tasks[4]
    fisher_batches = sample_batches(task4["train_x"], task4["train_y"],
                                    HESS_CFG.fisher_batch, HESS_CFG.n_fisher_batches,
                                    seed=seed + 909)
    V = top_k_fisher_eigenvectors(baseline, fisher_batches,
                                  k=HESS_CFG.top_k, iters=HESS_CFG.lanczos_iters,
                                  seed=seed + 42)
    u_fisher_top1 = V[:, 0] / (V[:, 0].norm() + 1e-12)

    # Generate updates on task 4 and measure d_dir_fisher cos + FTD
    x4, y4 = task4["train_x"], task4["train_y"]
    per = 15
    norms = UPDATE_GEN.target_norms
    a = gen_sgd_updates(baseline, x4, y4, per, norms, UPDATE_GEN.sgd_inner_steps,
                        UPDATE_GEN.sgd_inner_lr, UPDATE_GEN.sgd_inner_batch,
                        seed=seed + 501)
    b = gen_orth_updates(baseline, x4, y4, per, norms,
                         UPDATE_GEN.sgd_inner_batch, seed=seed + 502)
    c = gen_random_sphere_updates(baseline, x4, y4, per, norms,
                                  UPDATE_GEN.sgd_inner_batch, seed=seed + 503)
    d = gen_convex_mix_updates(baseline, x4, y4, per, norms,
                               UPDATE_GEN.sgd_inner_steps, UPDATE_GEN.sgd_inner_lr,
                               UPDATE_GEN.sgd_inner_batch, seed=seed + 504)
    updates = a + b + c + d

    g_ref = _reference_gradient_mnist(baseline, x4, y4,
                                      UPDATE_GEN.sgd_inner_batch, seed=seed + 991)
    tx_norm = torch.from_numpy(tx.reshape(len(tx), -1).astype(np.float32) / 255.0)
    ty_all = torch.from_numpy(ty.astype(np.int64))
    probes = build_probes(tx_norm, ty_all, PROBE.n_probes,
                          seed=seed + 10_000, subset_size=3000)

    ref_ftd = probe_ftd(baseline, probes, PROBE.probe_steps, PROBE.probe_lr,
                        PROBE.probe_batch, seed=seed)
    ref_past = [eval_loss(baseline, t["test_x"], t["test_y"]) for t in tasks[:4]]
    theta_before = flat_params(baseline)
    rows = []
    for uid, u in enumerate(updates):
        m = deepcopy(baseline)
        set_flat_params(m, theta_before + u.delta)
        ftd_after = probe_ftd(m, probes, PROBE.probe_steps, PROBE.probe_lr,
                              PROBE.probe_batch, seed=seed)
        ftd = ftd_after - ref_ftd
        after_past = [eval_loss(m, t["test_x"], t["test_y"]) for t in tasks[:4]]
        ptd = float(sum(a2 - r for a2, r in zip(after_past, ref_past)))
        rows.append({
            "update_id": uid, "source": u.source, "target_norm": u.target_norm,
            "mag": float(u.delta.norm().item()),
            "d_dir": cos(u.delta, g_ref),
            "d_dir_fisher_top1": cos(u.delta, u_fisher_top1),
            "ctb": float(u.ctb), "ptd": ptd, "ftd": ftd,
        })
    df = pd.DataFrame(rows)

    covariates = ["mag", "ctb", "ptd", "d_dir_fisher_top1"]
    matched = ptd_caliper_match(df, caliper=STATS.ptd_primary_caliper)
    if len(matched) >= 20:
        r = fit_beta4_regression(matched, covariates=covariates, outcome="ftd",
                                 cov_type=STATS.ols_cov_type)
        beta_fisher = float(r.coefs["d_dir_fisher_top1"])
        ci_lo = float(r.ci_low["d_dir_fisher_top1"])
        ci_hi = float(r.ci_high["d_dir_fisher_top1"])
        p_val = float(r.pvalues["d_dir_fisher_top1"])
        excludes = (ci_lo > 0) or (ci_hi < 0)
    else:
        beta_fisher = ci_lo = ci_hi = p_val = None
        excludes = False

    return {
        "seed": int(seed),
        "u_fisher_top1": u_fisher_top1.numpy(),   # save for pairwise cos
        "u_fisher_vs_gref_cos": float((u_fisher_top1 @ g_ref
                                       / (u_fisher_top1.norm() * g_ref.norm() + 1e-12)).item()),
        "beta_d_dir_fisher_top1": beta_fisher,
        "ci_low": ci_lo, "ci_high": ci_hi, "p": p_val,
        "excludes_zero": bool(excludes),
        "n_matched": int(len(matched)),
    }


def main() -> None:
    print(f"[E11] env={env_summary()}")
    per_seed = []
    for seed in BOOTSTRAP_SEEDS:
        print(f"\n[E11] --- seed={seed} ---")
        r = _seed_run(seed)
        print(f"[E11] seed {seed}: beta_d_dir_fisher_top1="
              f"{r['beta_d_dir_fisher_top1']}"
              + (f" CI=[{r['ci_low']:.5f},{r['ci_high']:.5f}] p={r['p']:.4g}"
                 if r["beta_d_dir_fisher_top1"] is not None else "")
              + f"  u_fisher_vs_gref={r['u_fisher_vs_gref_cos']:+.4f}")
        per_seed.append(r)

    # Pairwise cosine between per-seed Fisher-top1 vectors
    U = [torch.from_numpy(r["u_fisher_top1"]) for r in per_seed]
    seeds = [r["seed"] for r in per_seed]
    pair_cos = {}
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            ui, uj = U[i], U[j]
            c = float((ui @ uj / (ui.norm() * uj.norm() + 1e-12)).item())
            pair_cos[f"{seeds[i]}_vs_{seeds[j]}"] = c
            print(f"[E11] cos(u_fisher_{seeds[i]}, u_fisher_{seeds[j]}) = {c:+.4f}")

    # Also allow SIGN INVARIANCE: eigenvectors are only unique up to sign
    abs_pair = {k: abs(v) for k, v in pair_cos.items()}
    mean_abs_pairwise = float(np.mean(list(abs_pair.values())))
    median_abs_pairwise = float(np.median(list(abs_pair.values())))

    betas = [r["beta_d_dir_fisher_top1"] for r in per_seed
             if r["beta_d_dir_fisher_top1"] is not None]
    sign_consistent = (all(b > 0 for b in betas) or all(b < 0 for b in betas)) if betas else False
    all_sig = all(r["excludes_zero"] for r in per_seed
                  if r["beta_d_dir_fisher_top1"] is not None)

    stable_axis = mean_abs_pairwise > 0.5  # pre-registered threshold (D-022)

    verdict = ("STABLE_AXIS_AND_SIGNIFICANT" if (stable_axis and sign_consistent and all_sig)
               else "STABLE_AXIS_NULL_BETA" if stable_axis
               else "SIGNIFICANT_UNSTABLE_AXIS" if (sign_consistent and all_sig)
               else "NULL_ON_BOTH")

    # Save
    summary = {
        "hardware": env_summary(),
        "per_seed": [{"seed": r["seed"],
                      "beta_d_dir_fisher_top1": r["beta_d_dir_fisher_top1"],
                      "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p": r["p"],
                      "excludes_zero": r["excludes_zero"],
                      "u_fisher_vs_gref_cos": r["u_fisher_vs_gref_cos"],
                      "n_matched": r["n_matched"]}
                     for r in per_seed],
        "pairwise_u_fisher_cos": pair_cos,
        "pairwise_abs_cos": abs_pair,
        "mean_abs_pairwise": mean_abs_pairwise,
        "median_abs_pairwise": median_abs_pairwise,
        "stable_axis_pre_registered_threshold": 0.5,
        "stable_axis_achieved": bool(stable_axis),
        "beta_sign_consistent": bool(sign_consistent),
        "all_seeds_significant": bool(all_sig),
        "verdict": verdict,
    }
    with open(os.path.join(RESULTS_TABLES, "e11_summary.json"), "w") as f:
        json.dump(_stringify(summary), f, indent=2)

    # Plot: per-seed beta forest + pairwise heatmap of |cos|
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    ax = axes[0]
    ys = list(range(len(per_seed)))[::-1]
    for y, r in zip(ys, per_seed):
        if r["beta_d_dir_fisher_top1"] is None: continue
        ax.errorbar(r["beta_d_dir_fisher_top1"], y,
                    xerr=[[r["beta_d_dir_fisher_top1"] - r["ci_low"]],
                          [r["ci_high"] - r["beta_d_dir_fisher_top1"]]],
                    fmt=("s" if r["excludes_zero"] else "o"),
                    color=("firebrick" if r["excludes_zero"] else "gray"),
                    capsize=3, markersize=6)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ys); ax.set_yticklabels([f"seed={r['seed']}" for r in per_seed])
    ax.set_xlabel("beta_d_dir_fisher_top1 (95% CI, cal=0.02)")
    ax.set_title("E11: per-seed Fisher-top-1 coefficient")
    ax.grid(True, axis="x", alpha=0.3)

    ax = axes[1]
    M = np.abs(np.array([[float((U[i] @ U[j] /
                                (U[i].norm() * U[j].norm() + 1e-12)).item())
                          for j in range(len(U))]
                         for i in range(len(U))]))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(seeds))); ax.set_yticks(range(len(seeds)))
    ax.set_xticklabels([str(s) for s in seeds]); ax.set_yticklabels([str(s) for s in seeds])
    ax.set_title(f"E11: |cos| of Fisher-top1 across seeds "
                 f"(mean={mean_abs_pairwise:.3f})")
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_FIGS, "e11_fisher_axis_stability.png"), dpi=140)
    plt.close(fig)

    print(f"\n[E11] mean |cos| across seeds = {mean_abs_pairwise:.4f}  "
          f"(threshold 0.5 for stable axis)")
    print(f"[E11] VERDICT: {verdict}")


if __name__ == "__main__":
    main()
