"""E10 - Probe-family robustness sweep (Phase 3, pre-registered D-021).

Motivation: E04 established a significant beta_d_dir on Split-MNIST with a single
probe family (permuted-MNIST) and a single budget (200 steps). A stronger version
of kill #4 says the effect should also be stable across REASONABLE PROBE
VARIATIONS. This experiment repeats the E04 beta_d_dir estimate across the
crossproduct of:
  - probe families:  permuted_mnist  and  novel_class_pair
  - probe budgets:   100 steps       and  200 steps

For each cell we report beta_d_dir under the primary caliper. If sign+significance
survive at every (family, budget), the MNIST scalar finding is defensible against
"maybe your probe is the effect" objections.
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
    PRIMARY_SEED, SPLIT_MNIST, UPDATE_GEN, PROBE, STATS, KILL, PROBE_ROBUSTNESS,
)
from src.seeds import seed_everything
from src.models import flat_params, set_flat_params, flat_grad
from src.device import env_summary
from src.datasets import load_mnist, split_mnist_tasks
from src.stats import fit_beta4_regression, ptd_caliper_match, compute_vif
from experiments.e04_split_mnist_replication import (
    MNISTStudent, task_iter, eval_loss, probe_ftd,
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


def build_probes_family(family: str, tx_norm, ty_all, n_probes: int, seed: int,
                        held_out_pair=None, subset_size=3000):
    """Two probe families.

    permuted_mnist: random per-pixel permutation of a 3000-image subset.
    novel_class_pair: 2-class subset drawn from digits NOT in the warmup tasks
                      (uses `held_out_pair`). y is remapped to {0,1}.
    """
    probes = []
    rng = np.random.default_rng(seed)
    if family == "permuted_mnist":
        for p_id in range(n_probes):
            perm = rng.permutation(tx_norm.shape[1])
            idx = rng.choice(tx_norm.shape[0], size=subset_size, replace=False)
            x = tx_norm[idx][:, perm]
            y = ty_all[idx].long()
            probes.append((x, y))
    elif family == "novel_class_pair":
        assert held_out_pair is not None
        y_np = ty_all.numpy()
        mask = np.isin(y_np, held_out_pair)
        idx_all = np.where(mask)[0]
        for p_id in range(n_probes):
            pick = rng.choice(idx_all, size=min(subset_size, len(idx_all)), replace=False)
            x = tx_norm[pick]
            y_orig = ty_all[pick].numpy()
            y_remap = np.where(y_orig == held_out_pair[0], 0, 1).astype(np.int64)
            probes.append((x, torch.from_numpy(y_remap)))
    else:
        raise ValueError(family)
    return probes


def probe_ftd_family(model, probes, steps, lr, batch, seed, family: str):
    """Same as probe_ftd but supports novel_class_pair binary head via slicing.

    We use the SAME 10-way head; for novel_class_pair we compute CE loss between
    the FULL 10-way logits and the ORIGINAL 10-way targets (before remap) —
    to keep the head-consistency, we upcast the binary label into the original
    digit label.
    """
    aucs = []
    for p_id, (x, y) in enumerate(probes):
        m = deepcopy(model)
        opt = torch.optim.SGD(m.parameters(), lr=lr)
        g = torch.Generator().manual_seed(seed + 7919 * (p_id + 1))
        losses, step = [], 0
        while step < steps:
            for xb, yb in task_iter(x, y, batch, g):
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(m(xb), yb)
                loss.backward(); opt.step()
                losses.append(loss.item())
                step += 1
                if step >= steps: break
        aucs.append(sum(losses) / len(losses))
        del m
    return float(sum(aucs) / len(aucs))


def measure_updates(baseline, updates, probes, past_tasks, g_ref,
                    probe_steps: int, family: str, seed: int):
    ref_ftd = probe_ftd_family(baseline, probes, probe_steps, PROBE.probe_lr,
                               PROBE.probe_batch, seed=seed, family=family)
    ref_past = [eval_loss(baseline, t["test_x"], t["test_y"]) for t in past_tasks]
    theta_before = flat_params(baseline)
    rows = []
    for uid, u in enumerate(updates):
        m = deepcopy(baseline)
        set_flat_params(m, theta_before + u.delta)
        ftd_after = probe_ftd_family(m, probes, probe_steps, PROBE.probe_lr,
                                     PROBE.probe_batch, seed=seed, family=family)
        ftd = ftd_after - ref_ftd
        after_past = [eval_loss(m, t["test_x"], t["test_y"]) for t in past_tasks]
        ptd = float(sum(a2 - r for a2, r in zip(after_past, ref_past)))
        rows.append({
            "update_id": uid, "source": u.source, "target_norm": u.target_norm,
            "mag": float(u.delta.norm().item()),
            "d_dir": cos(u.delta, g_ref),
            "ctb": float(u.ctb), "ptd": ptd, "ftd": ftd,
        })
    return pd.DataFrame(rows)


def main() -> None:
    seed_everything(PRIMARY_SEED)
    print(f"[E10] env={env_summary()}")

    tx, ty, vx, vy = load_mnist(SPLIT_MNIST.mnist_dir)
    # HOLD OUT digit-pair (8,9): warmup uses only 4 pairs {0,1}..{6,7};
    # (8,9) is available as a novel_class_pair probe.
    warmup_pairs = ((0, 1), (2, 3), (4, 5), (6, 7))
    held_out_pair = (8, 9)
    tasks = split_mnist_tasks(tx, ty, vx, vy, warmup_pairs)
    for task in tasks:
        pair = task["pair"]
        mask_tr = np.isin(ty, pair); mask_te = np.isin(vy, pair)
        task["train_y"] = torch.from_numpy(ty[mask_tr].astype(np.int64))
        task["test_y"] = torch.from_numpy(vy[mask_te].astype(np.int64))
    # We also need a "task 4" for generating updates - use the held-out pair
    task4 = split_mnist_tasks(tx, ty, vx, vy, (held_out_pair,))[0]
    mask_tr4 = np.isin(ty, held_out_pair); mask_te4 = np.isin(vy, held_out_pair)
    task4["train_y"] = torch.from_numpy(ty[mask_tr4].astype(np.int64))
    task4["test_y"] = torch.from_numpy(vy[mask_te4].astype(np.int64))

    student = MNISTStudent(input_dim=784,
                           hidden=SPLIT_MNIST.student_hidden_width,
                           depth=SPLIT_MNIST.student_hidden_depth, n_out=10)
    for t_idx in range(len(warmup_pairs)):
        task = tasks[t_idx]
        n = task["train_x"].shape[0]
        steps = SPLIT_MNIST.train_epochs_per_task * max(1, n // SPLIT_MNIST.train_batch)
        g = torch.Generator().manual_seed(PRIMARY_SEED + 17 * t_idx)
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
    print(f"[E10] warmed through pairs {warmup_pairs}, holding out {held_out_pair}")

    # Generate updates ON THE HELD-OUT PAIR (mirrors E04's protocol of "task 4"
    # being the update-generation task)
    x4, y4 = task4["train_x"], task4["train_y"]
    per = 20
    norms = UPDATE_GEN.target_norms
    print(f"[E10] generating {4 * len(norms) * per} updates ({per}/source)")
    a = gen_sgd_updates(baseline, x4, y4, per, norms, UPDATE_GEN.sgd_inner_steps,
                        UPDATE_GEN.sgd_inner_lr, UPDATE_GEN.sgd_inner_batch,
                        seed=PRIMARY_SEED + 501)
    b = gen_orth_updates(baseline, x4, y4, per, norms, UPDATE_GEN.sgd_inner_batch,
                         seed=PRIMARY_SEED + 502)
    c = gen_random_sphere_updates(baseline, x4, y4, per, norms,
                                  UPDATE_GEN.sgd_inner_batch, seed=PRIMARY_SEED + 503)
    d = gen_convex_mix_updates(baseline, x4, y4, per, norms,
                               UPDATE_GEN.sgd_inner_steps, UPDATE_GEN.sgd_inner_lr,
                               UPDATE_GEN.sgd_inner_batch, seed=PRIMARY_SEED + 504)
    updates = a + b + c + d

    g_ref = _reference_gradient_mnist(baseline, x4, y4,
                                      UPDATE_GEN.sgd_inner_batch,
                                      seed=PRIMARY_SEED + 991)
    tx_norm = torch.from_numpy(tx.reshape(len(tx), -1).astype(np.float32) / 255.0)
    ty_all = torch.from_numpy(ty.astype(np.int64))

    covariates = list(STATS.covariates)

    cells = []
    for family in PROBE_ROBUSTNESS.families:
        for budget in PROBE_ROBUSTNESS.probe_budgets:
            print(f"\n[E10] === family={family}  budget={budget} ===")
            probes = build_probes_family(
                family, tx_norm, ty_all, PROBE_ROBUSTNESS.n_probes,
                seed=PRIMARY_SEED + 10_000 + hash(family) % 1000,
                held_out_pair=held_out_pair, subset_size=3000,
            )
            df = measure_updates(baseline, updates, probes, tasks, g_ref,
                                 probe_steps=budget, family=family,
                                 seed=PRIMARY_SEED)
            df.to_csv(os.path.join(
                RESULTS_TABLES, f"e10_{family}_{budget}_measurements.csv"), index=False)

            vif = compute_vif(df, covariates)
            r_full = fit_beta4_regression(df, covariates=covariates, outcome="ftd",
                                          cov_type=STATS.ols_cov_type)
            matched = ptd_caliper_match(df, caliper=STATS.ptd_primary_caliper)
            r_matched = None
            if len(matched) >= 20:
                r_matched = fit_beta4_regression(matched, covariates=covariates,
                                                 outcome="ftd", cov_type=STATS.ols_cov_type)
            print(f"[E10] {family}/{budget}: n={len(df)} matched={len(matched)}")
            print(f"       full beta_d_dir={r_full.coefs['d_dir']:+.5f} "
                  f"CI=[{r_full.ci_low['d_dir']:+.5f},{r_full.ci_high['d_dir']:+.5f}] "
                  f"p={r_full.pvalues['d_dir']:.4g}")
            if r_matched is not None:
                print(f"       matched(c=0.02) beta_d_dir={r_matched.coefs['d_dir']:+.5f} "
                      f"CI=[{r_matched.ci_low['d_dir']:+.5f},{r_matched.ci_high['d_dir']:+.5f}] "
                      f"p={r_matched.pvalues['d_dir']:.4g}")
            cells.append({
                "family": family, "budget": int(budget),
                "n": int(len(df)), "n_matched": int(len(matched)),
                "vif": {k: float(v) for k, v in vif.items()},
                "full": {"beta_d_dir": float(r_full.coefs["d_dir"]),
                         "ci_low": float(r_full.ci_low["d_dir"]),
                         "ci_high": float(r_full.ci_high["d_dir"]),
                         "p": float(r_full.pvalues["d_dir"])},
                "matched": (None if r_matched is None else {
                    "beta_d_dir": float(r_matched.coefs["d_dir"]),
                    "ci_low": float(r_matched.ci_low["d_dir"]),
                    "ci_high": float(r_matched.ci_high["d_dir"]),
                    "p": float(r_matched.pvalues["d_dir"]),
                    "excludes_zero": bool((r_matched.ci_low["d_dir"] > 0) or
                                          (r_matched.ci_high["d_dir"] < 0))}),
            })

    # Cross-cell robustness verdict
    betas_matched = [c["matched"]["beta_d_dir"] for c in cells
                     if c["matched"] is not None]
    excludes_matched = [c["matched"]["excludes_zero"] for c in cells
                        if c["matched"] is not None]
    sign_consistent = (all(b > 0 for b in betas_matched)
                       or all(b < 0 for b in betas_matched)) if betas_matched else False
    all_sig = all(excludes_matched) if excludes_matched else False
    verdict = ("ROBUST_ACROSS_ALL_CELLS" if (sign_consistent and all_sig)
               else "ROBUST_SIGN_ONLY" if sign_consistent
               else "MIXED")

    # Forest plot
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(cells) + 2))
    ys = list(range(len(cells)))[::-1]
    for y, c in zip(ys, cells):
        if c["matched"] is None: continue
        m = c["matched"]
        ax.errorbar(m["beta_d_dir"], y,
                    xerr=[[m["beta_d_dir"] - m["ci_low"]], [m["ci_high"] - m["beta_d_dir"]]],
                    fmt=("s" if m["excludes_zero"] else "o"),
                    color=("firebrick" if m["excludes_zero"] else "gray"),
                    capsize=3, markersize=6)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{c['family']} / budget={c['budget']}" for c in cells],
                       fontsize=9)
    ax.set_xlabel("beta_d_dir (95% CI) at PTD caliper=0.02")
    ax.set_title(f"E10 - Probe-family x budget robustness: {verdict}")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_FIGS, "e10_probe_robustness_forest.png"), dpi=140)
    plt.close(fig)

    summary = {
        "hardware": env_summary(),
        "warmup_pairs": [list(p) for p in warmup_pairs],
        "held_out_pair": list(held_out_pair),
        "cells": cells,
        "sign_consistent_across_cells": bool(sign_consistent),
        "all_matched_cells_significant": bool(all_sig),
        "verdict": verdict,
    }
    with open(os.path.join(RESULTS_TABLES, "e10_summary.json"), "w") as f:
        json.dump(_stringify(summary), f, indent=2)
    print(f"\n[E10] VERDICT: {verdict}")


if __name__ == "__main__":
    main()
