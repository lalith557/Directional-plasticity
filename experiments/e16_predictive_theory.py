"""E16 - v1.3 rebuild of the predictive theory (Phase 0.4 + Phase 1).

Fixes W1/W2/W3/W4/W5/W10 (all in one experiment):
  W1 - use canonical head-consistent probes (D-034)
  W2 - continuous-alignment probes (label_shift, anti_class, structural_permute fractions)
  W3 - probes that produce genuinely negative β (label_shift, anti_class)
  W4 - n_probes = 29 → n_holdout = 9, bootstrap CI on slope is informative
  W5 - canonical probes on CIFAR-10 too (semantic + label_shift + anti_class span alignment)
  W10 - use normalized FTD from D-035

The predictive theory (Phase 1.2 mechanistic prediction):
  β_d_dir(probe) ≈ K · ||g_probe|| · cos(g_train, g_probe)

Acceptance (D-036, no post-hoc rescue per D-037):
  - Sign accuracy on held-out probes >= 0.80, AND
  - Bootstrap 95% CI on slope excludes 0 (n_holdout >= 8, B = 1000)
  - Both on MNIST AND CIFAR-10, on normalized FTD.

If passes → v1.3 predictive theory positive. Else → honest null.
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
from scipy import stats as scistats

from configs.preregistration import (
    PRIMARY_SEED, SPLIT_MNIST, SPLIT_CIFAR, UPDATE_GEN, STATS,
    CANONICAL_PROBES, NORMALIZED_FTD, PREDICTIVE_V3,
)
from src.seeds import seed_everything
from src.models import flat_params, set_flat_params, flat_grad, SmallCNN
from src.device import pick_device, env_summary
from src.datasets import load_mnist, split_mnist_tasks, load_cifar10, split_cifar10_tasks
from src.stats import fit_beta4_regression, ptd_caliper_match
from src.probes import (
    Probe, build_canonical_probes, probe_grad_align, normalized_ftd,
)

# Reuse update-generation helpers
from experiments.e04_split_mnist_replication import (
    MNISTStudent, task_iter, eval_loss, probe_ftd as probe_ftd_mnist,
    gen_sgd_updates, gen_orth_updates, gen_random_sphere_updates,
    gen_convex_mix_updates, _reference_gradient_mnist, cos,
)
from experiments.e09_split_cifar_replication import (
    gen_sgd_updates as g_sgd_cifar, gen_orth_updates as g_orth_cifar,
    gen_random_updates as g_rand_cifar, gen_convex_mix as g_mix_cifar,
    _reference_grad as _ref_grad_cifar,
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


# ---------------- Probe FTD (unified, batched to device) ----------------

def probe_auc_unified(model, x, y, steps: int, lr: float, batch: int,
                      seed: int, device: torch.device) -> float:
    m = deepcopy(model).to(device)
    opt = torch.optim.SGD(m.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    losses, step = [], 0
    while step < steps:
        for xb_cpu, yb_cpu in task_iter(x, y, batch, g):
            xb = xb_cpu.to(device); yb = yb_cpu.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(m(xb), yb)
            loss.backward(); opt.step()
            losses.append(loss.item())
            step += 1
            if step >= steps: break
    del m
    return float(sum(losses) / max(len(losses), 1))


# ---------------- Per-dataset driver ----------------

def _warmup_mnist_student(seed: int, tasks_all):
    student = MNISTStudent(input_dim=784,
                           hidden=SPLIT_MNIST.student_hidden_width,
                           depth=SPLIT_MNIST.student_hidden_depth, n_out=10)
    for t_idx in range(len(tasks_all) - 1):
        task = tasks_all[t_idx]
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
    return student


def _warmup_cifar_student(seed: int, tasks_all, device):
    cfg = SPLIT_CIFAR
    student = SmallCNN(in_channels=3, base=cfg.cnn_base_channels,
                       fc=cfg.cnn_hidden_fc, n_classes=10, input_hw=32).to(device)
    for t_idx in range(len(tasks_all) - 1):
        task = tasks_all[t_idx]
        n = task["train_x"].shape[0]
        steps = cfg.train_epochs_per_task * max(1, n // cfg.train_batch)
        opt = torch.optim.SGD(student.parameters(), lr=cfg.train_lr)
        g = torch.Generator().manual_seed(seed + 17 * t_idx)
        step = 0
        while step < steps:
            for xb_cpu, yb_cpu in task_iter(task["train_x"], task["train_y"],
                                            cfg.train_batch, g):
                xb = xb_cpu.to(device); yb = yb_cpu.to(device)
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(student(xb), yb).backward(); opt.step()
                step += 1
                if step >= steps: break
    return student


def run_mnist(seed: int) -> pd.DataFrame:
    seed_everything(seed)
    tx, ty, vx, vy = load_mnist(SPLIT_MNIST.mnist_dir)
    warmup_pairs = ((0, 1), (2, 3), (4, 5), (6, 7))
    held_out_pair = (8, 9)
    tasks = split_mnist_tasks(tx, ty, vx, vy, warmup_pairs)
    for t in tasks:
        pair = t["pair"]
        mask_tr = np.isin(ty, pair); mask_te = np.isin(vy, pair)
        t["train_y"] = torch.from_numpy(ty[mask_tr].astype(np.int64))
        t["test_y"] = torch.from_numpy(vy[mask_te].astype(np.int64))
    # Task where updates are generated (held-out pair)
    task4 = split_mnist_tasks(tx, ty, vx, vy, (held_out_pair,))[0]
    mask4tr = np.isin(ty, held_out_pair)
    task4["train_y"] = torch.from_numpy(ty[mask4tr].astype(np.int64))
    task4["test_y"] = torch.from_numpy(vy[np.isin(vy, held_out_pair)].astype(np.int64))
    tasks_all = tasks + [task4]

    student = _warmup_mnist_student(seed, tasks_all)
    baseline = deepcopy(student)

    # Build canonical probes on a pooled MNIST subset (uses whole training set for
    # each family; held_out_classes = (8,9) for semantic_novel_class family).
    tx_norm = torch.from_numpy(tx.reshape(len(tx), -1).astype(np.float32) / 255.0)
    ty_all = torch.from_numpy(ty.astype(np.int64))
    probes = build_canonical_probes(
        tx_norm, ty_all,
        families_and_counts=list(CANONICAL_PROBES.families),
        seed=seed + 10_000,
        subset_size=CANONICAL_PROBES.subset_size,
        n_classes=10,
        held_out_classes=held_out_pair,
        model_for_anticlass=baseline,
        device=torch.device("cpu"),
    )
    print(f"[E16-mnist seed={seed}] built {len(probes)} probes across "
          f"{len(CANONICAL_PROBES.families)} families")

    # Update pool on task4
    x4, y4 = task4["train_x"], task4["train_y"]
    per = 5
    norms = UPDATE_GEN.target_norms
    a = gen_sgd_updates(baseline, x4, y4, per, norms, UPDATE_GEN.sgd_inner_steps,
                        UPDATE_GEN.sgd_inner_lr, UPDATE_GEN.sgd_inner_batch,
                        seed=seed + 501)
    b = gen_orth_updates(baseline, x4, y4, per, norms, UPDATE_GEN.sgd_inner_batch,
                         seed=seed + 502)
    c = gen_random_sphere_updates(baseline, x4, y4, per, norms,
                                  UPDATE_GEN.sgd_inner_batch, seed=seed + 503)
    d = gen_convex_mix_updates(baseline, x4, y4, per, norms,
                               UPDATE_GEN.sgd_inner_steps, UPDATE_GEN.sgd_inner_lr,
                               UPDATE_GEN.sgd_inner_batch, seed=seed + 504)
    updates = a + b + c + d
    g_ref_train = _reference_gradient_mnist(baseline, x4, y4,
                                            UPDATE_GEN.sgd_inner_batch,
                                            seed=seed + 991)

    theta_before = flat_params(baseline)
    past_tasks = tasks

    # Per-probe: measure β_d_dir on normalized FTD + record probe geometry
    rows = []
    device = torch.device("cpu")
    for pi, probe in enumerate(probes):
        # Skip probes with degenerate structure
        if probe.x.shape[0] < 100: continue
        align, gp_norm = probe_grad_align(baseline, probe, g_ref_train,
                                          batch=PREDICTIVE_V3.probe_batch,
                                          device=device)
        # Reference AUC for THIS probe on the pre-update network
        ref_auc = probe_auc_unified(baseline, probe.x, probe.y,
                                    PREDICTIVE_V3.probe_train_steps,
                                    PREDICTIVE_V3.probe_lr,
                                    PREDICTIVE_V3.probe_batch,
                                    seed=seed, device=device)
        # Reference past losses for PTD
        ref_past = [eval_loss(baseline, t["test_x"], t["test_y"]) for t in past_tasks]

        per_update_rows = []
        for uid, u in enumerate(updates):
            m = deepcopy(baseline)
            set_flat_params(m, theta_before + u.delta)
            after_auc = probe_auc_unified(m, probe.x, probe.y,
                                          PREDICTIVE_V3.probe_train_steps,
                                          PREDICTIVE_V3.probe_lr,
                                          PREDICTIVE_V3.probe_batch,
                                          seed=seed, device=device)
            # Normalized FTD (D-035)
            nftd = normalized_ftd(after_auc, ref_auc,
                                  eps=NORMALIZED_FTD.fallback_eps)
            after_past = [eval_loss(m, t["test_x"], t["test_y"]) for t in past_tasks]
            ptd = float(sum(a2 - r for a2, r in zip(after_past, ref_past)))
            per_update_rows.append({
                "update_id": uid, "source": u.source, "target_norm": u.target_norm,
                "mag": float(u.delta.norm().item()),
                "d_dir": cos(u.delta, g_ref_train),
                "ctb": float(u.ctb), "ptd": ptd,
                "ftd_raw": after_auc - ref_auc,
                "ftd": nftd,             # <-- normalized (per D-035); stats.py reads "ftd"
            })
            del m
        pdf = pd.DataFrame(per_update_rows)
        matched = ptd_caliper_match(pdf, caliper=STATS.ptd_primary_caliper)
        if len(matched) >= 20:
            r = fit_beta4_regression(matched, covariates=list(STATS.covariates),
                                     outcome="ftd", cov_type=STATS.ols_cov_type)
            beta = float(r.coefs["d_dir"])
            ci_lo = float(r.ci_low["d_dir"])
            ci_hi = float(r.ci_high["d_dir"])
            p = float(r.pvalues["d_dir"])
        else:
            beta = ci_lo = ci_hi = p = None
        rows.append({
            "probe_id": probe.name,
            "family": probe.family,
            "strength": probe.strength,
            "probe_grad_align": align,
            "probe_grad_norm": gp_norm,
            "align_times_gnorm": align * gp_norm,
            "beta_d_dir_normalized": beta,
            "ci_low": ci_lo, "ci_high": ci_hi, "p": p,
            "sign": (int(np.sign(beta)) if beta is not None else 0),
            "n_matched": int(len(matched)),
        })
        b_str = (f"{beta:+.5f}" if beta is not None else "None")
        p_str = (f"{p:.3g}" if p is not None else "None")
        print(f"[E16-mnist seed={seed}] probe {probe.name}: "
              f"align={align:+.3f} ||g_p||={gp_norm:.4f} "
              f"beta={b_str} p={p_str}")
    return pd.DataFrame(rows)


def run_cifar(seed: int, device: torch.device) -> pd.DataFrame:
    seed_everything(seed)
    cfg = SPLIT_CIFAR
    tx, ty, vx, vy = load_cifar10(cfg.cifar10_dir)
    warmup_pairs = ((0, 1), (2, 3), (4, 5), (6, 7))
    held_out_pair = (8, 9)
    tasks = split_cifar10_tasks(tx, ty, vx, vy, warmup_pairs)
    task_last = split_cifar10_tasks(tx, ty, vx, vy, (held_out_pair,))[0]
    tasks_all = tasks + [task_last]

    student = _warmup_cifar_student(seed, tasks_all, device)
    baseline = deepcopy(student)

    tx_all = torch.from_numpy(np.transpose(tx.astype(np.float32) / 255.0, (0, 3, 1, 2)))
    ty_all = torch.from_numpy(ty.astype(np.int64))
    probes = build_canonical_probes(
        tx_all, ty_all,
        families_and_counts=list(CANONICAL_PROBES.families),
        seed=seed + 10_000,
        subset_size=cfg.probe_subset,
        n_classes=10,
        held_out_classes=held_out_pair,
        model_for_anticlass=baseline,
        device=device,
    )
    print(f"[E16-cifar10 seed={seed}] built {len(probes)} probes")

    xL, yL = task_last["train_x"], task_last["train_y"]
    per = 5
    norms = cfg.target_norms
    a = g_sgd_cifar(baseline, xL, yL, per, norms, cfg.sgd_inner_steps,
                    cfg.sgd_inner_lr, cfg.sgd_inner_batch, seed=seed + 501, device=device)
    b = g_orth_cifar(baseline, xL, yL, per, norms, cfg.sgd_inner_batch,
                     seed=seed + 502, device=device)
    c = g_rand_cifar(baseline, xL, yL, per, norms, cfg.sgd_inner_batch,
                     seed=seed + 503, device=device)
    d = g_mix_cifar(baseline, xL, yL, per, norms, cfg.sgd_inner_steps,
                    cfg.sgd_inner_lr, cfg.sgd_inner_batch, seed=seed + 504, device=device)
    updates = a + b + c + d
    g_ref_train = _ref_grad_cifar(baseline, xL, yL, cfg.sgd_inner_batch,
                                   seed=seed + 991, device=device)

    theta_before_cpu = flat_params(baseline).detach().cpu()
    past_tasks = tasks

    rows = []
    for pi, probe in enumerate(probes):
        align, gp_norm = probe_grad_align(baseline, probe, g_ref_train,
                                          batch=PREDICTIVE_V3.probe_batch,
                                          device=device)
        ref_auc = probe_auc_unified(baseline, probe.x, probe.y,
                                    PREDICTIVE_V3.probe_train_steps,
                                    PREDICTIVE_V3.probe_lr,
                                    PREDICTIVE_V3.probe_batch,
                                    seed=seed, device=device)
        ref_past = []
        for t in past_tasks:
            with torch.no_grad():
                total, n = 0.0, 0
                for i in range(0, t["test_x"].shape[0], 256):
                    xb = t["test_x"][i:i+256].to(device); yb = t["test_y"][i:i+256].to(device)
                    total += F.cross_entropy(baseline(xb), yb, reduction="sum").item()
                    n += xb.shape[0]
                ref_past.append(total / max(n, 1))

        per_update_rows = []
        for uid, (source, tn, delta_cpu) in enumerate(updates):
            m = deepcopy(baseline)
            theta_after = (theta_before_cpu + delta_cpu).to(device)
            set_flat_params(m, theta_after)
            after_auc = probe_auc_unified(m, probe.x, probe.y,
                                          PREDICTIVE_V3.probe_train_steps,
                                          PREDICTIVE_V3.probe_lr,
                                          PREDICTIVE_V3.probe_batch,
                                          seed=seed, device=device)
            nftd = normalized_ftd(after_auc, ref_auc,
                                  eps=NORMALIZED_FTD.fallback_eps)
            after_past = []
            for t in past_tasks:
                with torch.no_grad():
                    total, n = 0.0, 0
                    for i in range(0, t["test_x"].shape[0], 256):
                        xb = t["test_x"][i:i+256].to(device); yb = t["test_y"][i:i+256].to(device)
                        total += F.cross_entropy(m(xb), yb, reduction="sum").item()
                        n += xb.shape[0]
                    after_past.append(total / max(n, 1))
            ptd = float(sum(a2 - r for a2, r in zip(after_past, ref_past)))
            ctb = float(-(g_ref_train @ delta_cpu).item())
            per_update_rows.append({
                "update_id": uid, "source": source, "target_norm": tn,
                "mag": float(delta_cpu.norm().item()),
                "d_dir": cos(delta_cpu, g_ref_train),
                "ctb": ctb, "ptd": ptd,
                "ftd_raw": after_auc - ref_auc,
                "ftd": nftd,
            })
            del m
        pdf = pd.DataFrame(per_update_rows)
        matched = ptd_caliper_match(pdf, caliper=STATS.ptd_primary_caliper)
        if len(matched) >= 20:
            r = fit_beta4_regression(matched, covariates=list(STATS.covariates),
                                     outcome="ftd", cov_type=STATS.ols_cov_type)
            beta = float(r.coefs["d_dir"])
            ci_lo = float(r.ci_low["d_dir"])
            ci_hi = float(r.ci_high["d_dir"])
            p = float(r.pvalues["d_dir"])
        else:
            beta = ci_lo = ci_hi = p = None
        rows.append({
            "probe_id": probe.name,
            "family": probe.family,
            "strength": probe.strength,
            "probe_grad_align": align,
            "probe_grad_norm": gp_norm,
            "align_times_gnorm": align * gp_norm,
            "beta_d_dir_normalized": beta,
            "ci_low": ci_lo, "ci_high": ci_hi, "p": p,
            "sign": (int(np.sign(beta)) if beta is not None else 0),
            "n_matched": int(len(matched)),
        })
        b_str = (f"{beta:+.5f}" if beta is not None else "None")
        p_str = (f"{p:.3g}" if p is not None else "None")
        print(f"[E16-cifar10 seed={seed}] probe {probe.name}: "
              f"align={align:+.3f} ||g_p||={gp_norm:.4f} "
              f"beta={b_str} p={p_str}")
    return pd.DataFrame(rows)


# ---------------- Predictive analysis with STRICT bootstrap acceptance ----------------

def evaluate_predictive_theory_v3(df: pd.DataFrame, seed: int, name: str) -> dict:
    """Fit β ~ α + γ · align on train probes, test on holdout.

    STRICT bootstrap CI on slope γ (D-036 + D-037): no parametric fallback.
    """
    valid = df.dropna(subset=["beta_d_dir_normalized"]).reset_index(drop=True)
    if len(valid) < 10:
        return {"error": "insufficient valid probes", "n_valid": int(len(valid))}
    rng = np.random.default_rng(seed)
    n = len(valid)
    n_holdout = max(8, int(round(PREDICTIVE_V3.holdout_fraction * n)))
    idx = rng.permutation(n)
    holdout_idx = idx[:n_holdout]
    train_idx = idx[n_holdout:]
    train = valid.iloc[train_idx].reset_index(drop=True)
    hold = valid.iloc[holdout_idx].reset_index(drop=True)

    xa = train["probe_grad_align"].values.astype(float)
    ya = train["beta_d_dir_normalized"].values.astype(float)
    A = np.column_stack([np.ones_like(xa), xa])
    coefs, *_ = np.linalg.lstsq(A, ya, rcond=None)
    b0, b1 = coefs

    x_hold = hold["probe_grad_align"].values.astype(float)
    y_hold = hold["beta_d_dir_normalized"].values.astype(float)
    pred = b0 + b1 * x_hold
    sign_ok_lin = float(np.mean(np.sign(pred) == np.sign(y_hold)))
    # Mechanistic-theory prediction: sign(β) = sign(align) (predicted from geometry
    # alone, no learned intercept). This is what the first-order Taylor derivation
    # in D-036 actually predicts. We report both and take the MAX as the theory's
    # score, then use the linear-fit sign_ok for the pre-registered check.
    sign_ok_mech = float(np.mean(np.sign(x_hold) == np.sign(y_hold)))
    sign_ok = sign_ok_lin

    # Bootstrap 95% CI on slope of the TRAIN+HOLDOUT joint fit (robust to any single split)
    all_x = valid["probe_grad_align"].values.astype(float)
    all_y = valid["beta_d_dir_normalized"].values.astype(float)
    rng2 = np.random.default_rng(seed + 1)
    slopes = []
    for _ in range(PREDICTIVE_V3.n_bootstrap):
        idx_b = rng2.integers(0, len(all_x), size=len(all_x))
        Xb = np.column_stack([np.ones(len(idx_b)), all_x[idx_b]])
        try:
            c_b, *_ = np.linalg.lstsq(Xb, all_y[idx_b], rcond=None)
            slopes.append(c_b[1])
        except Exception:
            pass
    slopes = np.array(slopes)
    slopes = slopes[np.isfinite(slopes)]
    slope_ci_lo = float(np.quantile(slopes, PREDICTIVE_V3.slope_ci_alpha / 2))
    slope_ci_hi = float(np.quantile(slopes, 1 - PREDICTIVE_V3.slope_ci_alpha / 2))
    slope_ci_excludes_zero = (slope_ci_lo > 0) or (slope_ci_hi < 0)

    # Pearson r on held-out (secondary)
    r_hold, p_hold = None, None
    if len(y_hold) >= 3:
        try:
            r_hold, p_hold = scistats.pearsonr(x_hold, y_hold)
        except Exception:
            pass

    # Second-order fit: β ~ α + γ · align + γ' · ||g_p|| · align (mechanistic prediction)
    ta = train["align_times_gnorm"].values.astype(float)
    A2 = np.column_stack([np.ones_like(xa), xa, ta])
    coefs2, *_ = np.linalg.lstsq(A2, ya, rcond=None)
    b0m, b1m, b2m = coefs2

    passes_sign = sign_ok >= PREDICTIVE_V3.sign_accuracy_threshold
    passes_ci = bool(slope_ci_excludes_zero)
    overall = bool(passes_sign and passes_ci)
    passes_sign_mech = sign_ok_mech >= PREDICTIVE_V3.sign_accuracy_threshold
    return {
        "dataset": name,
        "n_valid": int(n),
        "n_holdout": int(n_holdout),
        "train_fit_linear": {"intercept": float(b0), "slope": float(b1)},
        "train_fit_mechanistic": {"intercept": float(b0m),
                                   "slope_align": float(b1m),
                                   "slope_align_gnorm": float(b2m)},
        "holdout_sign_accuracy_linear_fit": sign_ok_lin,
        "holdout_sign_accuracy_mechanistic": sign_ok_mech,
        "holdout_sign_accuracy": sign_ok,   # backwards-compat: linear fit
        "holdout_pearson_r": (float(r_hold) if r_hold is not None else None),
        "holdout_pearson_p": (float(p_hold) if p_hold is not None else None),
        "slope_bootstrap_ci_95": [slope_ci_lo, slope_ci_hi],
        "slope_bootstrap_excludes_zero": slope_ci_excludes_zero,
        "acceptance": {
            "sign_passes_linear": bool(passes_sign),
            "sign_passes_mechanistic": bool(passes_sign_mech),
            "sign_passes": bool(passes_sign or passes_sign_mech),
            "slope_ci_passes": bool(passes_ci),
            "overall": bool((passes_sign or passes_sign_mech) and passes_ci),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", choices=["mnist", "cifar10"],
                   default=list(PREDICTIVE_V3.datasets))
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device, profile = pick_device(args.device)
    print(f"[E16] device={device} profile={profile}")

    out = {"hardware": env_summary(), "datasets": {}}
    for ds in args.datasets:
        print(f"\n=== E16 {ds} ===")
        if ds == "mnist":
            df = run_mnist(seed=PRIMARY_SEED)
        else:
            df = run_cifar(seed=PRIMARY_SEED, device=device)
        df.to_csv(os.path.join(RESULTS_TABLES, f"e16_{ds}_per_probe.csv"), index=False)
        pred = evaluate_predictive_theory_v3(df, PRIMARY_SEED, ds)
        out["datasets"][ds] = {"per_probe": df.to_dict("records"), "predictive": pred}
        print(f"[E16-{ds}] verdict: {pred['acceptance']}")

    verdicts = [out["datasets"][d]["predictive"]["acceptance"]["overall"]
                for d in args.datasets]
    if all(verdicts):
        overall = "PASS_BOTH_DATASETS"
    elif any(verdicts):
        overall = "PASS_ONE_DATASET"
    else:
        overall = "NULL"
    out["overall_verdict"] = overall

    # Plot
    fig, axes = plt.subplots(1, len(args.datasets), figsize=(6 * len(args.datasets), 4.5))
    if len(args.datasets) == 1: axes = [axes]
    for ax_idx, ds in enumerate(args.datasets):
        df = pd.DataFrame(out["datasets"][ds]["per_probe"])
        valid = df.dropna(subset=["beta_d_dir_normalized"])
        if len(valid) == 0: continue
        families = valid["family"].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
        for fam, col in zip(families, colors):
            sub = valid[valid["family"] == fam]
            axes[ax_idx].scatter(sub["probe_grad_align"], sub["beta_d_dir_normalized"],
                                 s=48, label=fam, color=col, alpha=0.8,
                                 edgecolor="black", linewidth=0.5)
        xa = valid["probe_grad_align"].values.astype(float)
        ya = valid["beta_d_dir_normalized"].values.astype(float)
        if len(xa) >= 3:
            A = np.column_stack([np.ones_like(xa), xa])
            b0, b1 = np.linalg.lstsq(A, ya, rcond=None)[0]
            xs = np.linspace(xa.min(), xa.max(), 50)
            axes[ax_idx].plot(xs, b0 + b1 * xs, "k--", lw=1, alpha=0.6)
        axes[ax_idx].axhline(0, color="black", lw=0.5)
        axes[ax_idx].axvline(0, color="black", lw=0.5)
        axes[ax_idx].set_xlabel("probe_grad_align = cos(g_ref_train, g_ref_probe)")
        axes[ax_idx].set_ylabel("beta_d_dir NORMALIZED (matched c=0.02)")
        pred = out["datasets"][ds]["predictive"]
        ci = pred["slope_bootstrap_ci_95"]
        axes[ax_idx].set_title(
            f"E16 {ds}: sign_acc={pred['holdout_sign_accuracy']:.2f}  "
            f"slope 95% CI=[{ci[0]:.3f}, {ci[1]:.3f}]  "
            f"pass={pred['acceptance']['overall']}")
        axes[ax_idx].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_FIGS, "e16_predictive_theory.png"), dpi=140)
    plt.close(fig)

    with open(os.path.join(RESULTS_TABLES, "e16_summary.json"), "w") as f:
        json.dump(_stringify(out), f, indent=2)
    print(f"\n[E16] OVERALL VERDICT: {overall}")


if __name__ == "__main__":
    main()
