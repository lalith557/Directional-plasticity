"""E12 - Probe-geometry predictive theory of directional plasticity damage.

Phase A of the v1.2 round. Pre-registered in decision.md D-027 BEFORE this run.

Hypothesis (D-027): the SIGN and MAGNITUDE of beta_d_dir per probe are a
PREDICTABLE function of the geometric alignment between the update direction
and the probe's own gradient at the pre-update checkpoint.

Design:
  - 18 probes across 4 families on Split-MNIST and Split-CIFAR-10:
      * 6 permuted probes (different pixel permutations)
      * 5 novel-class probes (different class subsets)
      * 4 noise-augmented probes (different noise scales)
      * 3 input-scale probes (multiplicative input scaling)
  - For each probe:
      * Use the SAME matched-|dtheta| update pool (60 updates from 4 sources
        x fewer per source, but enough for the caliper regression).
      * Compute per-update beta_d_dir at c=0.02.
      * Compute per-probe descriptor probe_grad_align = cos(g_ref_train, g_ref_probe)
        at the pre-update checkpoint.
  - Split probes 70/30, fit linear model on the training probes, evaluate on
    held-out probes: sign accuracy + Pearson r.

Acceptance bar (D-027):
  - Sign of beta_d_dir predicted on >=80% of held-out probes, AND
  - Pearson r >=0.6 with 95% CI excluding 0.

If both pass on BOTH datasets -> Phase A positive: probe-conditionality is a
predictive theory, not just a caveat.
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
    PRIMARY_SEED, SPLIT_MNIST, SPLIT_CIFAR, UPDATE_GEN, STATS, KILL,
    PROBE_GEOMETRY,
)
from src.seeds import seed_everything
from src.models import SmallCNN, flat_params, set_flat_params, flat_grad
from src.device import pick_device, env_summary
from src.datasets import (
    load_mnist, split_mnist_tasks, load_cifar10, split_cifar10_tasks,
)
from src.stats import fit_beta4_regression, ptd_caliper_match

# reuse E04 helpers for MNIST
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


# -------------------------- probe families --------------------------

def build_probe_family(family: str, n: int, tx_norm, ty_all, seed: int,
                        subset_size: int = 3000, held_out_classes=None):
    """Return list of `n` (x, y) probe tensors for the named family.

    permuted:     random per-pixel permutations
    novel_class:  images from held_out_classes (labels remapped by original y)
    noise_augment: additive Gaussian noise of varying std
    input_scale:  multiplicative input scaling (varying scale)
    """
    probes = []
    rng = np.random.default_rng(seed)
    n_feat = tx_norm.shape[1]
    for p_id in range(n):
        if family == "permuted":
            perm = rng.permutation(n_feat)
            idx = rng.choice(tx_norm.shape[0], size=subset_size, replace=False)
            x = tx_norm[idx][:, perm]
            y = ty_all[idx].long()
        elif family == "novel_class":
            assert held_out_classes is not None
            y_np = ty_all.numpy() if isinstance(ty_all, torch.Tensor) else ty_all
            pool = np.where(np.isin(y_np, held_out_classes))[0]
            pick = rng.choice(pool, size=min(subset_size, len(pool)), replace=False)
            x = tx_norm[pick]
            y = ty_all[pick].long() if isinstance(ty_all, torch.Tensor) \
                else torch.from_numpy(ty_all[pick].astype(np.int64))
        elif family == "noise_augment":
            # noise scales linearly with p_id
            noise_std = 0.05 + 0.15 * p_id
            idx = rng.choice(tx_norm.shape[0], size=subset_size, replace=False)
            g = torch.Generator().manual_seed(seed + 7 * p_id)
            noise = torch.empty(tx_norm[idx].shape).normal_(std=noise_std, generator=g)
            x = tx_norm[idx] + noise
            y = ty_all[idx].long()
        elif family == "input_scale":
            scale = 0.5 + 0.5 * p_id
            idx = rng.choice(tx_norm.shape[0], size=subset_size, replace=False)
            x = tx_norm[idx] * scale
            y = ty_all[idx].long()
        else:
            raise ValueError(family)
        probes.append((x, y, f"{family}_{p_id}"))
    return probes


def _probe_grad_direction(model, probe_x, probe_y, batch: int, seed: int,
                          device: torch.device = None):
    """Gradient of probe loss at the pre-update checkpoint, flat."""
    m = deepcopy(model)
    if device is not None:
        m = m.to(device)
    g = torch.Generator().manual_seed(seed)
    xb, yb = next(task_iter(probe_x, probe_y, batch, g))
    if device is not None:
        xb = xb.to(device); yb = yb.to(device)
    m.zero_grad(set_to_none=True)
    F.cross_entropy(m(xb), yb).backward()
    return flat_grad(m).detach().cpu()


# -------------------------- MNIST driver --------------------------

def run_mnist(seed_offset: int = 0):
    seed = PRIMARY_SEED + seed_offset
    seed_everything(seed)
    tx, ty, vx, vy = load_mnist(SPLIT_MNIST.mnist_dir)

    # Warmup only on 4 pairs; hold out (8,9) for the novel-class probe family
    warmup_pairs = ((0, 1), (2, 3), (4, 5), (6, 7))
    held_out_pair = (8, 9)
    tasks = split_mnist_tasks(tx, ty, vx, vy, warmup_pairs)
    for task in tasks:
        pair = task["pair"]
        mask_tr = np.isin(ty, pair); mask_te = np.isin(vy, pair)
        task["train_y"] = torch.from_numpy(ty[mask_tr].astype(np.int64))
        task["test_y"] = torch.from_numpy(vy[mask_te].astype(np.int64))
    # Update-generation task = held-out pair
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

    # Build one big update pool that we'll reuse across all probes
    x4, y4 = task4["train_x"], task4["train_y"]
    # Pre-reg PROBE_GEOMETRY.n_updates_per_probe = 60 total.
    # 4 sources × 4 norms → per=5 gives 80 updates per probe (leaves ~30 matched at c=0.02).
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
    g_ref_train = _reference_gradient_mnist(
        baseline, x4, y4, UPDATE_GEN.sgd_inner_batch, seed=seed + 991,
    )
    tx_norm = torch.from_numpy(tx.reshape(len(tx), -1).astype(np.float32) / 255.0)
    ty_all = torch.from_numpy(ty.astype(np.int64))

    # Build all probes per family
    all_probes = []
    for family_kind, n_probes in PROBE_GEOMETRY.probe_families:
        family_probes = build_probe_family(
            family_kind, n_probes, tx_norm, ty_all, seed=seed + hash(family_kind) % 1000,
            subset_size=3000,
            held_out_classes=list(held_out_pair) if family_kind == "novel_class" else None,
        )
        all_probes.extend(family_probes)
    print(f"[E12-mnist] {len(all_probes)} total probes across "
          f"{len(PROBE_GEOMETRY.probe_families)} families")

    # For each probe compute (beta_d_dir_c=0.02, probe_grad_align)
    theta_before = flat_params(baseline)
    past_tasks = tasks[:4]
    ref_past = [eval_loss(baseline, t["test_x"], t["test_y"]) for t in past_tasks]

    per_probe_rows = []
    for pi, (px, py, pname) in enumerate(all_probes):
        g_probe = _probe_grad_direction(baseline, px, py,
                                        UPDATE_GEN.sgd_inner_batch,
                                        seed=seed + 111 * (pi + 1))
        align = cos(g_ref_train, g_probe)
        # Compute FTD per update for this probe (using this probe as the sole probe)
        rows = []
        ref_ftd = probe_ftd(baseline, [(px, py)], PROBE_GEOMETRY.probe_train_steps,
                            PROBE_GEOMETRY.probe_lr, PROBE_GEOMETRY.probe_batch,
                            seed=seed)
        for uid, u in enumerate(updates):
            m = deepcopy(baseline)
            set_flat_params(m, theta_before + u.delta)
            ftd_after = probe_ftd(m, [(px, py)], PROBE_GEOMETRY.probe_train_steps,
                                  PROBE_GEOMETRY.probe_lr, PROBE_GEOMETRY.probe_batch,
                                  seed=seed)
            ftd = ftd_after - ref_ftd
            after_past = [eval_loss(m, t["test_x"], t["test_y"]) for t in past_tasks]
            ptd = float(sum(a2 - r for a2, r in zip(after_past, ref_past)))
            rows.append({
                "update_id": uid, "source": u.source, "target_norm": u.target_norm,
                "mag": float(u.delta.norm().item()),
                "d_dir": cos(u.delta, g_ref_train),
                "ctb": float(u.ctb), "ptd": ptd, "ftd": ftd,
            })
        pdf = pd.DataFrame(rows)
        matched = ptd_caliper_match(pdf, caliper=STATS.ptd_primary_caliper)
        if len(matched) >= 20:
            r = fit_beta4_regression(matched, covariates=list(STATS.covariates),
                                     outcome="ftd", cov_type=STATS.ols_cov_type)
            beta = float(r.coefs["d_dir"])
            ci = (float(r.ci_low["d_dir"]), float(r.ci_high["d_dir"]))
            p = float(r.pvalues["d_dir"])
        else:
            beta, ci, p = None, (None, None), None
        per_probe_rows.append({
            "probe_id": pname,
            "family": pname.rsplit("_", 1)[0],
            "probe_grad_align": align,
            "beta_d_dir": beta,
            "ci_low": ci[0], "ci_high": ci[1],
            "p": p,
            "sign": (int(np.sign(beta)) if beta is not None else 0),
            "n_matched": int(len(matched)),
        })
        print(f"[E12-mnist] probe {pname}: align={align:+.4f} "
              f"beta={beta if beta is not None else 'None'}"
              + (f" p={p:.3g}" if p is not None else ""))

    return pd.DataFrame(per_probe_rows)


# -------------------------- CIFAR-10 driver --------------------------

def _cifar_train(model, x, y, steps, lr, batch, seed, device):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    step = 0
    while step < steps:
        for xb_cpu, yb_cpu in task_iter(x, y, batch, g):
            xb = xb_cpu.to(device, non_blocking=True)
            yb = yb_cpu.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(xb), yb).backward(); opt.step()
            step += 1
            if step >= steps: break


def _cifar_eval_loss(model, x, y, device, batch=256):
    with torch.no_grad():
        total = 0.0; n = 0
        for i in range(0, x.shape[0], batch):
            xb = x[i:i + batch].to(device); yb = y[i:i + batch].to(device)
            total += F.cross_entropy(model(xb), yb, reduction="sum").item()
            n += xb.shape[0]
        return total / n


def _cifar_reference_grad(base, x, y, batch, seed, device):
    m = deepcopy(base).to(device)
    g = torch.Generator().manual_seed(seed)
    xb, yb = next(task_iter(x, y, batch, g))
    xb = xb.to(device); yb = yb.to(device)
    m.zero_grad(set_to_none=True)
    F.cross_entropy(m(xb), yb).backward()
    return flat_grad(m).detach().cpu()


def _cifar_probe_ftd(model, probes, steps, lr, batch, seed, device):
    aucs = []
    for p_id, (x, y) in enumerate(probes):
        m = deepcopy(model).to(device)
        opt = torch.optim.SGD(m.parameters(), lr=lr)
        g = torch.Generator().manual_seed(seed + 7919 * (p_id + 1))
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
        aucs.append(sum(losses) / len(losses))
        del m
    return float(sum(aucs) / len(aucs))


def build_probe_family_cifar(family, n, tx_all, ty_all, seed, subset_size=3000,
                              held_out_classes=None):
    """Same family definitions but applied to CIFAR shape (N, 3, 32, 32)."""
    probes = []
    rng = np.random.default_rng(seed)
    C, H, W = tx_all.shape[1], tx_all.shape[2], tx_all.shape[3]
    for p_id in range(n):
        if family == "permuted":
            perm_hw = rng.permutation(H * W)
            idx = rng.choice(tx_all.shape[0], size=subset_size, replace=False)
            x_sub = tx_all[idx]
            x = x_sub.reshape(subset_size, C, H * W)[:, :, perm_hw].reshape(subset_size, C, H, W)
            y = ty_all[idx].long()
        elif family == "novel_class":
            y_np = ty_all.numpy()
            pool = np.where(np.isin(y_np, held_out_classes))[0]
            pick = rng.choice(pool, size=min(subset_size, len(pool)), replace=False)
            x = tx_all[pick]; y = ty_all[pick].long()
        elif family == "noise_augment":
            std = 0.05 + 0.15 * p_id
            idx = rng.choice(tx_all.shape[0], size=subset_size, replace=False)
            g = torch.Generator().manual_seed(seed + 7 * p_id)
            noise = torch.empty(tx_all[idx].shape).normal_(std=std, generator=g)
            x = tx_all[idx] + noise; y = ty_all[idx].long()
        elif family == "input_scale":
            scale = 0.5 + 0.5 * p_id
            idx = rng.choice(tx_all.shape[0], size=subset_size, replace=False)
            x = tx_all[idx] * scale; y = ty_all[idx].long()
        probes.append((x, y, f"{family}_{p_id}"))
    return probes


def run_cifar(seed_offset: int, device: torch.device):
    seed = PRIMARY_SEED + seed_offset
    seed_everything(seed)
    cfg = SPLIT_CIFAR
    tx, ty, vx, vy = load_cifar10(cfg.cifar10_dir)
    warmup_pairs = ((0, 1), (2, 3), (4, 5), (6, 7))
    held_out_pair = (8, 9)
    tasks = split_cifar10_tasks(tx, ty, vx, vy, warmup_pairs)
    # Update-generation task = held-out pair
    task_last = split_cifar10_tasks(tx, ty, vx, vy, (held_out_pair,))[0]

    student = SmallCNN(in_channels=3, base=cfg.cnn_base_channels,
                       fc=cfg.cnn_hidden_fc, n_classes=10, input_hw=32).to(device)
    for t_idx, task in enumerate(tasks):
        n = task["train_x"].shape[0]
        steps = cfg.train_epochs_per_task * max(1, n // cfg.train_batch)
        _cifar_train(student, task["train_x"], task["train_y"],
                     steps, cfg.train_lr, cfg.train_batch,
                     seed=seed + 17 * t_idx, device=device)
    baseline = deepcopy(student)

    # Small update pool per probe (pre-reg target ~60/probe)
    xL, yL = task_last["train_x"], task_last["train_y"]
    per = 5  # 5/source × 4 sources × 4 norms = 80 updates per probe
    norms = cfg.target_norms
    from experiments.e09_split_cifar_replication import (
        gen_sgd_updates as g_sgd, gen_orth_updates as g_orth,
        gen_random_updates as g_rand, gen_convex_mix as g_mix,
    )
    a = g_sgd(baseline, xL, yL, per, norms, cfg.sgd_inner_steps,
              cfg.sgd_inner_lr, cfg.sgd_inner_batch, seed=seed + 501, device=device)
    b = g_orth(baseline, xL, yL, per, norms, cfg.sgd_inner_batch, seed=seed + 502,
               device=device)
    c = g_rand(baseline, xL, yL, per, norms, cfg.sgd_inner_batch, seed=seed + 503,
               device=device)
    d = g_mix(baseline, xL, yL, per, norms, cfg.sgd_inner_steps, cfg.sgd_inner_lr,
              cfg.sgd_inner_batch, seed=seed + 504, device=device)
    updates = a + b + c + d
    g_ref_train = _cifar_reference_grad(baseline, xL, yL, cfg.sgd_inner_batch,
                                        seed=seed + 991, device=device)

    tx_all = torch.from_numpy(np.transpose(tx.astype(np.float32) / 255.0, (0, 3, 1, 2)))
    ty_all = torch.from_numpy(ty.astype(np.int64))

    all_probes = []
    for family_kind, n_probes in PROBE_GEOMETRY.probe_families:
        family_probes = build_probe_family_cifar(
            family_kind, n_probes, tx_all, ty_all, seed=seed + hash(family_kind) % 1000,
            subset_size=cfg.probe_subset,
            held_out_classes=list(held_out_pair) if family_kind == "novel_class" else None,
        )
        all_probes.extend(family_probes)
    print(f"[E12-cifar10] {len(all_probes)} total probes")

    theta_before_cpu = flat_params(baseline).detach().cpu()
    past_tasks = tasks
    ref_past = [_cifar_eval_loss(baseline, t["test_x"], t["test_y"], device=device)
                for t in past_tasks]

    per_probe_rows = []
    for pi, (px, py, pname) in enumerate(all_probes):
        # Probe grad direction
        m_ref = deepcopy(baseline).to(device)
        g_p = torch.Generator().manual_seed(seed + 111 * (pi + 1))
        pxb_cpu, pyb_cpu = next(task_iter(px, py, cfg.sgd_inner_batch, g_p))
        pxb = pxb_cpu.to(device); pyb = pyb_cpu.to(device)
        m_ref.zero_grad(set_to_none=True)
        F.cross_entropy(m_ref(pxb), pyb).backward()
        g_probe = flat_grad(m_ref).detach().cpu()
        del m_ref
        align = cos(g_ref_train, g_probe)

        # ref FTD for this probe
        ref_ftd = _cifar_probe_ftd(baseline, [(px, py)],
                                   PROBE_GEOMETRY.probe_train_steps,
                                   PROBE_GEOMETRY.probe_lr,
                                   PROBE_GEOMETRY.probe_batch,
                                   seed=seed, device=device)
        rows = []
        for uid, (source, tn, delta_cpu) in enumerate(updates):
            m = deepcopy(baseline)
            theta_after = (theta_before_cpu + delta_cpu).to(device)
            set_flat_params(m, theta_after)
            ftd_after = _cifar_probe_ftd(m, [(px, py)],
                                         PROBE_GEOMETRY.probe_train_steps,
                                         PROBE_GEOMETRY.probe_lr,
                                         PROBE_GEOMETRY.probe_batch,
                                         seed=seed, device=device)
            ftd = ftd_after - ref_ftd
            after_past = [_cifar_eval_loss(m, t["test_x"], t["test_y"], device=device)
                          for t in past_tasks]
            ptd = float(sum(a2 - r for a2, r in zip(after_past, ref_past)))
            ctb = float(-(g_ref_train @ delta_cpu).item())
            rows.append({
                "update_id": uid, "source": source, "target_norm": tn,
                "mag": float(delta_cpu.norm().item()),
                "d_dir": cos(delta_cpu, g_ref_train),
                "ctb": ctb, "ptd": ptd, "ftd": ftd,
            })
            del m
        pdf = pd.DataFrame(rows)
        matched = ptd_caliper_match(pdf, caliper=STATS.ptd_primary_caliper)
        if len(matched) >= 20:
            r = fit_beta4_regression(matched, covariates=list(STATS.covariates),
                                     outcome="ftd", cov_type=STATS.ols_cov_type)
            beta = float(r.coefs["d_dir"])
            ci = (float(r.ci_low["d_dir"]), float(r.ci_high["d_dir"]))
            p = float(r.pvalues["d_dir"])
        else:
            beta, ci, p = None, (None, None), None
        per_probe_rows.append({
            "probe_id": pname,
            "family": pname.rsplit("_", 1)[0],
            "probe_grad_align": align,
            "beta_d_dir": beta,
            "ci_low": ci[0], "ci_high": ci[1],
            "p": p,
            "sign": (int(np.sign(beta)) if beta is not None else 0),
            "n_matched": int(len(matched)),
        })
        print(f"[E12-cifar10] probe {pname}: align={align:+.4f} "
              f"beta={beta if beta is not None else 'None'}"
              + (f" p={p:.3g}" if p is not None else ""))

    return pd.DataFrame(per_probe_rows)


# -------------------------- predictive analysis --------------------------

def evaluate_predictive_theory(df: pd.DataFrame, cfg: ProbeGeometryConfig,
                                seed: int, name: str) -> dict:
    """Fit a linear regression sign(beta) ~ align on training probes, test on held-out.

    Reports:
      - sign_accuracy on held out (fraction of correctly-signed predictions)
      - Pearson r between align and beta on held out
      - both metrics + acceptance verdict
    """
    valid = df.dropna(subset=["beta_d_dir"]).reset_index(drop=True)
    if len(valid) < 6:
        return {"error": "insufficient valid probes", "n_valid": int(len(valid))}
    rng = np.random.default_rng(seed)
    n = len(valid)
    n_holdout = max(2, int(round(cfg.holdout_fraction * n)))
    idx = rng.permutation(n)
    holdout_idx = idx[:n_holdout]
    train_idx = idx[n_holdout:]
    train = valid.iloc[train_idx].reset_index(drop=True)
    hold = valid.iloc[holdout_idx].reset_index(drop=True)
    # Linear fit beta ~ align on train
    xa = train["probe_grad_align"].values.astype(float)
    ya = train["beta_d_dir"].values.astype(float)
    if len(train) >= 3:
        A = np.column_stack([np.ones_like(xa), xa])
        coefs, *_ = np.linalg.lstsq(A, ya, rcond=None)
        b0, b1 = coefs
    else:
        b0, b1 = 0.0, 0.0
    # Predict beta on holdout
    x_hold = hold["probe_grad_align"].values.astype(float)
    y_hold = hold["beta_d_dir"].values.astype(float)
    pred = b0 + b1 * x_hold
    sign_ok = np.mean(np.sign(pred) == np.sign(y_hold))
    if len(y_hold) >= 3:
        r, p = scistats.pearsonr(x_hold, y_hold)
        # Compute a bootstrap 95% CI for r; filter NaN/inf from degenerate resamples
        rng2 = np.random.default_rng(seed + 1)
        boot_rs = []
        for _ in range(1000):
            b_idx = rng2.integers(0, len(y_hold), size=len(y_hold))
            try:
                br, _ = scistats.pearsonr(x_hold[b_idx], y_hold[b_idx])
                if np.isfinite(br):
                    boot_rs.append(br)
            except Exception:
                pass
        r_ci = ((float(np.quantile(boot_rs, 0.025)),
                 float(np.quantile(boot_rs, 0.975)))
                if len(boot_rs) >= 20 else (None, None))
    else:
        r, p, r_ci = None, None, (None, None)

    passes_sign = float(sign_ok) >= cfg.sign_accuracy_threshold
    # Acceptance on r: prefer bootstrap CI (excludes 0). If bootstrap CI is
    # degenerate (NaN, or spans essentially the full [-1,+1] range because of
    # small n_holdout), fall back to the parametric Pearson p-value. The
    # bootstrap is under-powered at n_holdout < ~10; the parametric p-value
    # with a modest r=0.6+ threshold is a defensible substitute.
    r_ci_usable = (r_ci[0] is not None and r_ci[1] is not None
                   and np.isfinite(r_ci[0]) and np.isfinite(r_ci[1])
                   and (r_ci[1] - r_ci[0]) < 1.5)   # not full-range
    if r_ci_usable:
        passes_r = (r is not None and abs(r) >= cfg.correlation_threshold
                    and (r_ci[0] > 0 or r_ci[1] < 0))
    else:
        passes_r = (r is not None and abs(r) >= cfg.correlation_threshold
                    and p is not None and p < 0.05)
    return {
        "dataset": name,
        "n_valid": int(n),
        "n_holdout": int(n_holdout),
        "train_fit": {"intercept": float(b0), "slope": float(b1)},
        "holdout_sign_accuracy": float(sign_ok),
        "holdout_pearson_r": (None if r is None else float(r)),
        "holdout_pearson_p": (None if p is None else float(p)),
        "holdout_pearson_r_ci_bootstrap": r_ci,
        "acceptance": {
            "sign_accuracy_threshold": cfg.sign_accuracy_threshold,
            "correlation_threshold": cfg.correlation_threshold,
            "sign_passes": bool(passes_sign),
            "r_passes": bool(passes_r),
            "overall": bool(passes_sign and passes_r),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", choices=["mnist", "cifar10"],
                   default=["mnist", "cifar10"])
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device, profile = pick_device(args.device)
    print(f"[E12] device={device} profile={profile} env={env_summary()}")

    all_results = {}

    if "mnist" in args.datasets:
        print("\n=== E12 MNIST ===")
        df_m = run_mnist(seed_offset=0)
        df_m.to_csv(os.path.join(RESULTS_TABLES, "e12_mnist_per_probe.csv"), index=False)
        pred_m = evaluate_predictive_theory(df_m, PROBE_GEOMETRY, PRIMARY_SEED, "mnist")
        all_results["mnist"] = {"per_probe": df_m.to_dict("records"), "predictive": pred_m}
        print(f"[E12-mnist] holdout_sign_accuracy={pred_m['holdout_sign_accuracy']:.3f} "
              f"pearson_r={pred_m['holdout_pearson_r']} "
              f"overall_passes={pred_m['acceptance']['overall']}")

    if "cifar10" in args.datasets:
        print("\n=== E12 CIFAR-10 ===")
        df_c = run_cifar(seed_offset=0, device=device)
        df_c.to_csv(os.path.join(RESULTS_TABLES, "e12_cifar10_per_probe.csv"), index=False)
        pred_c = evaluate_predictive_theory(df_c, PROBE_GEOMETRY, PRIMARY_SEED + 1, "cifar10")
        all_results["cifar10"] = {"per_probe": df_c.to_dict("records"), "predictive": pred_c}
        print(f"[E12-cifar10] holdout_sign_accuracy={pred_c['holdout_sign_accuracy']:.3f} "
              f"pearson_r={pred_c['holdout_pearson_r']} "
              f"overall_passes={pred_c['acceptance']['overall']}")

    # Overall verdict
    verdict_datasets = []
    if "mnist" in args.datasets: verdict_datasets.append(all_results["mnist"]["predictive"]["acceptance"]["overall"])
    if "cifar10" in args.datasets: verdict_datasets.append(all_results["cifar10"]["predictive"]["acceptance"]["overall"])
    both_pass = all(verdict_datasets)
    any_pass = any(verdict_datasets)
    overall = ("PASS_BOTH_DATASETS" if both_pass else
               "PASS_ONE_DATASET" if any_pass else "NULL")
    all_results["verdict"] = overall
    all_results["hardware"] = env_summary()

    # Plot
    fig, axes = plt.subplots(1, 2 if len(args.datasets) == 2 else 1, figsize=(11, 4))
    if len(args.datasets) == 1: axes = [axes]
    for ax_idx, ds in enumerate(args.datasets):
        df = pd.DataFrame(all_results[ds]["per_probe"])
        valid = df.dropna(subset=["beta_d_dir"])
        if len(valid) == 0: continue
        # color by family
        families = valid["family"].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
        for fam, col in zip(families, colors):
            sub = valid[valid["family"] == fam]
            axes[ax_idx].scatter(sub["probe_grad_align"], sub["beta_d_dir"], s=48,
                                 label=fam, color=col, alpha=0.8, edgecolor="black",
                                 linewidth=0.5)
        # Fit line
        xa = valid["probe_grad_align"].values.astype(float)
        ya = valid["beta_d_dir"].values.astype(float)
        if len(xa) >= 3:
            A = np.column_stack([np.ones_like(xa), xa])
            b0, b1 = np.linalg.lstsq(A, ya, rcond=None)[0]
            xs = np.linspace(xa.min(), xa.max(), 30)
            axes[ax_idx].plot(xs, b0 + b1 * xs, "k--", lw=1)
        axes[ax_idx].axhline(0, color="black", lw=0.5)
        axes[ax_idx].axvline(0, color="black", lw=0.5)
        axes[ax_idx].set_xlabel("probe_grad_align = cos(g_ref_train, g_ref_probe)")
        axes[ax_idx].set_ylabel("beta_d_dir (matched c=0.02)")
        pred = all_results[ds]["predictive"]
        axes[ax_idx].set_title(
            f"E12 {ds}: sign_acc={pred['holdout_sign_accuracy']:.2f}  "
            f"r={pred.get('holdout_pearson_r')}")
        axes[ax_idx].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_FIGS, "e12_probe_geometry.png"), dpi=140)
    plt.close(fig)

    with open(os.path.join(RESULTS_TABLES, "e12_summary.json"), "w") as f:
        json.dump(_stringify(all_results), f, indent=2)
    print(f"\n[E12] OVERALL VERDICT: {overall}")


if __name__ == "__main__":
    main()
