"""E09 - Split-CIFAR replication (Phase 4).

Motivation: E04 established a small but reproducible directional signal on Split-MNIST
(beta_d_dir ~ +0.002, sign-stable across 5 seeds), while synthetic showed a null. The
open question from D-016 is whether that MNIST scalar is a **scale-specific artifact**
of a 118k-parameter fully-connected network, or whether it **replicates at CIFAR scale**
with a small convolutional network doing real representation learning.

Protocol: mirror E04 exactly, but on a SmallCNN and one of {Split-CIFAR-10, Split-CIFAR-100}
selected by CLI flag. Same 4-source update generator, same Control-C caliper sweep, same
kill-condition gates. Pre-registered in decision.md D-019.

Usage:
    python experiments/e09_split_cifar_replication.py --dataset cifar10
    python experiments/e09_split_cifar_replication.py --dataset cifar100 --cpu-scale 0.25
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
    PRIMARY_SEED, SPLIT_CIFAR, STATS, KILL, NOVEL_CLASS_CIFAR,
)
from src.seeds import seed_everything
from src.models import SmallCNN, flat_params, set_flat_params, flat_grad
from src.device import pick_device, env_summary
from src.datasets import (
    load_cifar10, load_cifar100, split_cifar10_tasks, split_cifar100_tasks,
)
from src.stats import (
    compute_vif, fit_beta4_regression, bootstrap_beta4, ptd_caliper_match,
    within_vs_between_caliper_variance,
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


def task_iter(x, y, batch, generator):
    n = x.shape[0]
    perm = torch.randperm(n, generator=generator)
    for i in range(0, n, batch):
        idx = perm[i:i + batch]
        yield x[idx], y[idx]


def eval_loss(model, x, y, device, batch=256):
    """Batched eval to fit in GPU memory."""
    with torch.no_grad():
        total = 0.0; n = 0
        for i in range(0, x.shape[0], batch):
            xb = x[i:i + batch].to(device, non_blocking=True)
            yb = y[i:i + batch].to(device, non_blocking=True)
            total += F.cross_entropy(model(xb), yb, reduction="sum").item()
            n += xb.shape[0]
        return total / n


def train_task(model, x, y, steps, lr, batch, seed, device):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    step, losses = 0, []
    while step < steps:
        for xb_cpu, yb_cpu in task_iter(x, y, batch, g):
            xb = xb_cpu.to(device, non_blocking=True)
            yb = yb_cpu.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            step += 1
            if step >= steps: break
    return losses


def build_probes(tx_norm, ty_all, n_probes, seed, subset_size, device):
    """Structural probes: random per-pixel permutations (channel-preserving)."""
    probes = []
    rng = np.random.default_rng(seed)
    C, H, W = tx_norm.shape[1], tx_norm.shape[2], tx_norm.shape[3]
    for p_id in range(n_probes):
        # Permute pixels within each channel, same permutation across all images
        perm_hw = rng.permutation(H * W)
        idx = rng.choice(tx_norm.shape[0], size=subset_size, replace=False)
        x_sub = tx_norm[idx]                       # (n, C, H, W)
        x_flat = x_sub.reshape(subset_size, C, H * W)
        x_perm = x_flat[:, :, perm_hw].reshape(subset_size, C, H, W)
        probes.append((x_perm, ty_all[idx].long()))
    return probes


def build_novel_class_probes(tx_norm_full, ty_full, held_out_classes: tuple,
                              n_probes: int, seed: int, subset_size: int):
    """Novel-class probes: images drawn ONLY from held-out classes.

    Labels are the ORIGINAL 10-way indices (the shared 10-way head handles both
    warmup pairs and the held-out pair). This mirrors the MNIST E10 protocol.
    """
    probes = []
    rng = np.random.default_rng(seed)
    y_np = ty_full.numpy() if isinstance(ty_full, torch.Tensor) else ty_full
    mask = np.isin(y_np, list(held_out_classes))
    idx_all = np.where(mask)[0]
    for p_id in range(n_probes):
        pick = rng.choice(idx_all, size=min(subset_size, len(idx_all)), replace=False)
        x = tx_norm_full[pick]
        y = ty_full[pick].long() if isinstance(ty_full, torch.Tensor) \
            else torch.from_numpy(ty_full[pick].astype(np.int64))
        probes.append((x, y))
    return probes


def probe_ftd(model, probes, steps, lr, batch, seed, device):
    aucs = []
    for p_id, (x, y) in enumerate(probes):
        m = deepcopy(model).to(device)
        opt = torch.optim.SGD(m.parameters(), lr=lr)
        g = torch.Generator().manual_seed(seed + 7919 * (p_id + 1))
        losses, step = [], 0
        while step < steps:
            for xb_cpu, yb_cpu in task_iter(x, y, batch, g):
                xb = xb_cpu.to(device, non_blocking=True)
                yb = yb_cpu.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(m(xb), yb)
                loss.backward(); opt.step()
                losses.append(loss.item())
                step += 1
                if step >= steps: break
        aucs.append(sum(losses) / len(losses))
        del m
    return float(sum(aucs) / len(aucs))


# ---------- update sources on CIFAR ----------

def _reference_grad(base, x, y, batch, seed, device):
    m = deepcopy(base).to(device)
    g = torch.Generator().manual_seed(seed)
    xb_cpu, yb_cpu = next(task_iter(x, y, batch, g))
    xb = xb_cpu.to(device); yb = yb_cpu.to(device)
    m.zero_grad(set_to_none=True)
    F.cross_entropy(m(xb), yb).backward()
    return flat_grad(m).detach().cpu()


def _renorm(delta, target):
    n = delta.norm()
    return delta if n < 1e-12 else delta * (target / n)


def gen_sgd_updates(base, x, y, per, norms, steps, lr, batch, seed, device):
    theta_before = flat_params(base).detach().cpu()
    out = []
    for k, norm in enumerate(norms):
        for s in range(per):
            m = deepcopy(base).to(device)
            train_task(m, x, y, steps, lr, batch, seed + 101 * k + s, device)
            d = (flat_params(m).detach().cpu() - theta_before).float()
            d = _renorm(d, norm)
            out.append(("sgd_seed", norm, d))
            del m
    return out


def gen_orth_updates(base, x, y, per, norms, batch, seed, device):
    theta_before = flat_params(base).detach().cpu()
    g_ref = _reference_grad(base, x, y, batch, seed + 991, device)
    g_hat = g_ref / (g_ref.norm() + 1e-12)
    D = g_ref.numel()
    out = []
    for k, norm in enumerate(norms):
        for s in range(per):
            g = torch.Generator().manual_seed(seed + 313 * k + 7 * s + 1)
            eps = torch.empty(D).normal_(generator=g)
            d_perp = eps - (eps @ g_hat) * g_hat
            out.append(("orth_perturb", norm, _renorm(d_perp, norm)))
    return out


def gen_random_updates(base, x, y, per, norms, batch, seed, device):
    theta_before = flat_params(base).detach().cpu()
    D = theta_before.numel()
    out = []
    for k, norm in enumerate(norms):
        for s in range(per):
            g = torch.Generator().manual_seed(seed + 991 * k + 13 * s + 3)
            v = torch.empty(D).normal_(generator=g)
            out.append(("random_sphere", norm, _renorm(v, norm)))
    return out


def gen_convex_mix(base, x, y, per, norms, steps, lr, batch, seed, device):
    theta_before = flat_params(base).detach().cpu()
    D = theta_before.numel()
    n_pool = max(per, 5)
    sgd_pool = []
    for i in range(n_pool):
        m = deepcopy(base).to(device)
        train_task(m, x, y, steps, lr, batch, seed + 707 * i + 1, device)
        d = flat_params(m).detach().cpu() - theta_before
        sgd_pool.append(d / (d.norm() + 1e-12))
        del m
    rand_pool = []
    for i in range(n_pool):
        g = torch.Generator().manual_seed(seed + 1213 * i + 2)
        v = torch.empty(D).normal_(generator=g)
        rand_pool.append(v / (v.norm() + 1e-12))
    alphas = torch.linspace(0.05, 0.95, 5).tolist()
    out = []
    for k, norm in enumerate(norms):
        for s in range(per):
            alpha = alphas[s % len(alphas)]
            u_sgd = sgd_pool[s % n_pool]
            u_rand = rand_pool[(s + k) % n_pool]
            mixed = alpha * u_sgd + (1 - alpha) * u_rand
            out.append(("convex_mix", norm, _renorm(mixed, norm)))
    return out


def cos(a, b):
    return float((a @ b / ((a.norm() + 1e-12) * (b.norm() + 1e-12))).item())


def run(dataset: str, cpu_scale: float, device: torch.device, profile: str,
        probe_mode: str = "permuted") -> dict:
    seed_everything(PRIMARY_SEED)
    cfg = SPLIT_CIFAR
    per = cfg.updates_per_source_per_norm
    probe_steps = cfg.probe_steps
    if profile == "cpu":
        per = max(3, int(round(per * cpu_scale)))
        probe_steps = max(30, int(round(probe_steps * cpu_scale)))

    if dataset == "cifar10":
        print(f"[E09-{dataset}] loading CIFAR-10 ...")
        tx, ty, vx, vy = load_cifar10(cfg.cifar10_dir)
        if probe_mode == "novel_class":
            # D-031: warmup only on the pre-registered warmup pairs; use held-out for updates
            warmup_pairs = NOVEL_CLASS_CIFAR.warmup_pairs
            held_out_pair = NOVEL_CLASS_CIFAR.held_out_pair
            tasks = split_cifar10_tasks(tx, ty, vx, vy, warmup_pairs)
            # Add held-out pair as the last "update-generation" task
            update_task = split_cifar10_tasks(tx, ty, vx, vy, (held_out_pair,))[0]
            tasks.append(update_task)
        else:
            tasks = split_cifar10_tasks(tx, ty, vx, vy, cfg.cifar10_class_pairs)
        n_out = 10
    elif dataset == "cifar100":
        print(f"[E09-{dataset}] loading CIFAR-100 ...")
        tx, ty, vx, vy = load_cifar100(cfg.cifar100_dir)
        tasks = split_cifar100_tasks(tx, ty, vx, vy, cfg.cifar100_task_size)
        n_out = int(max(ty.max(), vy.max())) + 1
    else:
        raise ValueError(dataset)

    print(f"[E09-{dataset}] tasks: {len(tasks)}  head: {n_out}-way  per={per}  probe_steps={probe_steps} probe_mode={probe_mode}")

    student = SmallCNN(in_channels=3, base=cfg.cnn_base_channels,
                       fc=cfg.cnn_hidden_fc, n_classes=n_out, input_hw=32).to(device)
    print(f"[E09-{dataset}] student params: {sum(p.numel() for p in student.parameters())}")

    n_warmup = len(tasks) - 1
    for t_idx in range(n_warmup):
        task = tasks[t_idx]
        n = task["train_x"].shape[0]
        steps = cfg.train_epochs_per_task * max(1, n // cfg.train_batch)
        train_task(student, task["train_x"], task["train_y"],
                   steps, cfg.train_lr, cfg.train_batch,
                   seed=PRIMARY_SEED + 17 * t_idx, device=device)
        tl = eval_loss(student, task["test_x"], task["test_y"], device=device)
        print(f"[E09-{dataset}] warmup task {t_idx}: test_loss={tl:.3f}")
    baseline = deepcopy(student)

    # Probes
    tx_all = torch.from_numpy(np.transpose(tx.astype(np.float32) / 255.0, (0, 3, 1, 2)))
    ty_all = torch.from_numpy(ty.astype(np.int64))
    if probe_mode == "permuted":
        probes = build_probes(tx_all, ty_all, cfg.n_probes,
                              seed=PRIMARY_SEED + 10_000, subset_size=cfg.probe_subset,
                              device=device)
    elif probe_mode == "novel_class":
        # Draw probes only from the held-out class pair, using original 10-way labels
        probes = build_novel_class_probes(
            tx_all, ty_all, held_out_classes=NOVEL_CLASS_CIFAR.held_out_pair,
            n_probes=NOVEL_CLASS_CIFAR.n_probes,
            seed=PRIMARY_SEED + 10_000, subset_size=NOVEL_CLASS_CIFAR.probe_subset,
        )
        # Use the pre-registered probe hyperparameters
        probe_steps = NOVEL_CLASS_CIFAR.probe_steps
    else:
        raise ValueError(f"unknown probe_mode: {probe_mode}")

    task_last = tasks[-1]
    xL, yL = task_last["train_x"], task_last["train_y"]
    norms = cfg.target_norms
    print(f"[E09-{dataset}] generating updates ({len(norms)} norms x 4 sources x {per})")

    a = gen_sgd_updates(baseline, xL, yL, per, norms, cfg.sgd_inner_steps,
                        cfg.sgd_inner_lr, cfg.sgd_inner_batch,
                        seed=PRIMARY_SEED + 501, device=device)
    b = gen_orth_updates(baseline, xL, yL, per, norms, cfg.sgd_inner_batch,
                         seed=PRIMARY_SEED + 502, device=device)
    c = gen_random_updates(baseline, xL, yL, per, norms, cfg.sgd_inner_batch,
                           seed=PRIMARY_SEED + 503, device=device)
    d = gen_convex_mix(baseline, xL, yL, per, norms, cfg.sgd_inner_steps,
                       cfg.sgd_inner_lr, cfg.sgd_inner_batch,
                       seed=PRIMARY_SEED + 504, device=device)
    updates = a + b + c + d
    print(f"[E09-{dataset}] total updates: {len(updates)}")

    g_ref = _reference_grad(baseline, xL, yL, cfg.sgd_inner_batch,
                            seed=PRIMARY_SEED + 991, device=device)
    ref_ftd = probe_ftd(baseline, probes, probe_steps, cfg.probe_lr, cfg.probe_batch,
                        seed=PRIMARY_SEED, device=device)
    print(f"[E09-{dataset}] reference FTD: {ref_ftd:.4f}")

    past_tasks = tasks[:-1]
    ref_past = [eval_loss(baseline, t["test_x"], t["test_y"], device=device)
                for t in past_tasks]

    theta_before_cpu = flat_params(baseline).detach().cpu()
    rows = []
    for uid, (source, tn, delta_cpu) in enumerate(updates):
        m = deepcopy(baseline)
        theta_after = (theta_before_cpu + delta_cpu).to(device)
        set_flat_params(m, theta_after)
        ftd_after = probe_ftd(m, probes, probe_steps, cfg.probe_lr, cfg.probe_batch,
                              seed=PRIMARY_SEED, device=device)
        ftd = ftd_after - ref_ftd
        after_past = [eval_loss(m, t["test_x"], t["test_y"], device=device)
                      for t in past_tasks]
        ptd = float(sum(a2 - r for a2, r in zip(after_past, ref_past)))
        # ctb approx: -g_ref . delta
        ctb = float(-(g_ref @ delta_cpu).item())
        rows.append({
            "update_id": uid, "source": source, "target_norm": tn,
            "mag": float(delta_cpu.norm().item()),
            "d_dir": cos(delta_cpu, g_ref),
            "ctb": ctb, "ptd": ptd, "ftd": ftd,
        })
        del m
        if (uid + 1) % 20 == 0:
            print(f"[E09-{dataset}] measured {uid + 1}/{len(updates)}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_TABLES, f"e09_{dataset}_measurements.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[E09-{dataset}] source summary:")
    print(df.groupby("source")[["mag", "ctb", "ptd", "d_dir", "ftd"]]
            .agg(["mean", "std"]).round(4).to_string())

    covariates = list(STATS.covariates)
    vif = compute_vif(df, covariates)
    print(f"\n[E09-{dataset}] VIF:", vif)

    full = fit_beta4_regression(df, covariates=covariates, outcome="ftd",
                                cov_type=STATS.ols_cov_type)
    print(f"[E09-{dataset}] full n={full.n_obs} R^2={full.r_squared:.4f} "
          f"beta_d_dir={full.coefs['d_dir']:.5f} "
          f"CI=[{full.ci_low['d_dir']:.5f},{full.ci_high['d_dir']:.5f}] "
          f"p={full.pvalues['d_dir']:.4g}")

    b_mean, b_lo, b_hi, sf = bootstrap_beta4(df, covariates=covariates, outcome="ftd",
                                             n_iter=STATS.bootstrap_iterations,
                                             seed=PRIMARY_SEED)
    print(f"[E09-{dataset}] bootstrap mean={b_mean:.5f} CI=[{b_lo:.5f},{b_hi:.5f}] "
          f"sign_flip={sf:.3f}")

    caliper_results = []
    for cal in STATS.ptd_calipers:
        matched = ptd_caliper_match(df, caliper=cal)
        wvb = within_vs_between_caliper_variance(matched, caliper=cal) \
            if len(matched) > 0 else float("nan")
        if len(matched) < 20:
            caliper_results.append({"caliper": float(cal), "n_matched": int(len(matched)),
                                    "beta_d_dir": None, "ci_low": None, "ci_high": None,
                                    "p": None, "within_over_total_var": float(wvb),
                                    "excludes_zero": False})
            continue
        r = fit_beta4_regression(matched, covariates=covariates, outcome="ftd",
                                 cov_type=STATS.ols_cov_type)
        excl = (r.ci_low["d_dir"] > 0) or (r.ci_high["d_dir"] < 0)
        caliper_results.append({
            "caliper": float(cal), "n_matched": int(len(matched)),
            "beta_d_dir": float(r.coefs["d_dir"]),
            "ci_low": float(r.ci_low["d_dir"]),
            "ci_high": float(r.ci_high["d_dir"]),
            "p": float(r.pvalues["d_dir"]),
            "within_over_total_var": float(wvb),
            "excludes_zero": bool(excl),
        })
        print(f"[E09-{dataset}] c={cal}: n={len(matched)} "
              f"beta={r.coefs['d_dir']:.5f} "
              f"CI=[{r.ci_low['d_dir']:.5f},{r.ci_high['d_dir']:.5f}] "
              f"p={r.pvalues['d_dir']:.4g} excludes0={excl} wvb={wvb:.3f}")

    betas = [c["beta_d_dir"] for c in caliper_results if c.get("beta_d_dir") is not None]
    sign_consistent = (all(b > 0 for b in betas) or all(b < 0 for b in betas)) if betas else False
    excludes_all = all(c.get("excludes_zero") for c in caliper_results
                       if c.get("beta_d_dir") is not None)
    ctrl_c_ok = all((c.get("within_over_total_var") or 1.0) <= KILL.within_caliper_ratio_max
                    for c in caliper_results if c.get("beta_d_dir") is not None)
    max_vif = max(vif.values())
    verdict = ("PASS" if (excludes_all and sign_consistent and ctrl_c_ok
                          and max_vif <= KILL.max_vif)
               else "PASS_with_vif_caveat" if (excludes_all and sign_consistent and ctrl_c_ok)
               else "PARTIAL" if excludes_all else "NULL")

    fig, ax = plt.subplots(figsize=(6, 4))
    for src, sub in df.groupby("source"):
        ax.scatter(sub["d_dir"], sub["ftd"], alpha=0.6, label=src, s=18)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("d_dir (cos with reference gradient)")
    ax.set_ylabel("FTD")
    tag = f"{dataset}" if probe_mode == "permuted" else f"{dataset}_{probe_mode}"
    ax.set_title(f"E09 - Split-{dataset.upper()} ({probe_mode}): FTD vs d_dir")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_FIGS, f"e09_{tag}_ftd_vs_ddir.png"), dpi=140)
    plt.close(fig)

    summary = {
        "dataset": dataset,
        "probe_mode": probe_mode,
        "hardware": env_summary(),
        "n_updates": int(len(df)),
        "sources": sorted(df["source"].unique().tolist()),
        "vif": {k: float(v) for k, v in vif.items()},
        "full_regression": {"coefs": full.coefs, "ci_low": full.ci_low,
                            "ci_high": full.ci_high, "p": full.pvalues,
                            "r_squared": full.r_squared},
        "bootstrap": {"mean": b_mean, "ci_low": b_lo, "ci_high": b_hi,
                      "sign_flip_fraction": sf},
        "control_c_sweep": caliper_results,
        "sign_consistent_across_calipers": bool(sign_consistent),
        "max_vif": float(max_vif),
        "kill_conditions": {
            "kill1_vif_passed": bool(max_vif <= KILL.max_vif),
            "kill2_excludes_all": bool(excludes_all),
            "kill3_control_c_passed": bool(ctrl_c_ok),
            "kill4_sign_consistent": bool(sign_consistent),
        },
        "verdict": verdict,
    }
    df.to_csv(os.path.join(RESULTS_TABLES, f"e09_{tag}_measurements.csv"), index=False)
    out_json = os.path.join(RESULTS_TABLES, f"e09_{tag}_summary.json")
    with open(out_json, "w") as f:
        json.dump(_stringify(summary), f, indent=2)
    print(f"[E09-{tag}] VERDICT: {verdict}")
    print(f"[E09-{tag}] wrote {out_json}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10", "cifar100", "both"], default="both")
    p.add_argument("--probe", choices=["permuted", "novel_class"], default="permuted")
    p.add_argument("--cpu-scale", type=float, default=SPLIT_CIFAR.cpu_scale_updates)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device, profile = pick_device(args.device)
    print(f"[E09] device={device} profile={profile} env={env_summary()}")
    if args.dataset in ("cifar10", "both"):
        run("cifar10", cpu_scale=args.cpu_scale, device=device, profile=profile,
            probe_mode=args.probe)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if args.dataset in ("cifar100", "both"):
        run("cifar100", cpu_scale=args.cpu_scale, device=device, profile=profile,
            probe_mode=args.probe)


if __name__ == "__main__":
    main()
