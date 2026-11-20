"""E04 — Split-MNIST replication (kill #5 gate).

Replicate the Candidate B protocol on Split-MNIST: 5 tasks of 2 classes each,
sequential training, generate matched-‖Δθ‖ updates at the pre-registered late
checkpoint, measure FTD on held-out probe tasks (constructed as class-flipped
MNIST or, better, novel-digit-pair probes drawn from *unseen* digits).

Since Split-MNIST uses all ten digits, we can't hold out digits. We instead
use the *permuted-MNIST* probe (a random per-pixel permutation) as a
structurally-novel probe task — the standard technique in continual-learning
literature for measuring plasticity.
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
    PRIMARY_SEED, SPLIT_MNIST, UPDATE_GEN, STATS, KILL, PROBE,
)
from src.seeds import seed_everything
from src.models import MLP, flat_params, set_flat_params, flat_grad
from src.datasets import load_mnist, split_mnist_tasks
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


# -------- Task-level helpers --------

def task_iter(x: torch.Tensor, y: torch.Tensor, batch: int, generator):
    n = x.shape[0]
    perm = torch.randperm(n, generator=generator)
    for i in range(0, n, batch):
        idx = perm[i:i + batch]
        yield x[idx], y[idx]


def task_loss(model, x, y):
    return F.cross_entropy(model(x), y)


def eval_loss(model, x, y):
    with torch.no_grad():
        return float(F.cross_entropy(model(x), y).item())


def train_task(model, x, y, steps, lr, batch, seed):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    step = 0
    losses = []
    while step < steps:
        for xb, yb in task_iter(x, y, batch, g):
            opt.zero_grad(set_to_none=True)
            loss = task_loss(model, xb, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            step += 1
            if step >= steps:
                break
    return losses


def build_probes(train_x, train_y, n_probes: int, seed: int, subset_size: int = 4000):
    """Build permuted-MNIST probes: 10-way classification but pixels permuted."""
    probes = []
    rng = np.random.default_rng(seed)
    for p_id in range(n_probes):
        perm = rng.permutation(train_x.shape[1])
        idx = rng.choice(train_x.shape[0], size=subset_size, replace=False)
        x = train_x[idx][:, perm]
        y = train_y[idx].long()
        probes.append((x, y))
    return probes


def probe_ftd(model, probes, steps, lr, batch, seed):
    aucs = []
    for p_id, (x, y) in enumerate(probes):
        m = deepcopy(model)
        opt = torch.optim.SGD(m.parameters(), lr=lr)
        g = torch.Generator().manual_seed(seed + 7919 * (p_id + 1))
        losses = []
        step = 0
        while step < steps:
            for xb, yb in task_iter(x, y, batch, g):
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(m(xb), yb)
                loss.backward()
                opt.step()
                losses.append(loss.item())
                step += 1
                if step >= steps:
                    break
        aucs.append(float(sum(losses) / len(losses)))
    return float(sum(aucs) / len(aucs))


# -------- MNIST-flavored student --------

class MNISTStudent(nn.Module):
    def __init__(self, input_dim=784, hidden=128, depth=2, n_out=10):
        super().__init__()
        layers = []
        prev = input_dim
        for _ in range(depth):
            layers += [nn.Linear(prev, hidden), nn.ReLU()]
            prev = hidden
        layers += [nn.Linear(prev, n_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -------- Matched-|delta| update sources for MNIST --------

def gen_sgd_updates(base, x, y, n, target_norms, steps, lr, batch, seed):
    from src.update_generator import UpdateSample, _renormalize
    theta_before = flat_params(base)
    out = []
    for k, norm in enumerate(target_norms):
        for s in range(n):
            m = deepcopy(base)
            g = torch.Generator().manual_seed(seed + 101 * k + s)
            losses = train_task(m, x, y, steps, lr, batch, seed + 101 * k + s)
            delta = flat_params(m) - theta_before
            delta = _renormalize(delta, norm)
            ctb = float(losses[0] - losses[-1]) if len(losses) >= 2 else 0.0
            out.append(UpdateSample("sgd_seed", norm, delta, ctb, theta_before))
    return out


def _reference_gradient_mnist(base, x, y, batch, seed):
    from src.models import flat_grad
    m = deepcopy(base)
    g = torch.Generator().manual_seed(seed)
    xb, yb = next(task_iter(x, y, batch, g))
    m.zero_grad(set_to_none=True)
    F.cross_entropy(m(xb), yb).backward()
    return flat_grad(m)


def gen_orth_updates(base, x, y, n, target_norms, batch, seed, orth_std=1.0):
    from src.update_generator import UpdateSample, _renormalize
    theta_before = flat_params(base)
    g_ref = _reference_gradient_mnist(base, x, y, batch, seed + 991)
    g_hat = g_ref / (g_ref.norm() + 1e-12)
    D = g_ref.numel()
    out = []
    for k, norm in enumerate(target_norms):
        for s in range(n):
            g = torch.Generator().manual_seed(seed + 313 * k + 7 * s + 1)
            eps = torch.empty(D).normal_(std=orth_std, generator=g)
            d_perp = eps - (eps @ g_hat) * g_hat
            delta = _renormalize(d_perp, norm)
            ctb_approx = float(-(g_ref @ delta).item())
            out.append(UpdateSample("orth_perturb", norm, delta, ctb_approx, theta_before))
    return out


def gen_random_sphere_updates(base, x, y, n, target_norms, batch, seed):
    from src.update_generator import UpdateSample, _renormalize
    theta_before = flat_params(base)
    g_ref = _reference_gradient_mnist(base, x, y, batch, seed + 991)
    D = g_ref.numel()
    out = []
    for k, norm in enumerate(target_norms):
        for s in range(n):
            g = torch.Generator().manual_seed(seed + 991 * k + 13 * s + 3)
            v = torch.empty(D).normal_(generator=g)
            delta = _renormalize(v, norm)
            ctb_approx = float(-(g_ref @ delta).item())
            out.append(UpdateSample("random_sphere", norm, delta, ctb_approx, theta_before))
    return out


def gen_convex_mix_updates(base, x, y, n, target_norms, steps, lr, batch, seed):
    from src.update_generator import UpdateSample, _renormalize
    theta_before = flat_params(base)
    g_ref = _reference_gradient_mnist(base, x, y, batch, seed + 991)
    D = g_ref.numel()
    n_pool = max(n, 5)
    sgd_pool = []
    for i in range(n_pool):
        m = deepcopy(base)
        train_task(m, x, y, steps, lr, batch, seed + 707 * i + 1)
        d = flat_params(m) - theta_before
        sgd_pool.append(d / (d.norm() + 1e-12))
    rand_pool = []
    for i in range(n_pool):
        g = torch.Generator().manual_seed(seed + 1213 * i + 2)
        v = torch.empty(D).normal_(generator=g)
        rand_pool.append(v / (v.norm() + 1e-12))
    alphas = torch.linspace(0.05, 0.95, 5).tolist()
    out = []
    for k, norm in enumerate(target_norms):
        for s in range(n):
            alpha = alphas[s % len(alphas)]
            u_sgd = sgd_pool[s % n_pool]
            u_rand = rand_pool[(s + k) % n_pool]
            mixed = alpha * u_sgd + (1 - alpha) * u_rand
            delta = _renormalize(mixed, norm)
            ctb_approx = float(-(g_ref @ delta).item())
            out.append(UpdateSample("convex_mix", norm, delta, ctb_approx, theta_before))
    return out


def cos(a, b):
    return float((a @ b / ((a.norm() + 1e-12) * (b.norm() + 1e-12))).item())


def _stringify(obj):
    if isinstance(obj, dict): return {k: _stringify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_stringify(v) for v in obj]
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.integer): return int(obj)
    return obj


def main() -> None:
    seed_everything(PRIMARY_SEED)

    tx, ty, vx, vy = load_mnist(SPLIT_MNIST.mnist_dir)
    tasks = split_mnist_tasks(tx, ty, vx, vy, SPLIT_MNIST.task_class_pairs)
    # Remap task-y from {0,1} pair to 10-way (using original digit as class)
    # so a single 10-way head is used across tasks and probes.
    # We need original y — reload full labels
    for i, task in enumerate(tasks):
        pair = task["pair"]
        # rebuild train_y/test_y with the original digit labels
        mask_tr = np.isin(ty, pair)
        mask_te = np.isin(vy, pair)
        task["train_y"] = torch.from_numpy(ty[mask_tr].astype(np.int64))
        task["test_y"] = torch.from_numpy(vy[mask_te].astype(np.int64))
    print(f"[E04] Split-MNIST: {len(tasks)} tasks (10-way head)")

    # Build student
    student = MNISTStudent(
        input_dim=784,
        hidden=SPLIT_MNIST.student_hidden_width,
        depth=SPLIT_MNIST.student_hidden_depth,
        n_out=10,
    )
    print(f"[E04] student params: {sum(p.numel() for p in student.parameters())}")

    # Sequential training on tasks 0..3 (leave task 4 for the update generator)
    for t_idx in range(4):
        task = tasks[t_idx]
        # steps per task = epochs * (N / batch)
        n = task["train_x"].shape[0]
        steps = SPLIT_MNIST.train_epochs_per_task * max(1, n // SPLIT_MNIST.train_batch)
        train_task(student, task["train_x"], task["train_y"],
                   steps, SPLIT_MNIST.train_lr, SPLIT_MNIST.train_batch,
                   seed=PRIMARY_SEED + 17 * t_idx)
        te_loss = eval_loss(student, task["test_x"], task["test_y"])
        print(f"[E04] warmup task {t_idx} pair={task['pair']}: test loss={te_loss:.3f}")
    baseline = deepcopy(student)

    # Build permuted-MNIST probes (novel structure)
    tx_all_norm = torch.from_numpy(tx.reshape(len(tx), -1).astype(np.float32) / 255.0)
    ty_all = torch.from_numpy(ty.astype(np.int64))
    probes = build_probes(tx_all_norm, ty_all, n_probes=PROBE.n_probes,
                          seed=PRIMARY_SEED + 10_000, subset_size=3000)

    # Generate matched updates on task 4
    task4 = tasks[4]
    x4, y4 = task4["train_x"], task4["train_y"]
    per = 20  # smaller than synthetic for CPU time
    norms = UPDATE_GEN.target_norms
    print(f"[E04] generating updates on task 4 pair={task4['pair']} "
          f"({len(norms)} norms x {per}/source x 4 sources)")

    a = gen_sgd_updates(baseline, x4, y4, per, norms, steps=UPDATE_GEN.sgd_inner_steps,
                        lr=UPDATE_GEN.sgd_inner_lr, batch=UPDATE_GEN.sgd_inner_batch,
                        seed=PRIMARY_SEED + 501)
    b = gen_orth_updates(baseline, x4, y4, per, norms, batch=UPDATE_GEN.sgd_inner_batch,
                         seed=PRIMARY_SEED + 502)
    c = gen_random_sphere_updates(baseline, x4, y4, per, norms,
                                  batch=UPDATE_GEN.sgd_inner_batch,
                                  seed=PRIMARY_SEED + 503)
    d = gen_convex_mix_updates(baseline, x4, y4, per, norms, steps=UPDATE_GEN.sgd_inner_steps,
                               lr=UPDATE_GEN.sgd_inner_lr, batch=UPDATE_GEN.sgd_inner_batch,
                               seed=PRIMARY_SEED + 504)
    updates = a + b + c + d
    print(f"[E04] total updates: {len(updates)}")

    g_ref = _reference_gradient_mnist(baseline, x4, y4, UPDATE_GEN.sgd_inner_batch,
                                      seed=PRIMARY_SEED + 991)
    # Reference FTD before update
    ref_ftd = probe_ftd(baseline, probes, PROBE.probe_steps, PROBE.probe_lr,
                        PROBE.probe_batch, seed=PRIMARY_SEED)
    print(f"[E04] reference FTD (baseline): {ref_ftd:.4f}")

    # Reference past losses for PTD
    past_tasks = tasks[:4]
    ref_past = [eval_loss(baseline, t["test_x"], t["test_y"]) for t in past_tasks]

    # Measure per-update FTD, PTD, D_dir
    theta_before = flat_params(baseline)
    rows = []
    for uid, u in enumerate(updates):
        m = deepcopy(baseline)
        set_flat_params(m, theta_before + u.delta)
        ftd_after = probe_ftd(m, probes, PROBE.probe_steps, PROBE.probe_lr,
                              PROBE.probe_batch, seed=PRIMARY_SEED)
        ftd = ftd_after - ref_ftd
        after_past = [eval_loss(m, t["test_x"], t["test_y"]) for t in past_tasks]
        ptd = float(sum(a - r for a, r in zip(after_past, ref_past)))
        rows.append({
            "update_id": uid,
            "source": u.source,
            "target_norm": u.target_norm,
            "mag": float(u.delta.norm().item()),
            "d_dir": cos(u.delta, g_ref),
            "ctb": float(u.ctb),
            "ptd": ptd,
            "ftd": ftd,
        })
        if (uid + 1) % 40 == 0:
            print(f"[E04] measured {uid + 1}/{len(updates)} updates")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_TABLES, "e04_measurements.csv"), index=False)
    covariates = list(STATS.covariates)

    print(f"\n[E04] Source summary:")
    print(df.groupby("source")[["mag", "ctb", "ptd", "d_dir", "ftd"]]
            .agg(["mean", "std"]).round(4).to_string())

    vif = compute_vif(df, covariates)
    print(f"\n[E04] VIF:")
    for k, v in vif.items():
        flag = "  <-- OVER" if v > STATS.vif_threshold else ""
        print(f"  {k}: {v:.3f}{flag}")

    full = fit_beta4_regression(df, covariates=covariates, outcome="ftd",
                                cov_type=STATS.ols_cov_type)
    print(f"\n[E04] Full OLS n={full.n_obs}  R^2={full.r_squared:.4f}")
    print(f"       beta_d_dir = {full.coefs['d_dir']:.5f}  "
          f"CI=[{full.ci_low['d_dir']:.5f},{full.ci_high['d_dir']:.5f}]  "
          f"p={full.pvalues['d_dir']:.4g}")

    b_mean, b_lo, b_hi, sf = bootstrap_beta4(df, covariates=covariates, outcome="ftd",
                                             n_iter=STATS.bootstrap_iterations,
                                             seed=PRIMARY_SEED)
    print(f"[E04] Bootstrap beta_d_dir mean={b_mean:.5f} "
          f"CI=[{b_lo:.5f},{b_hi:.5f}] sign_flip={sf:.3f}")

    caliper_results = []
    for cal in STATS.ptd_calipers:
        matched = ptd_caliper_match(df, caliper=cal)
        wvb = within_vs_between_caliper_variance(matched, caliper=cal) \
            if len(matched) > 0 else float("nan")
        if len(matched) < 20:
            caliper_results.append({
                "caliper": float(cal), "n_matched": int(len(matched)),
                "beta_d_dir": None, "ci_low": None, "ci_high": None,
                "p": None, "within_over_total_var": float(wvb),
                "excludes_zero": False,
            })
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
        print(f"[E04] c={cal}: n={len(matched)} beta={r.coefs['d_dir']:.5f} "
              f"CI=[{r.ci_low['d_dir']:.5f},{r.ci_high['d_dir']:.5f}] "
              f"p={r.pvalues['d_dir']:.4g} excludes0={excl} wvb={wvb:.3f}")

    betas = [c["beta_d_dir"] for c in caliper_results if c.get("beta_d_dir") is not None]
    sign_consistent = (all(b > 0 for b in betas) or all(b < 0 for b in betas)) if betas else False
    excludes_all = all(c.get("excludes_zero") for c in caliper_results
                       if c.get("beta_d_dir") is not None)
    ctrl_c_ok = all(
        (c.get("within_over_total_var") or 1.0) <= KILL.within_caliper_ratio_max
        for c in caliper_results if c.get("beta_d_dir") is not None
    )
    verdict = ("PASS" if (excludes_all and sign_consistent and ctrl_c_ok)
               else "PARTIAL" if excludes_all else "NULL")
    print(f"\n[E04] Verdict: {verdict}  "
          f"excludes_all={excludes_all}  sign_consistent={sign_consistent}  ctrl_c={ctrl_c_ok}")

    # Plots
    fig, ax = plt.subplots(figsize=(6, 4))
    for src, sub in df.groupby("source"):
        ax.scatter(sub["d_dir"], sub["ftd"], alpha=0.6, label=src, s=18)
    ax.set_xlabel("d_dir (cos with reference gradient)")
    ax.set_ylabel("FTD (probe learning-speed damage)")
    ax.set_title("E04 - Split-MNIST replication: FTD vs d_dir")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_FIGS, "e04_ftd_vs_ddir.png"), dpi=140)
    plt.close(fig)

    summary = {
        "n_updates": int(len(df)),
        "sources": sorted(df["source"].unique().tolist()),
        "vif": {k: float(v) for k, v in vif.items()},
        "full_regression": {
            "coefs": full.coefs, "ci_low": full.ci_low, "ci_high": full.ci_high,
            "p": full.pvalues, "r_squared": full.r_squared,
        },
        "bootstrap": {"mean": b_mean, "ci_low": b_lo, "ci_high": b_hi,
                      "sign_flip_fraction": sf},
        "control_c_sweep": caliper_results,
        "sign_consistent_across_calipers": bool(sign_consistent),
        "verdict": verdict,
    }
    summary = _stringify(summary)
    with open(os.path.join(RESULTS_TABLES, "e04_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[E04] wrote {os.path.join(RESULTS_TABLES, 'e04_summary.json')}")


if __name__ == "__main__":
    main()
