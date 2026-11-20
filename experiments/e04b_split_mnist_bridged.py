"""E04b - Split-MNIST with the convex-mix bridging source (Phase 2, D-013+D-024).

Motivation: E04's beta_d_dir was significant, sign-stable across 5 seeds (E06/N2),
BUT the full-sample VIF for ctb (7.63) and d_dir (6.46) was over the pre-registered
5.0 gate. This experiment repeats E04 with the convex_mix source added (already
validated on synthetic in E03b) to bring the two problematic VIFs into range and
test whether the significant beta_d_dir survives a clean identifiability gate.

This is a post-hoc, labeled robustness run (per D-023) - not a re-write of E04.
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
from src.models import flat_params, set_flat_params, flat_grad
from src.device import pick_device, env_summary
from src.datasets import load_mnist, split_mnist_tasks
from src.stats import (
    compute_vif, fit_beta4_regression, bootstrap_beta4, ptd_caliper_match,
    within_vs_between_caliper_variance,
)
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


def main() -> None:
    device, profile = pick_device("auto")
    print(f"[E04b] device={device} profile={profile}")
    seed_everything(PRIMARY_SEED)

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
    # warm through tasks 0..3
    for t_idx in range(4):
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

    tx_norm = torch.from_numpy(tx.reshape(len(tx), -1).astype(np.float32) / 255.0)
    ty_all = torch.from_numpy(ty.astype(np.int64))
    probes = build_probes(tx_norm, ty_all, PROBE.n_probes,
                          seed=PRIMARY_SEED + 10_000, subset_size=3000)

    task4 = tasks[4]
    x4, y4 = task4["train_x"], task4["train_y"]
    # DELIBERATELY use MORE per source in E04b so the added convex_mix has
    # enough sample count to bring VIF down under matching (still small enough
    # to run in ~20 min on CPU or ~3 min on GPU-CPU-transfer for this MLP).
    per = 30
    norms = UPDATE_GEN.target_norms
    print(f"[E04b] generating {4 * 4 * per} updates ({per}/source)")
    a = gen_sgd_updates(baseline, x4, y4, per, norms, UPDATE_GEN.sgd_inner_steps,
                        UPDATE_GEN.sgd_inner_lr, UPDATE_GEN.sgd_inner_batch,
                        seed=PRIMARY_SEED + 501)
    b = gen_orth_updates(baseline, x4, y4, per, norms, UPDATE_GEN.sgd_inner_batch,
                         seed=PRIMARY_SEED + 502)
    c = gen_random_sphere_updates(baseline, x4, y4, per, norms,
                                  UPDATE_GEN.sgd_inner_batch, seed=PRIMARY_SEED + 503)
    d = gen_convex_mix_updates(baseline, x4, y4, per, norms, UPDATE_GEN.sgd_inner_steps,
                               UPDATE_GEN.sgd_inner_lr, UPDATE_GEN.sgd_inner_batch,
                               seed=PRIMARY_SEED + 504)
    updates = a + b + c + d
    print(f"[E04b] total updates: {len(updates)}")

    g_ref = _reference_gradient_mnist(baseline, x4, y4,
                                      UPDATE_GEN.sgd_inner_batch, seed=PRIMARY_SEED + 991)
    ref_ftd = probe_ftd(baseline, probes, PROBE.probe_steps, PROBE.probe_lr,
                        PROBE.probe_batch, seed=PRIMARY_SEED)
    past_tasks = tasks[:4]
    ref_past = [eval_loss(baseline, t["test_x"], t["test_y"]) for t in past_tasks]
    theta_before = flat_params(baseline)

    rows = []
    for uid, u in enumerate(updates):
        m = deepcopy(baseline)
        set_flat_params(m, theta_before + u.delta)
        ftd_after = probe_ftd(m, probes, PROBE.probe_steps, PROBE.probe_lr,
                              PROBE.probe_batch, seed=PRIMARY_SEED)
        ftd = ftd_after - ref_ftd
        after_past = [eval_loss(m, t["test_x"], t["test_y"]) for t in past_tasks]
        ptd = float(sum(a2 - r for a2, r in zip(after_past, ref_past)))
        rows.append({
            "update_id": uid, "source": u.source, "target_norm": u.target_norm,
            "mag": float(u.delta.norm().item()),
            "d_dir": cos(u.delta, g_ref),
            "ctb": float(u.ctb), "ptd": ptd, "ftd": ftd,
        })
        if (uid + 1) % 60 == 0:
            print(f"[E04b] measured {uid + 1}/{len(updates)}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_TABLES, "e04b_measurements.csv"), index=False)
    print(f"\n[E04b] source summary:")
    print(df.groupby("source")[["mag", "ctb", "ptd", "d_dir", "ftd"]]
          .agg(["mean", "std"]).round(4).to_string())

    covariates = list(STATS.covariates)
    vif = compute_vif(df, covariates)
    print(f"\n[E04b] VIF (gate = {STATS.vif_threshold}):")
    for k, v in vif.items():
        print(f"  {k}: {v:.3f}{'  <-- OVER' if v > STATS.vif_threshold else ''}")

    # PER-SOURCE and PER-CALIPER VIF checks — the paper reviewer's question
    # 'is the coefficient identifiable in the matched subset?'
    per_caliper_vif = {}
    for cal in STATS.ptd_calipers:
        matched = ptd_caliper_match(df, caliper=cal)
        if len(matched) < 40: continue
        per_caliper_vif[cal] = compute_vif(matched, covariates)

    full = fit_beta4_regression(df, covariates=covariates, outcome="ftd",
                                cov_type=STATS.ols_cov_type)
    print(f"\n[E04b] full OLS n={full.n_obs} R^2={full.r_squared:.4f} "
          f"beta_d_dir={full.coefs['d_dir']:.5f} "
          f"CI=[{full.ci_low['d_dir']:.5f},{full.ci_high['d_dir']:.5f}] "
          f"p={full.pvalues['d_dir']:.4g}")

    b_mean, b_lo, b_hi, sf = bootstrap_beta4(df, covariates=covariates, outcome="ftd",
                                             n_iter=STATS.bootstrap_iterations,
                                             seed=PRIMARY_SEED)
    print(f"[E04b] bootstrap mean={b_mean:.5f} CI=[{b_lo:.5f},{b_hi:.5f}] "
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
        v_cal = per_caliper_vif.get(cal, {})
        caliper_results.append({
            "caliper": float(cal), "n_matched": int(len(matched)),
            "beta_d_dir": float(r.coefs["d_dir"]),
            "ci_low": float(r.ci_low["d_dir"]),
            "ci_high": float(r.ci_high["d_dir"]),
            "p": float(r.pvalues["d_dir"]),
            "within_over_total_var": float(wvb),
            "excludes_zero": bool(excl),
            "vif_matched": {k: float(vv) for k, vv in v_cal.items()},
        })
        print(f"[E04b] c={cal}: n={len(matched)} "
              f"beta={r.coefs['d_dir']:.5f} "
              f"CI=[{r.ci_low['d_dir']:.5f},{r.ci_high['d_dir']:.5f}] "
              f"p={r.pvalues['d_dir']:.4g} excludes0={excl} wvb={wvb:.3f} "
              f"VIF_matched={{k:round(v,2) for k,v in v_cal.items()}}")

    betas = [c["beta_d_dir"] for c in caliper_results if c.get("beta_d_dir") is not None]
    sign_consistent = (all(b > 0 for b in betas) or all(b < 0 for b in betas)) if betas else False
    excludes_all = all(c.get("excludes_zero") for c in caliper_results
                       if c.get("beta_d_dir") is not None)
    ctrl_c_ok = all((c.get("within_over_total_var") or 1.0) <= KILL.within_caliper_ratio_max
                    for c in caliper_results if c.get("beta_d_dir") is not None)
    # NEW: caliper-matched VIF gate
    matched_vif_ok = all(
        max(c.get("vif_matched", {}).values(), default=0) <= KILL.max_vif
        for c in caliper_results if c.get("beta_d_dir") is not None
    )
    verdict = ("PASS" if (excludes_all and sign_consistent and ctrl_c_ok and matched_vif_ok)
               else "PASS_matched_only" if (excludes_all and sign_consistent and ctrl_c_ok)
               else "PARTIAL" if excludes_all else "NULL")

    fig, ax = plt.subplots(figsize=(6, 4))
    for src, sub in df.groupby("source"):
        ax.scatter(sub["d_dir"], sub["ftd"], alpha=0.6, label=src, s=18)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("d_dir")
    ax.set_ylabel("FTD")
    ax.set_title("E04b - Split-MNIST bridged (convex_mix added)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_FIGS, "e04b_ftd_vs_ddir.png"), dpi=140)
    plt.close(fig)

    summary = {
        "hardware": env_summary(),
        "n_updates": int(len(df)),
        "sources": sorted(df["source"].unique().tolist()),
        "vif_full": {k: float(v) for k, v in vif.items()},
        "vif_per_caliper": {str(k): {kk: float(vv) for kk, vv in v.items()}
                            for k, v in per_caliper_vif.items()},
        "full_regression": {"coefs": full.coefs, "ci_low": full.ci_low,
                            "ci_high": full.ci_high, "p": full.pvalues,
                            "r_squared": full.r_squared},
        "bootstrap": {"mean": b_mean, "ci_low": b_lo, "ci_high": b_hi,
                      "sign_flip_fraction": sf},
        "control_c_sweep": caliper_results,
        "sign_consistent_across_calipers": bool(sign_consistent),
        "kill_conditions": {
            "kill1_full_vif_passed": bool(max(vif.values()) <= KILL.max_vif),
            "kill1_matched_vif_passed": bool(matched_vif_ok),
            "kill2_excludes_all": bool(excludes_all),
            "kill3_control_c_passed": bool(ctrl_c_ok),
            "kill4_sign_consistent": bool(sign_consistent),
        },
        "verdict": verdict,
    }
    out_json = os.path.join(RESULTS_TABLES, "e04b_summary.json")
    with open(out_json, "w") as f:
        json.dump(_stringify(summary), f, indent=2)
    print(f"\n[E04b] VERDICT: {verdict}")
    print(f"[E04b] wrote {out_json}")


if __name__ == "__main__":
    main()
