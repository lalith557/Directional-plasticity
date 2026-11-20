"""E03b — Robustness re-run of Candidate B with convex-mixture generator added.

Motivation: E03's pre-registered result was PASS (all caliper CIs excluded 0)
but had a sign flip at caliper=0.05 (see result.md R03 + decision.md D-013).
That flip is consistent with SGD-source and random-source PTD ranges being
disjoint, so wide-caliper matching hardly matches anything.

Adding a convex-mixture source (interpolating SGD and random unit vectors and
renormalizing) gives updates whose d_dir and PTD *continuously* span the space
between the two extremes — making within-caliper matching meaningful and
letting us tell whether the sign flip was an artifact of the missing bridge.
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
from copy import deepcopy

from configs.preregistration import (
    PRIMARY_SEED, TEACHER_STUDENT, PROBE, UPDATE_GEN, STATS, KILL,
)
from src.seeds import seed_everything
from src.models import MLP, flat_params
from src.teacher_student import build_teacher, make_task_stream, build_probe_tasks
from src.training import train_task_and_capture_delta
from src.update_generator import generate_all_updates, _reference_gradient
from src.metrics import build_measurement_frame, measurements_to_dataframe
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


def _sequential_warmup(student, train_by_task, n_warmup_tasks, lr, steps, batch, seed):
    for t in range(n_warmup_tasks):
        g = torch.Generator().manual_seed(seed + 17 * t)
        train_task_and_capture_delta(
            student, train_by_task[t], n_steps=steps, lr=lr,
            mini_size=batch, generator=g,
        )
    return student


def _stringify(obj):
    if isinstance(obj, dict):
        return {k: _stringify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_stringify(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def main() -> None:
    seed_everything(PRIMARY_SEED)
    teacher = build_teacher(TEACHER_STUDENT, seed=PRIMARY_SEED)
    train_by_task, eval_by_task = make_task_stream(
        teacher, TEACHER_STUDENT, seed=PRIMARY_SEED,
    )
    probes = build_probe_tasks(teacher, TEACHER_STUDENT, PROBE, seed=PRIMARY_SEED)
    student = MLP(
        input_dim=TEACHER_STUDENT.input_dim,
        output_dim=TEACHER_STUDENT.output_dim,
        hidden_width=TEACHER_STUDENT.hidden_width,
        hidden_depth=TEACHER_STUDENT.hidden_depth,
        activation=TEACHER_STUDENT.activation,
    )
    student = _sequential_warmup(
        student, train_by_task, n_warmup_tasks=TEACHER_STUDENT.n_tasks - 1,
        lr=5e-3, steps=300, batch=128, seed=PRIMARY_SEED,
    )
    baseline = deepcopy(student)

    updates = generate_all_updates(
        baseline, train_by_task[TEACHER_STUDENT.n_tasks - 1], UPDATE_GEN,
        seed=PRIMARY_SEED + 500, include_convex_mix=True,
    )
    print(f"[E03b] generated {len(updates)} updates (including convex_mix)")

    g_ref = _reference_gradient(
        baseline, train_by_task[TEACHER_STUDENT.n_tasks - 1],
        UPDATE_GEN.sgd_inner_batch, seed=PRIMARY_SEED + 991,
    )
    g_task = _reference_gradient(
        baseline, train_by_task[TEACHER_STUDENT.n_tasks - 1],
        UPDATE_GEN.sgd_inner_batch, seed=PRIMARY_SEED + 12345,
    )
    random_ref = torch.randn_like(flat_params(baseline))
    past = list(eval_by_task[:TEACHER_STUDENT.n_tasks - 1])
    frame = build_measurement_frame(
        baseline, updates, probes, past,
        reference_gradient=g_ref, task_gradient=g_task, random_reference=random_ref,
        probe_cfg=PROBE, probe_seed=PRIMARY_SEED,
    )
    df = measurements_to_dataframe(frame).rename(columns={"delta_norm": "mag"})
    df.to_csv(os.path.join(RESULTS_TABLES, "e03b_measurements.csv"), index=False)

    covariates = list(STATS.covariates)
    print(f"\n[E03b] Source summary:")
    print(df.groupby("source")[["mag", "ctb", "ptd", "d_dir", "ftd"]]
            .agg(["mean", "std"]).round(4).to_string())

    vif = compute_vif(df, covariates)
    print(f"\n[E03b] VIF (threshold={STATS.vif_threshold}):")
    for k, v in vif.items():
        flag = "  <-- OVER" if v > STATS.vif_threshold else ""
        print(f"  {k}: {v:.3f}{flag}")

    full = fit_beta4_regression(
        df, covariates=covariates, outcome="ftd",
        cov_type=STATS.ols_cov_type, alpha=STATS.beta4_significance_alpha,
    )
    print(f"\n[E03b] Full OLS n={full.n_obs} R^2={full.r_squared:.4f}")
    print(f"       beta_d_dir = {full.coefs['d_dir']:.5f}  "
          f"CI=[{full.ci_low['d_dir']:.5f},{full.ci_high['d_dir']:.5f}]  "
          f"p={full.pvalues['d_dir']:.4g}")

    b_mean, b_lo, b_hi, sf = bootstrap_beta4(
        df, covariates=covariates, outcome="ftd",
        n_iter=STATS.bootstrap_iterations, alpha=STATS.bootstrap_alpha, seed=PRIMARY_SEED,
    )
    print(f"[E03b] Bootstrap beta_d_dir mean={b_mean:.5f} "
          f"CI=[{b_lo:.5f},{b_hi:.5f}] sign_flip={sf:.3f}")

    # Control C sweep with the wider source set
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
        r = fit_beta4_regression(
            matched, covariates=covariates, outcome="ftd",
            cov_type=STATS.ols_cov_type, alpha=STATS.beta4_significance_alpha,
        )
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
        print(f"[E03b] c={cal}: n={len(matched)} beta={r.coefs['d_dir']:.5f} "
              f"CI=[{r.ci_low['d_dir']:.5f},{r.ci_high['d_dir']:.5f}] "
              f"p={r.pvalues['d_dir']:.4g} excludes0={excl} "
              f"wvb={wvb:.3f}")

    # Consistent-sign robustness (stronger than pre-registered "excludes zero")
    betas = [c["beta_d_dir"] for c in caliper_results if c.get("beta_d_dir") is not None]
    sign_consistent = (all(b > 0 for b in betas) or all(b < 0 for b in betas))
    excludes_all = all(c.get("excludes_zero") for c in caliper_results
                       if c.get("beta_d_dir") is not None)
    ctrl_c_ok = all(
        (c.get("within_over_total_var") or 1.0) <= KILL.within_caliper_ratio_max
        for c in caliper_results if c.get("beta_d_dir") is not None
    )
    verdict = ("PASS" if (excludes_all and sign_consistent and ctrl_c_ok)
               else "PARTIAL" if excludes_all else "FAIL")
    print(f"\n[E03b] Verdict: {verdict}  "
          f"excludes_all={excludes_all} sign_consistent={sign_consistent} ctrl_c={ctrl_c_ok}")

    # Plots
    fig, ax = plt.subplots(figsize=(6, 4))
    for src, sub in df.groupby("source"):
        ax.scatter(sub["d_dir"], sub["ftd"], alpha=0.6, label=src, s=18)
    ax.set_xlabel("d_dir (cos with reference gradient)")
    ax.set_ylabel("FTD")
    ax.set_title("E03b - FTD vs d_dir with convex_mix bridging source")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_FIGS, "e03b_ftd_vs_ddir.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    calipers = [c["caliper"] for c in caliper_results if c.get("beta_d_dir") is not None]
    betas = [c["beta_d_dir"] for c in caliper_results if c.get("beta_d_dir") is not None]
    lows = [c["ci_low"] for c in caliper_results if c.get("beta_d_dir") is not None]
    highs = [c["ci_high"] for c in caliper_results if c.get("beta_d_dir") is not None]
    yerr = np.array([np.array(betas) - np.array(lows), np.array(highs) - np.array(betas)])
    ax.errorbar(range(len(calipers)), betas, yerr=yerr, fmt="o", capsize=4)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(range(len(calipers)))
    ax.set_xticklabels([f"c={c}" for c in calipers])
    ax.set_ylabel("beta_d_dir (95% CI)")
    ax.set_title("E03b - Control C sweep with convex_mix")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_FIGS, "e03b_control_c_sweep.png"), dpi=140)
    plt.close(fig)

    summary = {
        "n_updates": int(len(df)),
        "sources": sorted(df["source"].unique().tolist()),
        "vif": {k: float(v) for k, v in vif.items()},
        "full_regression": {
            "coefs": full.coefs, "ci_low": full.ci_low, "ci_high": full.ci_high,
            "p": full.pvalues, "r_squared": full.r_squared,
        },
        "bootstrap": {
            "mean": b_mean, "ci_low": b_lo, "ci_high": b_hi,
            "sign_flip_fraction": sf,
        },
        "control_c_sweep": caliper_results,
        "sign_consistent_across_calipers": bool(sign_consistent),
        "verdict": verdict,
    }
    summary = _stringify(summary)
    with open(os.path.join(RESULTS_TABLES, "e03b_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[E03b] wrote {os.path.join(RESULTS_TABLES, 'e03b_summary.json')}")


if __name__ == "__main__":
    main()
