"""E03 — Candidate B (decisive discovery experiment on synthetic data).

Q: After controlling for magnitude, current-task benefit, and past-task damage,
does update *direction* independently predict future-trainability damage?

Protocol (pre-registered):
  1. Reuse the E02 measurement frame (or regenerate deterministically).
  2. Fit the β4 regression FTD ~ mag + ctb + ptd + d_dir with HC3 SE.
  3. Bootstrap β4 (B=1000) for a distribution-free CI + sign-flip check.
  4. Run PTD-caliper matching sweep at {0.01, 0.02, 0.05} (Control C).
     β4 must exclude 0 in *all three* to pass.
  5. Write plots + JSON summary.
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


def build_frame(seed_offset: int = 0) -> pd.DataFrame:
    seed_everything(PRIMARY_SEED + seed_offset)
    teacher = build_teacher(TEACHER_STUDENT, seed=PRIMARY_SEED + seed_offset)
    train_by_task, eval_by_task = make_task_stream(
        teacher, TEACHER_STUDENT, seed=PRIMARY_SEED + seed_offset,
    )
    probes = build_probe_tasks(teacher, TEACHER_STUDENT, PROBE,
                               seed=PRIMARY_SEED + seed_offset)
    student = MLP(
        input_dim=TEACHER_STUDENT.input_dim,
        output_dim=TEACHER_STUDENT.output_dim,
        hidden_width=TEACHER_STUDENT.hidden_width,
        hidden_depth=TEACHER_STUDENT.hidden_depth,
        activation=TEACHER_STUDENT.activation,
    )
    student = _sequential_warmup(
        student, train_by_task,
        n_warmup_tasks=TEACHER_STUDENT.n_tasks - 1,
        lr=5e-3, steps=300, batch=128, seed=PRIMARY_SEED + seed_offset,
    )
    baseline = deepcopy(student)
    updates = generate_all_updates(
        baseline, train_by_task[TEACHER_STUDENT.n_tasks - 1], UPDATE_GEN,
        seed=PRIMARY_SEED + 500 + seed_offset,
    )
    g_ref = _reference_gradient(
        baseline, train_by_task[TEACHER_STUDENT.n_tasks - 1],
        UPDATE_GEN.sgd_inner_batch, seed=PRIMARY_SEED + 991 + seed_offset,
    )
    g_task = _reference_gradient(
        baseline, train_by_task[TEACHER_STUDENT.n_tasks - 1],
        UPDATE_GEN.sgd_inner_batch, seed=PRIMARY_SEED + 12345 + seed_offset,
    )
    random_ref = torch.randn_like(flat_params(baseline))
    past = list(eval_by_task[:TEACHER_STUDENT.n_tasks - 1])
    frame = build_measurement_frame(
        baseline, updates, probes, past,
        reference_gradient=g_ref, task_gradient=g_task, random_reference=random_ref,
        probe_cfg=PROBE, probe_seed=PRIMARY_SEED + seed_offset,
    )
    df = measurements_to_dataframe(frame).rename(columns={"delta_norm": "mag"})
    return df


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
    print("[E03] Building measurement frame ...")
    df = build_frame(seed_offset=0)
    df.to_csv(os.path.join(RESULTS_TABLES, "e03_measurements.csv"), index=False)
    print(f"[E03] frame shape: {df.shape}")

    covariates = list(STATS.covariates)

    # ---------- Full-sample regression ----------
    full = fit_beta4_regression(
        df, covariates=covariates, outcome="ftd",
        cov_type=STATS.ols_cov_type, alpha=STATS.beta4_significance_alpha,
    )
    print(f"\n[E03] Full-sample OLS ({STATS.ols_cov_type}), n={full.n_obs}, R^2={full.r_squared:.4f}")
    print(f"       beta_d_dir = {full.coefs['d_dir']:.5f}  "
          f"CI=[{full.ci_low['d_dir']:.5f}, {full.ci_high['d_dir']:.5f}]  "
          f"p={full.pvalues['d_dir']:.4g}")

    # ---------- Bootstrap ----------
    b_mean, b_lo, b_hi, sign_flip = bootstrap_beta4(
        df, covariates=covariates, outcome="ftd",
        n_iter=STATS.bootstrap_iterations, alpha=STATS.bootstrap_alpha, seed=PRIMARY_SEED,
    )
    print(f"[E03] Bootstrap beta_d_dir: mean={b_mean:.5f}  "
          f"95% CI=[{b_lo:.5f}, {b_hi:.5f}]  sign_flip={sign_flip:.3f}")

    # ---------- Control C sweep (PTD caliper matching) ----------
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
                "reason": "insufficient matched samples",
            })
            print(f"[E03] caliper={cal}: only {len(matched)} matched (skipped)")
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
        print(f"[E03] caliper={cal}: n={len(matched)}  beta={r.coefs['d_dir']:.5f}  "
              f"CI=[{r.ci_low['d_dir']:.5f},{r.ci_high['d_dir']:.5f}]  "
              f"p={r.pvalues['d_dir']:.4g}  excludes0={excl}  "
              f"within/total_var={wvb:.3f}")

    # ---------- Robustness verdict ----------
    excludes_all = all(c.get("excludes_zero") for c in caliper_results
                       if c.get("beta_d_dir") is not None)
    passed_kill2 = excludes_all
    passed_kill4 = sign_flip <= KILL.max_sign_flip_fraction
    # Control C failure kill #3: within/total var must be small across calipers
    ctrl_c_ok = all(
        (c.get("within_over_total_var") or 1.0) <= KILL.within_caliper_ratio_max
        for c in caliper_results if c.get("beta_d_dir") is not None
    )

    verdict = ("PASS" if (passed_kill2 and passed_kill4 and ctrl_c_ok)
               else "FAIL_or_PARTIAL")

    # ---------- Plots ----------
    fig, ax = plt.subplots(figsize=(6, 4))
    for src, sub in df.groupby("source"):
        ax.scatter(sub["d_dir"], sub["ftd"], alpha=0.6, label=src, s=18)
    ax.set_xlabel("d_dir (cos with reference gradient)")
    ax.set_ylabel("FTD (probe learning-speed damage)")
    ax.set_title("E03 — FTD vs d_dir by update source (synthetic)")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_FIGS, "e03_ftd_vs_ddir.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    calipers = [c["caliper"] for c in caliper_results if c.get("beta_d_dir") is not None]
    betas   = [c["beta_d_dir"] for c in caliper_results if c.get("beta_d_dir") is not None]
    lows    = [c["ci_low"]     for c in caliper_results if c.get("beta_d_dir") is not None]
    highs   = [c["ci_high"]    for c in caliper_results if c.get("beta_d_dir") is not None]
    yerr = np.array([np.array(betas) - np.array(lows), np.array(highs) - np.array(betas)])
    ax.errorbar(range(len(calipers)), betas, yerr=yerr, fmt="o", capsize=4)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(range(len(calipers)))
    ax.set_xticklabels([f"c={c}" for c in calipers])
    ax.set_ylabel("beta_d_dir (95% CI)")
    ax.set_title("E03 — Control C sweep: beta_d_dir across PTD calipers")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_FIGS, "e03_control_c_sweep.png"), dpi=140)
    plt.close(fig)

    summary = {
        "n_updates": int(len(df)),
        "full_regression": {
            "coefs": full.coefs, "ci_low": full.ci_low, "ci_high": full.ci_high,
            "p": full.pvalues, "vif": full.vif, "r_squared": full.r_squared,
        },
        "bootstrap": {
            "mean": b_mean, "ci_low": b_lo, "ci_high": b_hi,
            "sign_flip_fraction": sign_flip,
            "n_iter": STATS.bootstrap_iterations,
        },
        "control_c_sweep": caliper_results,
        "kill_conditions": {
            "kill1_identifiability_max_vif": float(max(full.vif.values())),
            "kill1_passed": bool(max(full.vif.values()) <= KILL.max_vif),
            "kill2_null_residual_passed_all_calipers": bool(passed_kill2),
            "kill3_control_c_passed": bool(ctrl_c_ok),
            "kill4_sign_flip_fraction": float(sign_flip),
            "kill4_passed": bool(passed_kill4),
        },
        "verdict": verdict,
    }
    summary = _stringify(summary)
    with open(os.path.join(RESULTS_TABLES, "e03_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[E03] VERDICT: {verdict}")
    print(f"[E03] wrote {os.path.join(RESULTS_TABLES, 'e03_summary.json')}")


if __name__ == "__main__":
    main()
