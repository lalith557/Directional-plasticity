"""E02 — Identifiability pre-check (kill condition #1 gate).

Goal: on the *synthetic* data, confirm that D_dir can be regressed from
{‖Δθ‖, CTB, PTD} without unacceptable multicollinearity. If any VIF > threshold
after decorrelation-by-design, we've hit kill #1 in Week 1 — the cheapest
possible failure — and either widen the update generator or redesign.

This script does NOT look at the β4 coefficient itself. It only asks: is the
design *identifiable*?
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from copy import deepcopy

from configs.preregistration import (
    PRIMARY_SEED, TEACHER_STUDENT, PROBE, UPDATE_GEN, STATS, KILL,
)
from src.seeds import seed_everything
from src.models import MLP, flat_params
from src.teacher_student import (
    build_teacher, make_task_stream, build_probe_tasks,
)
from src.training import train_task_and_capture_delta
from src.update_generator import generate_all_updates, _reference_gradient
from src.metrics import build_measurement_frame, measurements_to_dataframe
from src.stats import compute_vif


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "tables")


def _sequential_warmup(student, train_by_task, n_warmup_tasks: int,
                       lr: float, steps: int, batch: int, seed: int):
    """Train student sequentially on the first `n_warmup_tasks` tasks so it
    reaches a nontrivial plasticity regime before we generate candidate updates."""
    for t in range(n_warmup_tasks):
        g = torch.Generator().manual_seed(seed + 17 * t)
        train_task_and_capture_delta(
            student, train_by_task[t], n_steps=steps, lr=lr,
            mini_size=batch, generator=g,
        )
    return student


def main() -> None:
    seed_everything(PRIMARY_SEED)

    teacher = build_teacher(TEACHER_STUDENT, seed=PRIMARY_SEED)
    train_by_task, eval_by_task = make_task_stream(
        teacher, TEACHER_STUDENT, seed=PRIMARY_SEED,
    )
    probes = build_probe_tasks(teacher, TEACHER_STUDENT, PROBE, seed=PRIMARY_SEED)

    # Student pre-trained through tasks 0..N-2, so update at task N-1 is
    # applied from a "used" checkpoint where plasticity regime is nontrivial.
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
        lr=5e-3, steps=300, batch=128, seed=PRIMARY_SEED,
    )
    baseline = deepcopy(student)
    print(f"[E02] student warmed up on tasks 0..{TEACHER_STUDENT.n_tasks - 2}")

    # Generate matched-‖Δθ‖ updates from three sources (uses pre-registered counts)
    updates = generate_all_updates(
        baseline, train_by_task[TEACHER_STUDENT.n_tasks - 1], UPDATE_GEN,
        seed=PRIMARY_SEED + 500,
    )
    print(f"[E02] generated {len(updates)} candidate updates "
          f"(sources={UPDATE_GEN.sources}, norms={UPDATE_GEN.target_norms})")

    # Reference and secondary directions for descriptors
    g_ref = _reference_gradient(baseline, train_by_task[TEACHER_STUDENT.n_tasks - 1],
                                UPDATE_GEN.sgd_inner_batch, seed=PRIMARY_SEED + 991)
    g_task = _reference_gradient(baseline, train_by_task[TEACHER_STUDENT.n_tasks - 1],
                                 UPDATE_GEN.sgd_inner_batch, seed=PRIMARY_SEED + 12345)
    random_ref = torch.randn_like(flat_params(baseline))

    past_task_batches = list(eval_by_task[:TEACHER_STUDENT.n_tasks - 1])
    frame = build_measurement_frame(
        baseline, updates, probes, past_task_batches,
        reference_gradient=g_ref, task_gradient=g_task, random_reference=random_ref,
        probe_cfg=PROBE, probe_seed=PRIMARY_SEED,
    )
    df = measurements_to_dataframe(frame)

    # Rename to the covariate names statsmodels expects
    df = df.rename(columns={"delta_norm": "mag"})
    df.to_csv(os.path.join(RESULTS_DIR, "e02_measurements.csv"), index=False)

    print("\n[E02] Descriptor summary per source:")
    print(df.groupby("source")[["mag", "ctb", "ptd", "d_dir", "ftd"]]
            .agg(["mean", "std"]).round(4).to_string())

    # VIF gate
    covariates = list(STATS.covariates)
    vif = compute_vif(df, covariates)
    print(f"\n[E02] VIF (threshold={STATS.vif_threshold}):")
    for k, v in vif.items():
        flag = "  <-- OVER" if v > STATS.vif_threshold else ""
        print(f"  {k}: {v:.3f}{flag}")

    # Pairwise correlations for extra insight
    corr = df[covariates + ["ftd"]].corr().round(3)
    print(f"\n[E02] Correlation matrix (covariates + ftd):\n{corr.to_string()}")

    max_vif = max(vif.values())
    passed = max_vif <= KILL.max_vif
    verdict = "PASS" if passed else "FAIL (kill #1 triggered)"
    print(f"\n[E02] Identifiability gate: {verdict} (max VIF = {max_vif:.3f})")

    out = {
        "n_updates": int(len(df)),
        "sources": list(df["source"].unique()),
        "vif": {k: float(v) for k, v in vif.items()},
        "max_vif": float(max_vif),
        "passed": bool(passed),
        "vif_threshold": float(STATS.vif_threshold),
        "corr_ftd_d_dir": float(corr.loc["d_dir", "ftd"]),
    }
    with open(os.path.join(RESULTS_DIR, "e02_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[E02] wrote {os.path.join(RESULTS_DIR, 'e02_summary.json')}")


if __name__ == "__main__":
    main()
