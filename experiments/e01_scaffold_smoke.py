"""E01 — Smoke test that the whole synthetic pipeline runs end-to-end.

Not a scientific result; just a pipeline check. Should print a small handful of
numbers and exit cleanly in under a minute on CPU.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from copy import deepcopy

from configs.preregistration import (
    PRIMARY_SEED, TEACHER_STUDENT, PROBE, UPDATE_GEN,
)
from src.seeds import seed_everything
from src.models import MLP, flat_params, num_params
from src.teacher_student import (
    build_teacher, make_task_stream, build_probe_tasks, sample_task_batch,
)
from src.training import train_task_and_capture_delta, compute_probe_ftd
from src.update_generator import generate_all_updates, _reference_gradient
from src.metrics import build_measurement_frame, measurements_to_dataframe


def main() -> None:
    seed_everything(PRIMARY_SEED)

    # 1. Teacher + task stream
    teacher = build_teacher(TEACHER_STUDENT, seed=PRIMARY_SEED)
    train_by_task, eval_by_task = make_task_stream(
        teacher, TEACHER_STUDENT, seed=PRIMARY_SEED,
    )
    probes = build_probe_tasks(teacher, TEACHER_STUDENT, PROBE, seed=PRIMARY_SEED)
    print(f"[E01] teacher params: {num_params(teacher)}")
    print(f"[E01] tasks: {len(train_by_task)}  probes: {len(probes)}")

    # 2. Student trained briefly on task 0, checkpointed
    student = MLP(
        input_dim=TEACHER_STUDENT.input_dim,
        output_dim=TEACHER_STUDENT.output_dim,
        hidden_width=TEACHER_STUDENT.hidden_width,
        hidden_depth=TEACHER_STUDENT.hidden_depth,
        activation=TEACHER_STUDENT.activation,
    )
    print(f"[E01] student params: {num_params(student)}")
    g = torch.Generator().manual_seed(PRIMARY_SEED)
    delta, ctb, final = train_task_and_capture_delta(
        student, train_by_task[0], n_steps=100, lr=5e-3, mini_size=128, generator=g,
    )
    print(f"[E01] task0 warm-up: ctb={ctb:.4f}  final_loss={final:.4f}  ||dtheta||={delta.norm():.4f}")

    # 3. Generate a small batch of updates and measure them
    small_cfg = type(UPDATE_GEN)(**{**UPDATE_GEN.__dict__, "per_source_per_norm": 4,
                                    "target_norms": (0.1,)})
    updates = generate_all_updates(student, train_by_task[1], small_cfg,
                                   seed=PRIMARY_SEED + 100)
    print(f"[E01] generated {len(updates)} matched updates")

    g_ref = _reference_gradient(student, train_by_task[1], 128, seed=PRIMARY_SEED)
    g_task = _reference_gradient(student, train_by_task[1], 128, seed=PRIMARY_SEED + 5)
    random_ref = torch.randn_like(flat_params(student))

    frame = build_measurement_frame(
        student, updates, probes, past_task_batches=[eval_by_task[0]],
        reference_gradient=g_ref, task_gradient=g_task, random_reference=random_ref,
        probe_cfg=PROBE, probe_seed=PRIMARY_SEED,
    )
    df = measurements_to_dataframe(frame)
    print(f"[E01] measurement frame shape: {df.shape}")
    print(df[["source", "delta_norm", "d_dir", "ctb", "ptd", "ftd"]].to_string(index=False))
    print("[E01] OK")


if __name__ == "__main__":
    main()
