"""Per-update descriptors and FTD/PTD measurements."""
from __future__ import annotations
import torch
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional

from .models import flat_params, set_flat_params, flat_grad
from .teacher_student import TaskBatch, iterate_minibatches
from .training import _loss, evaluate, compute_probe_ftd


@dataclass
class UpdateMeasurements:
    update_id: int
    source: str
    target_norm: float
    delta_norm: float
    d_dir: float       # primary directional descriptor: cos(delta, g_ref)
    d_dir_task: float  # secondary: cos(delta, current-task gradient)
    d_dir_random: float  # tertiary: cos(delta, a fixed random reference)
    ctb: float
    ptd: float
    ftd: float


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    an = a.norm() + 1e-12
    bn = b.norm() + 1e-12
    return float((a @ b / (an * bn)).item())


def build_measurement_frame(
    base_model,
    updates,
    probe_tasks,
    past_task_batches: List[TaskBatch],
    reference_gradient: torch.Tensor,
    task_gradient: torch.Tensor,
    random_reference: torch.Tensor,
    probe_cfg,
    probe_seed: int,
    is_classification: bool = False,
) -> List[UpdateMeasurements]:
    """For each generated update, apply it, measure PTD/FTD, and pack descriptors."""
    theta_before = flat_params(base_model)
    # Reference FTD from *before* any update (baseline learning speed):
    ref_ftd = compute_probe_ftd(
        base_model, probe_tasks,
        probe_cfg.probe_steps, probe_cfg.probe_lr, probe_cfg.probe_batch,
        seed=probe_seed, is_classification=is_classification,
    )
    # PTD baseline: current loss on past tasks
    ref_past_losses = [evaluate(base_model, b, is_classification) for b in past_task_batches] \
        if past_task_batches else []

    frame: List[UpdateMeasurements] = []
    for uid, u in enumerate(updates):
        # Restore base and apply u.delta
        m = deepcopy(base_model)
        set_flat_params(m, theta_before + u.delta)

        # FTD (probe learning speed after the update)
        ftd_after = compute_probe_ftd(
            m, probe_tasks,
            probe_cfg.probe_steps, probe_cfg.probe_lr, probe_cfg.probe_batch,
            seed=probe_seed, is_classification=is_classification,
        )
        ftd = ftd_after - ref_ftd

        # PTD (loss on past tasks after update, summed increase)
        if past_task_batches:
            after_past = [evaluate(m, b, is_classification) for b in past_task_batches]
            ptd = float(sum(a - r for a, r in zip(after_past, ref_past_losses)))
        else:
            ptd = 0.0

        frame.append(UpdateMeasurements(
            update_id=uid,
            source=u.source,
            target_norm=u.target_norm,
            delta_norm=float(u.delta.norm().item()),
            d_dir=cos(u.delta, reference_gradient),
            d_dir_task=cos(u.delta, task_gradient),
            d_dir_random=cos(u.delta, random_reference),
            ctb=float(u.ctb),
            ptd=ptd,
            ftd=ftd,
        ))
    return frame


def measurements_to_dataframe(frame: List[UpdateMeasurements]):
    import pandas as pd
    return pd.DataFrame([m.__dict__ for m in frame])
