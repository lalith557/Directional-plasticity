"""Training loops used by both synthetic and MNIST experiments."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from copy import deepcopy
from typing import List, Tuple

from .models import MLP, flat_params, set_flat_params
from .teacher_student import TaskBatch, iterate_minibatches


def _loss(model, batch: TaskBatch, is_classification: bool) -> torch.Tensor:
    out = model(batch.x)
    if is_classification:
        return F.cross_entropy(out, batch.y)
    return F.mse_loss(out, batch.y)


def evaluate(model, batch: TaskBatch, is_classification: bool = False) -> float:
    with torch.no_grad():
        return float(_loss(model, batch, is_classification).item())


def train_task_and_capture_delta(
    model,
    train_batch: TaskBatch,
    n_steps: int,
    lr: float,
    mini_size: int,
    generator: torch.Generator,
    is_classification: bool = False,
) -> Tuple[torch.Tensor, float, float]:
    """Train `model` in place for `n_steps` SGD mini-batches.

    Returns (delta_theta, ctb, final_train_loss).
      delta_theta = theta_after - theta_before  (flat vector)
      ctb         = train-loss reduction over the segment
    """
    theta_before = flat_params(model)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    step = 0
    while step < n_steps:
        for mb in iterate_minibatches(train_batch, mini_size, generator):
            opt.zero_grad(set_to_none=True)
            loss = _loss(model, mb, is_classification)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            step += 1
            if step >= n_steps:
                break
    theta_after = flat_params(model)
    delta = theta_after - theta_before
    ctb = float(losses[0] - losses[-1]) if len(losses) >= 2 else 0.0
    return delta, ctb, float(losses[-1])


def apply_update(model, delta: torch.Tensor) -> None:
    theta = flat_params(model)
    set_flat_params(model, theta + delta)


def compute_probe_ftd(
    model,
    probe_tasks: List[Tuple[TaskBatch, TaskBatch]],
    probe_steps: int,
    probe_lr: float,
    probe_batch: int,
    seed: int,
    is_classification: bool = False,
) -> float:
    """Learning-speed FTD: mean over probes of AUC(loss over probe_steps).

    Higher AUC == slower learning == worse FTD.
    """
    aucs = []
    for p_id, (tr, _ev) in enumerate(probe_tasks):
        m = deepcopy(model)
        # De-freeze parameters if the probe copy was seeded from a frozen network
        for p in m.parameters():
            p.requires_grad_(True)
        opt = torch.optim.SGD(m.parameters(), lr=probe_lr)
        g = torch.Generator().manual_seed(seed + 7919 * (p_id + 1))
        losses = []
        step = 0
        while step < probe_steps:
            for mb in iterate_minibatches(tr, probe_batch, g):
                opt.zero_grad(set_to_none=True)
                loss = _loss(m, mb, is_classification)
                loss.backward()
                opt.step()
                losses.append(loss.item())
                step += 1
                if step >= probe_steps:
                    break
        # AUC (mean loss over the trajectory) — cheap proxy for area under curve.
        aucs.append(float(sum(losses) / len(losses)))
    return float(sum(aucs) / len(aucs))
