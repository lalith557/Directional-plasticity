"""Synthetic teacher-student task family used for the decisive experiment.

- One fixed random *teacher* MLP.
- Tasks are produced by rotating the teacher's first-layer weight matrix by
  small angles (givens-style rotations combined with a per-task orthogonal
  perturbation seeded from `task_id`). This gives a smoothly related family
  where task-to-task interference is present but controllable.
- The student learns to imitate the teacher (regression MSE).

This yields unlimited matched samples, ground-truth control over task overlap,
and runs in seconds on CPU.
"""
from __future__ import annotations
import math
import torch
from dataclasses import dataclass
from typing import Iterator, Tuple

from .models import MLP


@dataclass
class TaskBatch:
    x: torch.Tensor
    y: torch.Tensor


def build_teacher(cfg, seed: int) -> MLP:
    """Fan-in normalized random init so teacher outputs are unit-scale."""
    g = torch.Generator().manual_seed(seed)
    teacher = MLP(
        input_dim=cfg.input_dim,
        output_dim=cfg.output_dim,
        hidden_width=cfg.hidden_width,
        hidden_depth=cfg.hidden_depth,
        activation=cfg.activation,
    )
    for mod in teacher.modules():
        if isinstance(mod, torch.nn.Linear):
            fan_in = mod.weight.shape[1]
            # He init for relu-family
            std = (2.0 / fan_in) ** 0.5
            mod.weight.data.copy_(torch.empty_like(mod.weight).normal_(std=std, generator=g))
            mod.bias.data.zero_()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def _random_orthogonal(dim: int, generator: torch.Generator) -> torch.Tensor:
    """Random orthogonal matrix via QR of a Gaussian matrix."""
    a = torch.randn(dim, dim, generator=generator)
    q, r = torch.linalg.qr(a)
    # Make deterministic sign so QR is unique
    d = torch.diagonal(r).sign()
    d[d == 0] = 1
    return q * d


def task_rotation(
    teacher: MLP, task_id: int, angle_rad: float, seed: int
) -> MLP:
    """Return a *copy* of the teacher with its first-layer weight rotated.

    We use `task_id` and `seed` together so tasks are deterministic and
    distinguishable. angle == 0 → same task; larger angle → less overlap.
    """
    from copy import deepcopy
    t = deepcopy(teacher)
    if task_id == 0 or angle_rad == 0:
        return t
    first_linear: torch.nn.Linear = t.net[0]  # type: ignore
    w = first_linear.weight.data  # [hidden, input]
    dim_in = w.shape[1]
    g = torch.Generator().manual_seed(seed + 1000 * task_id)
    q = _random_orthogonal(dim_in, g)
    # Interpolate between identity and full rotation: cos*I + sin*Q_skew-symmetrized
    # Simplification: build a rotation R = expm(angle * skew(Q - Q.T)/2). We
    # approximate via first-order to avoid needing matrix exponential.
    skew = 0.5 * (q - q.T)
    skew_norm = skew.norm() + 1e-8
    skew = skew / skew_norm
    R = torch.eye(dim_in) * math.cos(angle_rad) + skew * math.sin(angle_rad)
    # Re-orthogonalize via QR to ensure exactness against numerical drift
    R, _ = torch.linalg.qr(R)
    first_linear.weight.data = w @ R
    for p in t.parameters():
        p.requires_grad_(False)
    return t


def sample_task_batch(
    teacher: MLP, batch_size: int, input_dim: int, input_std: float,
    generator: torch.Generator,
) -> TaskBatch:
    x = torch.randn(batch_size, input_dim, generator=generator) * input_std
    with torch.no_grad():
        y = teacher(x)
    return TaskBatch(x=x, y=y)


def make_task_stream(
    teacher: MLP, cfg, seed: int,
) -> Tuple[list, list]:
    """Return (train_batches_per_task, eval_batches_per_task) as fixed tensors."""
    train_by_task, eval_by_task = [], []
    for t_id in range(cfg.n_tasks):
        rotated = task_rotation(teacher, t_id, cfg.task_rotation_rad, seed)
        g_train = torch.Generator().manual_seed(seed + 10 * t_id)
        g_eval = torch.Generator().manual_seed(seed + 10 * t_id + 1)
        train_by_task.append(
            sample_task_batch(rotated, cfg.train_samples_per_task,
                              cfg.input_dim, cfg.input_std, g_train)
        )
        eval_by_task.append(
            sample_task_batch(rotated, cfg.eval_samples_per_task,
                              cfg.input_dim, cfg.input_std, g_eval)
        )
    return train_by_task, eval_by_task


def build_probe_tasks(
    teacher: MLP, cfg, probe_cfg, seed: int,
    use_novel_teacher: bool = False,
) -> list:
    """Return list of (train_batch, eval_batch) probes, held out from training.

    If `use_novel_teacher=True`, each probe uses an *independent* random teacher
    (different seed family). This measures true plasticity — the network's
    ability to learn something structurally new — rather than transfer to
    minor rotations of the training teacher.
    """
    probes = []
    for p_id in range(probe_cfg.n_probes):
        if use_novel_teacher:
            novel = build_teacher(cfg, seed=seed + 99991 * (p_id + 1))
            rotated = novel
        else:
            rotated = task_rotation(
                teacher, probe_cfg.probe_seed_offset + p_id,
                cfg.task_rotation_rad, seed,
            )
        g_train = torch.Generator().manual_seed(seed + 10 * (probe_cfg.probe_seed_offset + p_id))
        g_eval = torch.Generator().manual_seed(seed + 10 * (probe_cfg.probe_seed_offset + p_id) + 1)
        tr = sample_task_batch(rotated, cfg.train_samples_per_task,
                               cfg.input_dim, cfg.input_std, g_train)
        ev = sample_task_batch(rotated, cfg.eval_samples_per_task,
                               cfg.input_dim, cfg.input_std, g_eval)
        probes.append((tr, ev))
    return probes


def iterate_minibatches(
    batch: TaskBatch, mini_size: int, generator: torch.Generator,
) -> Iterator[TaskBatch]:
    n = batch.x.shape[0]
    perm = torch.randperm(n, generator=generator)
    for i in range(0, n, mini_size):
        idx = perm[i:i + mini_size]
        yield TaskBatch(x=batch.x[idx], y=batch.y[idx])
