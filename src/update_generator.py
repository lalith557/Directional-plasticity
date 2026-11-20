"""Generate matched-‖Δθ‖ updates from three decorrelated sources.

The goal is to sample many candidate updates Δθ that:
  - all share a target L2 norm (Control A),
  - span many *directions* so D_dir can be regressed independently,
  - collectively decorrelate D_dir from CTB and PTD as much as possible
    (needed to survive the VIF gate; see decision.md D-003).
"""
from __future__ import annotations
import torch
from copy import deepcopy
from dataclasses import dataclass
from typing import List

from .models import flat_params, set_flat_params, flat_grad
from .teacher_student import TaskBatch, iterate_minibatches
from .training import train_task_and_capture_delta, _loss  # noqa: F401


@dataclass
class UpdateSample:
    source: str            # one of "sgd_seed", "orth_perturb", "random_sphere"
    target_norm: float
    delta: torch.Tensor    # flat vector; exactly renormalized to target_norm
    ctb: float             # current-task benefit measured before renormalize
    theta_before: torch.Tensor  # baseline parameters (shared across samples)


def _renormalize(delta: torch.Tensor, target_norm: float) -> torch.Tensor:
    n = delta.norm()
    if n < 1e-12:
        return delta
    return delta * (target_norm / n)


def _reference_gradient(
    model, train_batch: TaskBatch, mini_size: int, seed: int,
    is_classification: bool = False,
) -> torch.Tensor:
    """One full-batch (or single-minibatch) gradient at theta_before."""
    m = deepcopy(model)
    for p in m.parameters():
        p.requires_grad_(True)
    g = torch.Generator().manual_seed(seed)
    mb = next(iterate_minibatches(train_batch, mini_size, g))
    m.zero_grad(set_to_none=True)
    loss = _loss(m, mb, is_classification)
    loss.backward()
    return flat_grad(m)


def generate_sgd_seed_updates(
    base_model, train_batch: TaskBatch, cfg, target_norms, seed: int,
    per_source_per_norm: int, is_classification: bool = False,
) -> List[UpdateSample]:
    """Run short SGD trajectories from base_model with different data orders."""
    out: List[UpdateSample] = []
    theta_before = flat_params(base_model)
    for k, norm in enumerate(target_norms):
        for s in range(per_source_per_norm):
            m = deepcopy(base_model)
            for p in m.parameters():
                p.requires_grad_(True)
            g = torch.Generator().manual_seed(seed + 101 * k + s)
            delta, ctb, _ = train_task_and_capture_delta(
                m, train_batch, cfg.sgd_inner_steps, cfg.sgd_inner_lr,
                cfg.sgd_inner_batch, g, is_classification=is_classification,
            )
            delta = _renormalize(delta, norm)
            out.append(UpdateSample("sgd_seed", norm, delta, ctb, theta_before))
    return out


def generate_orth_perturb_updates(
    base_model, train_batch: TaskBatch, cfg, target_norms, seed: int,
    per_source_per_norm: int, is_classification: bool = False,
) -> List[UpdateSample]:
    """Orthogonal perturbations of the reference gradient."""
    out: List[UpdateSample] = []
    theta_before = flat_params(base_model)
    g_ref = _reference_gradient(
        base_model, train_batch, cfg.sgd_inner_batch, seed=seed + 991,
        is_classification=is_classification,
    )
    g_hat = g_ref / (g_ref.norm() + 1e-12)
    D = g_ref.numel()
    for k, norm in enumerate(target_norms):
        for s in range(per_source_per_norm):
            g = torch.Generator().manual_seed(seed + 313 * k + 7 * s + 1)
            eps = torch.empty(D).normal_(std=cfg.orth_noise_std, generator=g)
            # Remove component along g_hat: d_perp = (I - ĝĝᵀ) eps
            d_perp = eps - (eps @ g_hat) * g_hat
            delta = _renormalize(d_perp, norm)
            # Approximate CTB by first-order Taylor: -g·Δθ (positive == loss reduction)
            ctb_approx = float(-(g_ref @ delta).item())
            out.append(UpdateSample("orth_perturb", norm, delta, ctb_approx, theta_before))
    return out


def generate_random_sphere_updates(
    base_model, train_batch: TaskBatch, cfg, target_norms, seed: int,
    per_source_per_norm: int, is_classification: bool = False,
) -> List[UpdateSample]:
    """Isotropic random directions on the ‖Δθ‖-sphere."""
    out: List[UpdateSample] = []
    theta_before = flat_params(base_model)
    g_ref = _reference_gradient(
        base_model, train_batch, cfg.sgd_inner_batch, seed=seed + 991,
        is_classification=is_classification,
    )
    D = g_ref.numel()
    for k, norm in enumerate(target_norms):
        for s in range(per_source_per_norm):
            g = torch.Generator().manual_seed(seed + 991 * k + 13 * s + 3)
            v = torch.empty(D).normal_(generator=g)
            delta = _renormalize(v, norm)
            ctb_approx = float(-(g_ref @ delta).item())  # random dot with grad
            out.append(UpdateSample("random_sphere", norm, delta, ctb_approx, theta_before))
    return out


def generate_convex_mix_updates(
    base_model, train_batch: TaskBatch, cfg, target_norms, seed: int,
    per_source_per_norm: int, is_classification: bool = False,
    n_alpha_steps: int = 5,
) -> List[UpdateSample]:
    """Continuous mixture of SGD-direction and random-direction.

    delta_mix(alpha) = alpha * u_sgd + (1-alpha) * u_random,  then renormalize.

    Sweeping alpha in (0, 1) gives updates whose d_dir smoothly interpolates
    from ~0 (pure random) to ~-1 (pure SGD). Introduced *after* seeing E03's
    sign flip at caliper 0.05 to enable within-source PTD caliper matching
    (see decision.md D-013).
    """
    out: List[UpdateSample] = []
    theta_before = flat_params(base_model)
    g_ref = _reference_gradient(
        base_model, train_batch, cfg.sgd_inner_batch, seed=seed + 991,
        is_classification=is_classification,
    )
    D = g_ref.numel()
    # Pre-compute a pool of SGD deltas (short trajectories) and random deltas
    n_pool = max(per_source_per_norm, n_alpha_steps)
    sgd_pool = []
    for i in range(n_pool):
        m = deepcopy(base_model)
        for p in m.parameters():
            p.requires_grad_(True)
        g = torch.Generator().manual_seed(seed + 707 * i + 1)
        d, _, _ = train_task_and_capture_delta(
            m, train_batch, cfg.sgd_inner_steps, cfg.sgd_inner_lr,
            cfg.sgd_inner_batch, g, is_classification=is_classification,
        )
        sgd_pool.append(d / (d.norm() + 1e-12))
    rand_pool = []
    for i in range(n_pool):
        g = torch.Generator().manual_seed(seed + 1213 * i + 2)
        v = torch.empty(D).normal_(generator=g)
        rand_pool.append(v / (v.norm() + 1e-12))
    alphas = torch.linspace(0.05, 0.95, n_alpha_steps).tolist()
    for k, norm in enumerate(target_norms):
        for s in range(per_source_per_norm):
            alpha = alphas[s % n_alpha_steps]
            u_sgd = sgd_pool[s % n_pool]
            u_rand = rand_pool[(s + k) % n_pool]
            mixed = alpha * u_sgd + (1 - alpha) * u_rand
            delta = _renormalize(mixed, norm)
            ctb_approx = float(-(g_ref @ delta).item())
            out.append(UpdateSample("convex_mix", norm, delta, ctb_approx, theta_before))
    return out


def generate_all_updates(
    base_model, train_batch: TaskBatch, cfg, seed: int,
    is_classification: bool = False,
    include_convex_mix: bool = False,
) -> List[UpdateSample]:
    """Combine the three pre-registered sources; optionally add convex_mix (E03b+)."""
    per = cfg.per_source_per_norm
    norms = cfg.target_norms
    a = generate_sgd_seed_updates(
        base_model, train_batch, cfg, norms, seed + 1, per, is_classification,
    )
    b = generate_orth_perturb_updates(
        base_model, train_batch, cfg, norms, seed + 2, per, is_classification,
    )
    c = generate_random_sphere_updates(
        base_model, train_batch, cfg, norms, seed + 3, per, is_classification,
    )
    out = a + b + c
    if include_convex_mix:
        d = generate_convex_mix_updates(
            base_model, train_batch, cfg, norms, seed + 4, per, is_classification,
        )
        out = out + d
    return out
