"""Adaptive per-step directional controller for plasticity (v1.3, D-038).

The core mechanism (state-dependent, per-step):

  1. Every `reestimate_every` steps, sample K auxiliary probes from a plausible-
     future-task distribution and compute their mean pre-update gradient:
         g_hazard_hat(t) = normalize( mean_i grad_L_probe_i(θ_t) )
  2. On each optimizer step, compute the plain SGD update Δθ = -lr · g_train.
  3. Project: Δθ' = Δθ - α · (Δθ · û_hazard) û_hazard.
  4. Exact-renorm: Δθ' *= ‖Δθ‖ / ‖Δθ'‖  (kill #6 satisfied by construction).
  5. Apply θ ← θ + Δθ'.

Auxiliary probe sources (frozen in D-038):
  - permuted-input probes: random per-pixel permutations of current-task inputs
    (proxy for "structural novelty" — future structural probes)
  - semantic-novel-class probes: images from held-out classes with original
    labels (proxy for "semantic novelty" — future semantic probes)

We do NOT use FTD feedback (too expensive). We use gradient similarity as the
proxy for the direction that PREDICTIVE theory (Phase 1, E16) says drives β.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Callable, List, Optional, Tuple
import numpy as np

from .models import flat_params, set_flat_params, flat_grad


def _iter(x, y, batch, gen):
    n = x.shape[0]
    perm = torch.randperm(n, generator=gen)
    for i in range(0, n, batch):
        idx = perm[i:i + batch]
        yield x[idx], y[idx]


def sample_aux_probe_gradient(model, aux_x, aux_y, batch: int, seed: int,
                              device: torch.device) -> torch.Tensor:
    """One-batch gradient of the probe's loss at the current model."""
    m = deepcopy(model).to(device)
    g = torch.Generator().manual_seed(seed)
    xb_cpu, yb_cpu = next(_iter(aux_x, aux_y, batch, g))
    xb = xb_cpu.to(device); yb = yb_cpu.to(device)
    m.zero_grad(set_to_none=True)
    F.cross_entropy(m(xb), yb).backward()
    return flat_grad(m).detach().cpu()


def build_permuted_aux(train_x, train_y, n: int, seed: int, subset: int = 512
                        ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """n permuted-input auxiliary probes drawn from current-task samples."""
    rng = np.random.default_rng(seed)
    probes = []
    for i in range(n):
        idx = rng.choice(train_x.shape[0], size=min(subset, train_x.shape[0]),
                         replace=False)
        x_sub = train_x[idx]
        if x_sub.dim() == 2:
            perm = rng.permutation(x_sub.shape[1])
            x = x_sub[:, perm]
        elif x_sub.dim() == 4:
            n_, c, h, w = x_sub.shape
            perm = rng.permutation(h * w)
            x = x_sub.reshape(n_, c, h * w)[:, :, perm].reshape(n_, c, h, w)
        y = train_y[idx].long()
        probes.append((x, y))
    return probes


def build_semantic_aux(pool_x, pool_y, held_out_classes: Tuple[int, ...],
                       n: int, seed: int, subset: int = 512
                       ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """n semantic-novel-class auxiliary probes drawn from held-out classes with
    ORIGINAL labels (head-consistent, D-034 canonical)."""
    rng = np.random.default_rng(seed)
    y_np = pool_y.numpy() if isinstance(pool_y, torch.Tensor) else pool_y
    pool_idx = np.where(np.isin(y_np, list(held_out_classes)))[0]
    if len(pool_idx) == 0:
        return []
    probes = []
    for i in range(n):
        pick = rng.choice(pool_idx, size=min(subset, len(pool_idx)),
                          replace=False)
        x = pool_x[pick]
        y = pool_y[pick].long() if isinstance(pool_y, torch.Tensor) \
            else torch.from_numpy(pool_y[pick].astype(np.int64))
        probes.append((x, y))
    return probes


class AdaptiveDirectionalController:
    """Estimates u_hazard(t) from auxiliary probes, projects SGD updates away
    from it, and exact-renorms to preserve ‖Δθ‖.

    Usage:
      ctrl = AdaptiveDirectionalController(...)
      ctrl.begin_task(model, train_x, train_y, aux_semantic_pool)   # each new task
      for step in ...:
        # compute sgd_delta as usual
        applied_delta = ctrl.apply(model, sgd_delta, step_id, device)
        # then θ ← θ + applied_delta
    """

    def __init__(
        self,
        n_permuted_aux: int = 3,
        n_semantic_aux: int = 3,
        reestimate_every: int = 50,
        projection_alpha: float = 0.5,
        top_k: int = 1,
        held_out_classes: Optional[Tuple[int, ...]] = None,
        aux_pool_x: Optional[torch.Tensor] = None,
        aux_pool_y: Optional[torch.Tensor] = None,
        seed: int = 0,
        batch: int = 128,
    ):
        self.n_perm = n_permuted_aux
        self.n_sem = n_semantic_aux
        self.reest = reestimate_every
        self.alpha = float(projection_alpha)
        self.top_k = int(top_k)
        self.held_out = held_out_classes
        self.aux_pool_x = aux_pool_x
        self.aux_pool_y = aux_pool_y
        self.seed = seed
        self.batch = batch
        self.u_hazard: Optional[torch.Tensor] = None
        self._task_train_x = None
        self._task_train_y = None
        self._probes_perm = None
        self._probes_sem = None

    def begin_task(self, model, train_x, train_y):
        self._task_train_x = train_x
        self._task_train_y = train_y
        self._probes_perm = build_permuted_aux(train_x, train_y, self.n_perm,
                                                seed=self.seed + 101)
        if self.aux_pool_x is not None and self.held_out is not None:
            self._probes_sem = build_semantic_aux(self.aux_pool_x, self.aux_pool_y,
                                                    self.held_out, self.n_sem,
                                                    seed=self.seed + 202)
        else:
            self._probes_sem = []
        self.u_hazard = None    # force re-estimation

    def _reestimate(self, model, device, step_id: int):
        grads = []
        for i, (x, y) in enumerate(self._probes_perm):
            g_i = sample_aux_probe_gradient(model, x, y, self.batch,
                                             seed=self.seed + 1000 * step_id + i,
                                             device=device)
            grads.append(g_i / (g_i.norm() + 1e-12))
        for i, (x, y) in enumerate(self._probes_sem):
            g_i = sample_aux_probe_gradient(model, x, y, self.batch,
                                             seed=self.seed + 2000 * step_id + i,
                                             device=device)
            grads.append(g_i / (g_i.norm() + 1e-12))
        if not grads:
            self.u_hazard = None; return
        G = torch.stack(grads, dim=0)      # (K, D)
        if self.top_k <= 1:
            u = G.mean(dim=0)
            self.u_hazard = u / (u.norm() + 1e-12)
        else:
            # top-k SVD directions
            _, S, Vt = torch.linalg.svd(G, full_matrices=False)
            self.u_hazard = Vt[:self.top_k]      # (k, D)

    def apply(self, model, sgd_delta: torch.Tensor, step_id: int,
              device: torch.device) -> torch.Tensor:
        if self.alpha == 0.0:
            return sgd_delta
        if self.u_hazard is None or (step_id % self.reest == 0):
            self._reestimate(model, device, step_id)
        if self.u_hazard is None: return sgd_delta

        orig_norm = sgd_delta.norm()
        if self.u_hazard.dim() == 1:
            u = self.u_hazard
            d_perp = sgd_delta - self.alpha * (sgd_delta @ u) * u
        else:
            # multi-direction projection
            U = self.u_hazard          # (k, D)
            proj = U.T @ (U @ sgd_delta)
            d_perp = sgd_delta - self.alpha * proj
        # Exact renorm to preserve ‖Δθ‖ (kill #6)
        d_perp = d_perp * (orig_norm / (d_perp.norm() + 1e-12))
        return d_perp
