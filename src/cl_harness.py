"""v1.4 task-incremental continual-learning harness (fixes the degenerate regime).

Root cause of the collapse in v1.2/v1.3 (see D-041): every arm used a single
shared 10-way head under a class-IL evaluation while each task's labels were
remapped to 0/1 — so the head's 0/1 outputs raced across tasks and only the
LAST task's mapping was retained (final_accs = [0,0,0,0,x]).

The fix (D-043): standard task-incremental (TIL) protocol — a shared backbone
plus PER-TASK 2-way heads (`MultiHeadMLP` / `MultiHeadCNN`, already in
`src/models.py`); training and evaluation always route through the head that
belongs to the current task. Backbone plasticity is what we measure; head-race
dynamics no longer confound the metric.

This module exposes:
  - `prepare_mnist_til(...)`  and `prepare_cifar10_til(...)` — TIL task lists
  - `train_task_il_sequence(...)` — sequential trainer supporting SGD/EWC/
    replay/CBP + the two-sided directional controller
  - `probe_ftd_til(...)` — FTD probe on a fresh probe head over the current
    backbone (learning-speed AUC-of-loss)
  - `strict_matched_magnitude_stats(...)` — per-arm norm bookkeeping

Every update, whether from a baseline or the controller, is renormalized to
match the *plain* SGD step's norm at the same θ — the STRICT_MATCH protocol
(D-043). Any arm whose max deviation exceeds 2% is flagged and excluded from
the Pareto comparison in e19.
"""
from __future__ import annotations
import os
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import (
    MultiHeadMLP, MultiHeadCNN,
    flat_params, set_flat_params, flat_grad,
)
from .adaptive_controller import AdaptiveDirectionalController
from .datasets import (
    load_mnist, split_mnist_tasks,
    load_cifar10, split_cifar10_tasks,
)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_mnist_til(mnist_dir: str,
                       class_pairs=((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
                       ) -> List[dict]:
    """Split-MNIST as a TIL benchmark. Labels are already remapped to 0/1
    per-task inside `split_mnist_tasks`. Each dict carries `train_x` (flat),
    `train_y` (0/1), `test_x`, `test_y`, `pair` (original class ids)."""
    tx, ty, vx, vy = load_mnist(mnist_dir)
    return split_mnist_tasks(tx, ty, vx, vy, class_pairs)


def prepare_cifar10_til(cifar10_dir: str,
                         class_pairs=((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
                         ) -> List[dict]:
    """Split-CIFAR-10 as a TIL benchmark. Wraps `split_cifar10_tasks` and
    remaps labels to 0/1 within each task so the per-task 2-way head is stable."""
    tx, ty, vx, vy = load_cifar10(cifar10_dir)
    raw_tasks = split_cifar10_tasks(tx, ty, vx, vy, class_pairs)
    for t in raw_tasks:
        pair = t["pair"]
        # Remap original class labels {pair[0], pair[1]} to {0, 1}
        t["train_y"] = torch.where(t["train_y"] == pair[0],
                                    torch.zeros_like(t["train_y"]),
                                    torch.ones_like(t["train_y"]))
        t["test_y"] = torch.where(t["test_y"] == pair[0],
                                   torch.zeros_like(t["test_y"]),
                                   torch.ones_like(t["test_y"]))
    return raw_tasks


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def build_mnist_model(hidden: int, depth: int, n_tasks: int,
                       n_classes_per_task: int = 2) -> MultiHeadMLP:
    return MultiHeadMLP(input_dim=784, hidden=hidden, depth=depth,
                         n_tasks=n_tasks,
                         n_classes_per_task=n_classes_per_task)


def build_cifar_model(base: int, fc: int, n_tasks: int,
                       n_classes_per_task: int = 2) -> MultiHeadCNN:
    return MultiHeadCNN(in_channels=3, base=base, fc=fc, n_tasks=n_tasks,
                         n_classes_per_task=n_classes_per_task,
                         input_hw=32)


# ---------------------------------------------------------------------------
# Backbone/features accessor (works for both TIL models)
# ---------------------------------------------------------------------------

def _backbone_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(model, MultiHeadMLP):
        return model.backbone(x)
    if isinstance(model, MultiHeadCNN):
        return model.trunk(model.features(x))
    raise TypeError(f"Unsupported model type: {type(model)}")


def _feature_dim(model: nn.Module) -> int:
    if isinstance(model, MultiHeadMLP):
        return int(model.heads[0].in_features)
    if isinstance(model, MultiHeadCNN):
        return int(model.heads[0].in_features)
    raise TypeError(f"Unsupported model type: {type(model)}")


# ---------------------------------------------------------------------------
# TIL forward helper
# ---------------------------------------------------------------------------

def til_forward(model: nn.Module, x: torch.Tensor, task_id: int) -> torch.Tensor:
    return model(x, task_id)


# ---------------------------------------------------------------------------
# Data iteration
# ---------------------------------------------------------------------------

def _iter(x, y, batch, gen):
    n = x.shape[0]
    perm = torch.randperm(n, generator=gen)
    for i in range(0, n, batch):
        idx = perm[i:i + batch]
        yield x[idx], y[idx]


# ---------------------------------------------------------------------------
# TIL-aware adaptive controller — overrides `_reestimate` to route auxiliary
# probes through a FRESH probe head on top of the backbone features (so the
# multi-head model's forward signature is respected).
# ---------------------------------------------------------------------------

class TILAdaptiveDirectionalController(AdaptiveDirectionalController):
    """State-dependent directional controller for the TIL harness.

    The parent class assumes `model(x)` returns logits. Our TIL models take
    `(x, task_id)` and have no default output. We instead compute probe
    gradients through a FRESH 2-way probe head on top of the backbone features,
    then take `flat_grad(model)` — this measures the direction the SGD update
    would take through the *backbone* if the model were asked to learn a
    novel probe task.
    """

    def _reestimate(self, model, device, step_id: int):    # noqa: D401
        grads = []
        # Only permuted-input aux probes for TIL (semantic aux disabled in e19)
        for i, (x, y) in enumerate(self._probes_perm):
            g_i = _probe_gradient_til(
                model, x, y, self.batch,
                seed=self.seed + 1000 * step_id + i, device=device)
            grads.append(g_i / (g_i.norm() + 1e-12))
        if not grads:
            self.u_hazard = None; return
        G = torch.stack(grads, dim=0)
        if self.top_k <= 1:
            u = G.mean(dim=0)
            self.u_hazard = u / (u.norm() + 1e-12)
        else:
            _, S, Vt = torch.linalg.svd(G, full_matrices=False)
            self.u_hazard = Vt[:self.top_k]


def _probe_gradient_til(model, aux_x, aux_y, batch, seed, device):
    """Compute the gradient of a novel-probe loss on a fresh 2-way head on top
    of the current backbone features. Returns flat gradient over `model.parameters()`
    (backbone gradients are what matters; per-task heads receive 0 gradient
    since the loss doesn't pass through them)."""
    m = deepcopy(model).to(device)
    d = _feature_dim(m)
    probe_head = nn.Linear(d, int(aux_y.max().item()) + 1
                            if aux_y.numel() > 0 else 2).to(device)
    g = torch.Generator().manual_seed(seed)
    xb_cpu, yb_cpu = next(_iter(aux_x, aux_y, batch, g))
    xb = xb_cpu.to(device); yb = yb_cpu.to(device)
    m.zero_grad(set_to_none=True)
    for p in probe_head.parameters():
        p.grad = None
    feat = _backbone_features(m, xb)
    logits = probe_head(feat)
    F.cross_entropy(logits, yb).backward()
    return flat_grad(m).detach().cpu()


# ---------------------------------------------------------------------------
# FTD probe (learning-speed AUC-of-loss over K steps on a fresh probe head)
# ---------------------------------------------------------------------------

def build_permuted_probes(pool_x: torch.Tensor, pool_y: torch.Tensor,
                            n_probes: int, seed: int, subset: int = 3000,
                            n_classes: int = 2
                            ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """n_probes permuted-input probes with 2-class labels (mod n_classes).

    Each probe: random per-pixel permutation on a random subset of pool_x, and
    labels = (original_class_id % n_classes) so a fresh 2-way probe head is
    well-defined. Same protocol for every arm → fair FTD comparison.
    """
    rng = np.random.default_rng(seed)
    probes = []
    for _ in range(n_probes):
        idx = rng.choice(pool_x.shape[0], size=min(subset, pool_x.shape[0]),
                          replace=False)
        x_sub = pool_x[idx]
        if x_sub.dim() == 2:
            perm = rng.permutation(x_sub.shape[1])
            x = x_sub[:, perm].contiguous()
        elif x_sub.dim() == 4:
            n_, c, h, w = x_sub.shape
            perm = rng.permutation(h * w)
            x = (x_sub.reshape(n_, c, h * w)[:, :, perm]
                        .reshape(n_, c, h, w)
                        .contiguous())
        else:
            raise ValueError(f"pool_x shape {x_sub.shape}")
        y = (pool_y[idx].long() % n_classes).contiguous()
        probes.append((x, y))
    return probes


def probe_ftd_til(model: nn.Module, probes, probe_steps: int, probe_lr: float,
                   probe_batch: int, seed: int, device: torch.device,
                   n_classes: int = 2) -> float:
    """FTD proxy = mean over probes of the AUC of loss over `probe_steps` SGD
    steps of joint (probe_head + backbone) training on the probe. Higher AUC =
    slower learning = worse plasticity.
    """
    aucs = []
    for p_id, (x, y) in enumerate(probes):
        m = deepcopy(model).to(device)
        # Fresh probe head on top of backbone features
        d = _feature_dim(m)
        probe_head = nn.Linear(d, n_classes).to(device)
        params = list(m.parameters()) + list(probe_head.parameters())
        opt = torch.optim.SGD(params, lr=probe_lr)
        g = torch.Generator().manual_seed(seed + 7919 * (p_id + 1))
        losses, step = [], 0
        while step < probe_steps:
            for xb_cpu, yb_cpu in _iter(x, y, probe_batch, g):
                xb = xb_cpu.to(device); yb = yb_cpu.to(device)
                opt.zero_grad(set_to_none=True)
                feat = _backbone_features(m, xb)
                logits = probe_head(feat)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                opt.step()
                losses.append(loss.item())
                step += 1
                if step >= probe_steps: break
        aucs.append(sum(losses) / max(len(losses), 1))
        del m, probe_head
    return float(sum(aucs) / max(len(aucs), 1))


# ---------------------------------------------------------------------------
# Baseline pieces (EWC / Replay / CBP) — all TIL-aware
# ---------------------------------------------------------------------------

def _backbone_and_head_params(model: nn.Module, task_id: int):
    """Return the parameters that receive gradients on a training step for
    task_id: the shared backbone + head[task_id] only (per-task head learning
    is standard TIL)."""
    if isinstance(model, MultiHeadMLP):
        return list(model.backbone.parameters()) + list(model.heads[task_id].parameters())
    if isinstance(model, MultiHeadCNN):
        return (list(model.features.parameters())
                + list(model.trunk.parameters())
                + list(model.heads[task_id].parameters()))
    raise TypeError(type(model))


def _model_flat_grad(model: nn.Module) -> torch.Tensor:
    return flat_grad(model)


class ReplayMemoryTIL:
    """Per-task replay buffer. Stores task-labeled samples so we can route each
    replayed batch through its own head during EWC/replay training."""

    def __init__(self, per_task: int):
        self.per_task = per_task
        self.buffers: List[Tuple[torch.Tensor, torch.Tensor, int]] = []

    def add_task(self, x, y, task_id: int, seed: int):
        n = min(self.per_task, x.shape[0])
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(x.shape[0], generator=g)[:n]
        self.buffers.append((x[idx].clone(), y[idx].clone(), int(task_id)))

    def sample_from_random_past(self, n: int, seed: int, device):
        if not self.buffers or n <= 0: return None
        g = torch.Generator().manual_seed(seed)
        j = int(torch.randint(0, len(self.buffers), (1,), generator=g).item())
        x, y, tid = self.buffers[j]
        k = min(n, x.shape[0])
        idx = torch.randint(0, x.shape[0], (k,), generator=g)
        return x[idx].to(device), y[idx].to(device), tid


def _compute_ewc_fisher_til(model, past_tasks, batch, seed, device):
    """Empirical Fisher over past tasks, routing each batch through its
    task-appropriate head."""
    D = sum(p.numel() for p in model.parameters())
    fisher = torch.zeros(D, device=device)
    count = 0
    for t_idx, task in enumerate(past_tasks):
        m = deepcopy(model).to(device)
        g = torch.Generator().manual_seed(seed + 999 * (t_idx + 1))
        xb_cpu, yb_cpu = next(_iter(task["train_x"], task["train_y"], batch, g))
        xb = xb_cpu.to(device); yb = yb_cpu.to(device)
        m.zero_grad(set_to_none=True)
        logits = m(xb, t_idx)
        with torch.no_grad():
            y_hat = logits.argmax(-1)
        F.nll_loss(F.log_softmax(logits, -1), y_hat).backward()
        fisher = fisher + _model_flat_grad(m).pow(2)
        count += 1
    return fisher / max(count, 1)


def _apply_ewc_penalty(model, fisher, theta_ref, lam):
    off = 0
    for p in model.parameters():
        n = p.numel()
        f_i = fisher[off:off + n].view_as(p)
        r_i = theta_ref[off:off + n].view_as(p)
        if p.grad is None: p.grad = torch.zeros_like(p)
        p.grad.data.add_(lam * f_i * (p.data - r_i))
        off += n


class CBPBackboneUtility:
    """Simplified Continual Backprop on backbone Linear/Conv units. Tracks EMA
    of |mean activation| per unit; on each step, with probability `reset_rate`,
    re-initialize the lowest-utility unit of each layer."""

    def __init__(self, model, reset_rate=1e-4, ema=0.99):
        self.reset_rate = float(reset_rate); self.ema = float(ema)
        # backbone layers only (exclude heads)
        if isinstance(model, MultiHeadMLP):
            self.linears = [m for m in model.backbone.modules()
                             if isinstance(m, nn.Linear)]
        elif isinstance(model, MultiHeadCNN):
            self.linears = [m for m in list(model.features.modules())
                             + list(model.trunk.modules())
                             if isinstance(m, nn.Linear)]
        else:
            self.linears = []
        # Init utility as zero tensor of correct out-shape
        self.util = {}
        for l in self.linears:
            self.util[id(l)] = torch.zeros(l.out_features)
        self._acts = {}
        self._hooks = []

    def register(self, model):
        for l in self.linears:
            def make_hook(lid):
                def h(_m, _in, out):
                    with torch.no_grad():
                        # out: (batch, features) for Linear
                        self._acts[lid] = out.detach().float().abs().mean(0).cpu()
                return h
            self._hooks.append(l.register_forward_hook(make_hook(id(l))))

    def update(self, seed: int, step_count: int):
        for l in self.linears:
            u = self.util[id(l)]
            a = self._acts.get(id(l), torch.zeros_like(u))
            self.util[id(l)] = self.ema * u + (1 - self.ema) * a
        g = torch.Generator().manual_seed(seed + step_count)
        for l in self.linears:
            if torch.rand(1, generator=g).item() < self.reset_rate:
                u = self.util[id(l)]
                lo = int(u.argmin())
                with torch.no_grad():
                    nn.init.kaiming_uniform_(l.weight[lo:lo + 1], a=5 ** 0.5)
                    if l.bias is not None: l.bias.data[lo] = 0.0
                u[lo] = 0.0

    def close(self):
        for h in self._hooks: h.remove()


# ---------------------------------------------------------------------------
# Strict matched-magnitude wrapping
# ---------------------------------------------------------------------------

def strict_renorm_to(delta: torch.Tensor, target_norm: torch.Tensor,
                      eps: float = 1e-12) -> torch.Tensor:
    return delta * (target_norm / (delta.norm() + eps))


# ---------------------------------------------------------------------------
# Directional controller step (two-sided coefficient c ∈ ℝ)
# ---------------------------------------------------------------------------

def apply_directional_step(sgd_delta: torch.Tensor,
                             u_hazard: Optional[torch.Tensor],
                             c: float,
                             ref_norm: torch.Tensor) -> torch.Tensor:
    """Two-sided controller step:
        Δθ' = Δθ - c · (Δθ · û) · û
    then EXACT-renorm to ref_norm.
      c = 0        → pure SGD (renormed).
      c > 0        → subtract û-component (steer AWAY from û_hazard).
      c < 0        → add û-component     (steer TOWARD û_hazard).
    """
    if u_hazard is None or c == 0.0:
        return strict_renorm_to(sgd_delta, ref_norm)
    u = u_hazard.to(sgd_delta.device)
    comp = float((sgd_delta @ u).item())
    d = sgd_delta - c * comp * u
    return strict_renorm_to(d, ref_norm)


# ---------------------------------------------------------------------------
# Sequence trainer for TIL — supports every arm we care about in e19
# ---------------------------------------------------------------------------

def train_til_sequence(
    model: nn.Module,
    tasks: List[dict],
    method: str,                          # "sgd" | "ewc" | "replay" | "cbp" | "controller"
    lr: float, epochs: int, batch: int,
    seed: int, device: torch.device,
    probes: list,
    probe_steps: int, probe_lr: float, probe_batch: int,
    ewc_lambda: float = 100.0,
    ewc_fisher_batch: int = 200,
    replay_memory_per_task: int = 200,
    replay_mix_ratio: float = 0.5,
    cbp_reset_rate: float = 1e-4,
    cbp_utility_ema: float = 0.99,
    controller=None,                      # `AdaptiveDirectionalController` or None
    controller_coeff: float = 0.0,        # c ∈ ℝ for the two-sided controller
    verbose: bool = True,
) -> dict:
    model = model.to(device)
    per_task_acc = []
    per_task_probe_ftd = []
    per_step_deltas = []            # applied ‖delta‖
    per_step_ref_norms = []         # plain-SGD ‖delta‖ at same θ
    per_step_deviations = []        # |applied - ref| / ref

    # Method state
    ewc_fisher = None; ewc_ref = None
    replay: Optional[ReplayMemoryTIL] = None
    cbp: Optional[CBPBackboneUtility] = None
    if method == "replay":
        replay = ReplayMemoryTIL(per_task=replay_memory_per_task)
    if method == "cbp":
        cbp = CBPBackboneUtility(model, reset_rate=cbp_reset_rate,
                                  ema=cbp_utility_ema)
        cbp.register(model)

    for t_idx, task in enumerate(tasks):
        if controller is not None:
            controller.begin_task(model, task["train_x"], task["train_y"])
        x, y = task["train_x"], task["train_y"]
        n = x.shape[0]
        steps_per_task = epochs * max(1, n // batch)
        opt = torch.optim.SGD(_backbone_and_head_params(model, t_idx), lr=lr)
        g = torch.Generator().manual_seed(seed + 17 * t_idx)
        step = 0
        while step < steps_per_task:
            for xb_cpu, yb_cpu in _iter(x, y, batch, g):
                theta_before = flat_params(model)
                xb = xb_cpu.to(device); yb = yb_cpu.to(device)

                # Optional replay: mix a batch from a past task and add its loss
                # (routed through its own head)
                replay_batch = None
                if method == "replay" and replay is not None and len(replay.buffers) > 0:
                    mem_n = int(round(xb.shape[0] * replay_mix_ratio))
                    if mem_n > 0:
                        replay_batch = replay.sample_from_random_past(
                            mem_n, seed=seed + 37 * step, device=device)

                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(til_forward(model, xb, t_idx), yb)
                if replay_batch is not None:
                    rx, ry, rt = replay_batch
                    loss = loss + F.cross_entropy(til_forward(model, rx, rt), ry)
                loss.backward()
                if method == "ewc" and ewc_fisher is not None:
                    _apply_ewc_penalty(model, ewc_fisher, ewc_ref, ewc_lambda)

                # Plain-SGD reference delta at this θ
                grad = _model_flat_grad(model)
                sgd_delta = -lr * grad
                ref_norm = sgd_delta.norm()

                if method == "controller" and controller is not None:
                    # Estimate û_hazard periodically
                    if (controller.u_hazard is None
                        or (step % controller.reest == 0)):
                        controller._reestimate(model, device, step)
                    u_haz = controller.u_hazard
                    if u_haz is not None and u_haz.dim() > 1:
                        # For the two-sided sweep we only use top-1
                        u_haz = u_haz[0]
                    applied_delta = apply_directional_step(
                        sgd_delta, u_haz, controller_coeff, ref_norm)
                else:
                    applied_delta = strict_renorm_to(sgd_delta, ref_norm)

                set_flat_params(model, theta_before + applied_delta)

                a_norm = float(applied_delta.norm().item())
                r_norm = float(ref_norm.item())
                per_step_deltas.append(a_norm)
                per_step_ref_norms.append(r_norm)
                per_step_deviations.append(abs(a_norm - r_norm) / max(r_norm, 1e-12))

                if cbp is not None:
                    cbp.update(seed=seed + 111 * step, step_count=step)
                step += 1
                if step >= steps_per_task: break

        # After-task bookkeeping
        if method == "ewc":
            ewc_fisher = _compute_ewc_fisher_til(
                model, tasks[:t_idx + 1], batch=ewc_fisher_batch,
                seed=seed + 33333 + t_idx, device=device)
            ewc_ref = flat_params(model).detach().clone()
        if method == "replay":
            replay.add_task(x, y, task_id=t_idx, seed=seed + 987 * (t_idx + 1))

        # Task acc (TIL: use head[t_idx])
        with torch.no_grad():
            xt = task["test_x"].to(device); yt = task["test_y"].to(device)
            ta = float((til_forward(model, xt, t_idx).argmax(-1) == yt)
                        .float().mean().item())
        # Probe FTD (fresh probe head, joint train w/ backbone)
        pf = probe_ftd_til(model, probes, probe_steps, probe_lr, probe_batch,
                            seed=seed, device=device)
        per_task_acc.append(ta); per_task_probe_ftd.append(pf)
        if verbose:
            print(f"  [{method}] task {t_idx} acc={ta:.3f} probeFTD={pf:.3f}")

    if cbp is not None: cbp.close()

    # Final per-task accuracies (TIL: each through its own head)
    final_accs = []
    with torch.no_grad():
        for t_idx, t in enumerate(tasks):
            xt = t["test_x"].to(device); yt = t["test_y"].to(device)
            final_accs.append(float((til_forward(model, xt, t_idx).argmax(-1) == yt)
                                     .float().mean().item()))
    n = len(per_task_acc)
    bwt = float(sum(final_accs[i] - per_task_acc[i] for i in range(n - 1))
                / max(n - 1, 1))

    max_dev = float(max(per_step_deviations)) if per_step_deviations else 0.0
    mean_dev = float(np.mean(per_step_deviations)) if per_step_deviations else 0.0
    p95_dev = float(np.percentile(per_step_deviations, 95)) if per_step_deviations else 0.0

    return {
        "method": method,
        "controller_coeff": float(controller_coeff) if method == "controller" else None,
        "per_task_acc": per_task_acc,
        "per_task_probe_ftd": per_task_probe_ftd,
        "final_accs": final_accs,
        "mean_final_acc": float(sum(final_accs) / len(final_accs)),
        "BWT": bwt,
        "matched_magnitude": {
            "mean_applied_norm": float(np.mean(per_step_deltas)),
            "mean_ref_norm": float(np.mean(per_step_ref_norms)),
            "max_deviation": max_dev,
            "mean_deviation": mean_dev,
            "p95_deviation": p95_dev,
        },
    }


# ---------------------------------------------------------------------------
# Sanity gate
# ---------------------------------------------------------------------------

def sanity_gate(result: dict, min_mean_acc: float,
                 min_tasks_above_chance: int, task_chance: float) -> dict:
    ok_mean = result["mean_final_acc"] >= min_mean_acc
    n_above = sum(1 for a in result["final_accs"] if a >= task_chance)
    ok_retention = n_above >= min_tasks_above_chance
    return {
        "mean_final_acc": result["mean_final_acc"],
        "min_mean_acc": min_mean_acc,
        "n_tasks_above_chance": n_above,
        "min_tasks_above_chance": min_tasks_above_chance,
        "task_chance": task_chance,
        "passes": bool(ok_mean and ok_retention),
    }
