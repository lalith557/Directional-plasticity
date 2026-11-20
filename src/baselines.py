"""Reusable continual-learning baselines used by E13's Pareto comparison.

Every baseline exposes the same interface:
  train_sequence(model, tasks, hparams, device, probes, probe_cfg, seed, past_axis=None)
    -> {"per_task_acc": [...], "per_task_probe_ftd": [...], "final_accs": [...], "BWT": float,
        "mean_final_acc": float, "per_step_delta_norms": [...]}

Baselines:
  - sgd:    plain SGD (control)
  - ewc:    Elastic Weight Consolidation (Kirkpatrick et al. 2017)
  - replay: experience replay from a per-task memory buffer
  - cbp:    Continual Backprop-style utility-based unit resets (Dohare et al. 2021)
  - past_axis_controller: E13's own controller (project + rescale along û_past by
                          a tunable coefficient at exact matched ||dtheta||)

`train_sequence` is deliberately shape-agnostic and works with any torch.nn.Module
whose forward returns logits (batch, n_classes) and takes (batch, ...) input tensor.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Callable, Optional, List, Tuple

from .models import flat_params, set_flat_params, flat_grad
from .constraint import project_out


def _to(x, device): return x.to(device, non_blocking=True) if device is not None else x


def _iter(x, y, batch, gen):
    n = x.shape[0]
    perm = torch.randperm(n, generator=gen)
    for i in range(0, n, batch):
        idx = perm[i:i + batch]
        yield x[idx], y[idx]


def _acc(model, x, y, device, batch=256):
    with torch.no_grad():
        total, n = 0, 0
        for i in range(0, x.shape[0], batch):
            xb = _to(x[i:i + batch], device); yb = _to(y[i:i + batch], device)
            total += (model(xb).argmax(-1) == yb).sum().item()
            n += xb.shape[0]
        return total / max(n, 1)


def _loss(model, x, y, device, batch=256):
    with torch.no_grad():
        total, n = 0.0, 0
        for i in range(0, x.shape[0], batch):
            xb = _to(x[i:i + batch], device); yb = _to(y[i:i + batch], device)
            total += F.cross_entropy(model(xb), yb, reduction="sum").item()
            n += xb.shape[0]
        return total / max(n, 1)


def probe_ftd_generic(model, probes: list, probe_steps: int, probe_lr: float,
                      probe_batch: int, seed: int, device: torch.device) -> float:
    """AUC-of-loss over probe_steps SGD steps, averaged over probes."""
    aucs = []
    for p_id, (x, y) in enumerate(probes):
        m = deepcopy(model).to(device)
        opt = torch.optim.SGD(m.parameters(), lr=probe_lr)
        g = torch.Generator().manual_seed(seed + 7919 * (p_id + 1))
        losses, step = [], 0
        while step < probe_steps:
            for xb_cpu, yb_cpu in _iter(x, y, probe_batch, g):
                xb = _to(xb_cpu, device); yb = _to(yb_cpu, device)
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(m(xb), yb)
                loss.backward(); opt.step()
                losses.append(loss.item())
                step += 1
                if step >= probe_steps: break
        aucs.append(sum(losses) / max(len(losses), 1))
        del m
    return float(sum(aucs) / max(len(aucs), 1))


# ---------------- EWC helpers ----------------

def compute_ewc_fisher(model, tasks, batch: int, seed: int, device: torch.device) -> torch.Tensor:
    """Empirical Fisher = mean of squared gradients on samples from each past task."""
    D = sum(p.numel() for p in model.parameters())
    fisher = torch.zeros(D, device=device)
    count = 0
    for t_idx, task in enumerate(tasks):
        m = deepcopy(model).to(device)
        g = torch.Generator().manual_seed(seed + 999 * (t_idx + 1))
        for xb_cpu, yb_cpu in _iter(task["train_x"], task["train_y"], batch, g):
            xb = _to(xb_cpu, device); yb = _to(yb_cpu, device)
            m.zero_grad(set_to_none=True)
            logits = m(xb)
            with torch.no_grad():
                y_hat = logits.argmax(-1)
            F.nll_loss(F.log_softmax(logits, -1), y_hat).backward()
            fisher = fisher + flat_grad(m).pow(2)
            count += 1
            break  # 1 batch per task is enough for the standard empirical Fisher
    return fisher / max(count, 1)


def apply_ewc_penalty(model, fisher: torch.Tensor, theta_ref: torch.Tensor,
                      lam: float) -> None:
    off = 0
    for p in model.parameters():
        n = p.numel()
        f_i = fisher[off:off + n].view_as(p)
        r_i = theta_ref[off:off + n].view_as(p)
        if p.grad is None: p.grad = torch.zeros_like(p)
        p.grad.data.add_(lam * f_i * (p.data - r_i))
        off += n


# ---------------- Replay memory ----------------

class ReplayMemory:
    """Ring buffer with `per_task` samples per task; can sample mixed minibatches."""
    def __init__(self, per_task: int):
        self.per_task = per_task
        self.buffers: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def add_task(self, x, y, seed: int):
        n = min(self.per_task, x.shape[0])
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(x.shape[0], generator=g)[:n]
        self.buffers.append((x[idx].clone(), y[idx].clone()))

    def sample(self, n: int, seed: int, device):
        if not self.buffers or n <= 0: return None
        g = torch.Generator().manual_seed(seed)
        # Pool across all task buffers
        all_x = torch.cat([b[0] for b in self.buffers], dim=0)
        all_y = torch.cat([b[1] for b in self.buffers], dim=0)
        idx = torch.randint(0, all_x.shape[0], (n,), generator=g)
        return _to(all_x[idx], device), _to(all_y[idx], device)


# ---------------- Continual Backprop utility ----------------

class UnitUtilityTracker:
    """Rolling utility per hidden unit; on each step some fraction of low-utility
    units are re-initialized in place.

    Simplified CBP: track EMA of |activation * output-weight| per unit; sort;
    reset the K lowest with probability `reset_rate` per step.
    """
    def __init__(self, model: nn.Module, reset_rate: float, ema: float):
        self.reset_rate = reset_rate
        self.ema = ema
        self.linears = [m for m in model.modules() if isinstance(m, nn.Linear)][:-1]
        self.utility = {id(l): torch.zeros(l.out_features) for l in self.linears}
        self.step_count = 0

    def register(self, model: nn.Module):
        self._acts = {}
        self.hooks = []
        for l in self.linears:
            def make_hook(lid):
                def h(_m, _in, out):
                    self._acts[lid] = out.detach().mean(0).abs().cpu()
                return h
            self.hooks.append(l.register_forward_hook(make_hook(id(l))))

    def update_after_step(self, model: nn.Module, seed: int):
        # Update utility from most recent forward
        for l in self.linears:
            u = self.utility[id(l)]
            a = self._acts.get(id(l), torch.zeros_like(u))
            self.utility[id(l)] = self.ema * u + (1 - self.ema) * a
        self.step_count += 1
        # With prob = reset_rate, reset the lowest-utility unit in each layer
        g = torch.Generator().manual_seed(seed + self.step_count)
        for l in self.linears:
            if torch.rand(1, generator=g).item() < self.reset_rate:
                u = self.utility[id(l)]
                lo = int(u.argmin())
                with torch.no_grad():
                    nn.init.kaiming_uniform_(l.weight[lo:lo+1], a=5**0.5)
                    if l.bias is not None: l.bias[lo] = 0.0
                # Reset outgoing weights of next layer for that unit
                idx_in_next = self.linears.index(l) + 1
                if idx_in_next < len(self.linears):
                    l_next = self.linears[idx_in_next]
                    with torch.no_grad():
                        l_next.weight[:, lo] = 0.0
                u[lo] = 0.0

    def close(self):
        for h in self.hooks: h.remove()


# ---------------- Main sequence trainer ----------------

def train_sequence(
    model: nn.Module,
    tasks: list,
    lr: float,
    epochs: int,
    batch: int,
    seed: int,
    device: torch.device,
    method: str,
    probes: Optional[list] = None,
    probe_steps: int = 200,
    probe_lr: float = 5e-3,
    probe_batch: int = 128,
    hparams: Optional[dict] = None,
    past_axis: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> dict:
    """Sequentially train on all tasks and return per-task metrics + norm log."""
    hparams = hparams or {}
    per_task_acc, per_task_probe_ftd, per_step_norms = [], [], []
    # per-method state
    ewc_fisher = None; ewc_ref = None
    replay: Optional[ReplayMemory] = None
    cbp: Optional[UnitUtilityTracker] = None
    if method == "replay":
        replay = ReplayMemory(per_task=hparams.get("replay_memory_per_task", 200))
    if method == "cbp":
        cbp = UnitUtilityTracker(model,
                                 reset_rate=hparams.get("cbp_reset_rate", 1e-4),
                                 ema=hparams.get("cbp_utility_ema", 0.99))
        cbp.register(model)

    for t_idx, task in enumerate(tasks):
        x, y = task["train_x"], task["train_y"]
        n = x.shape[0]
        steps = epochs * max(1, n // batch)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        g = torch.Generator().manual_seed(seed + 17 * t_idx)
        step = 0
        while step < steps:
            for xb_cpu, yb_cpu in _iter(x, y, batch, g):
                theta_before = flat_params(model)
                xb = _to(xb_cpu, device); yb = _to(yb_cpu, device)
                # optional replay mixin
                if method == "replay" and replay is not None and len(replay.buffers) > 0:
                    mem_n = int(round(xb.shape[0] * hparams.get("replay_mix_ratio", 0.5)))
                    if mem_n > 0:
                        mem = replay.sample(mem_n, seed=seed + 37 * step, device=device)
                        if mem is not None:
                            xb = torch.cat([xb, mem[0]], dim=0)
                            yb = torch.cat([yb, mem[1]], dim=0)
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
                if method == "ewc" and ewc_fisher is not None:
                    apply_ewc_penalty(model, ewc_fisher, ewc_ref,
                                      lam=hparams.get("ewc_lambda", 100.0))
                grad = flat_grad(model)
                sgd_delta = -lr * grad
                orig_norm = sgd_delta.norm()
                # past-axis controller family
                if method == "past_axis_controller" and past_axis is not None:
                    c = float(hparams.get("axis_coeff", 0.0))  # -1..+1
                    u = past_axis / (past_axis.norm() + 1e-12)
                    comp_scalar = (sgd_delta @ u).item()
                    # Δθ' = Δθ + c * (Δθ·u) u
                    apply_delta = sgd_delta + c * comp_scalar * u
                    # Renormalize exactly to preserve ||dtheta||
                    apply_delta = apply_delta * (orig_norm / (apply_delta.norm() + 1e-12))
                else:
                    apply_delta = sgd_delta
                set_flat_params(model, theta_before + apply_delta)
                per_step_norms.append(float(apply_delta.norm().item()))
                if cbp is not None:
                    cbp.update_after_step(model, seed=seed + 111 * step)
                step += 1
                if step >= steps: break
        # After each task, book-keeping for method state
        if method == "ewc":
            # accumulate/refresh fisher & reference after this task
            ewc_fisher = compute_ewc_fisher(model, tasks[:t_idx + 1],
                                            batch=hparams.get("ewc_fisher_batch", 200),
                                            seed=seed + 33333 + t_idx, device=device)
            ewc_ref = flat_params(model).detach().clone()
        if method == "replay":
            replay.add_task(x, y, seed=seed + 987 * (t_idx + 1))

        # Evaluate on this task, and probe FTD
        ta = _acc(model, task["test_x"], task["test_y"], device=device)
        pf = (probe_ftd_generic(model, probes, probe_steps, probe_lr,
                                 probe_batch, seed=seed, device=device)
              if probes is not None else None)
        per_task_acc.append(ta)
        per_task_probe_ftd.append(pf)
        if verbose:
            print(f"[baseline:{method}] task {t_idx} "
                  f"acc={ta:.3f}"
                  + (f" probeFTD={pf:.3f}" if pf is not None else ""))

    if cbp is not None: cbp.close()
    final_accs = [_acc(model, t["test_x"], t["test_y"], device=device) for t in tasks]
    n = len(per_task_acc)
    bwt = float(sum(final_accs[i] - per_task_acc[i] for i in range(n - 1)) / max(n - 1, 1))
    return {
        "method": method,
        "per_task_acc": per_task_acc,
        "per_task_probe_ftd": per_task_probe_ftd,
        "final_accs": final_accs,
        "mean_final_acc": float(sum(final_accs) / len(final_accs)),
        "BWT": bwt,
        "per_step_delta_norms": per_step_norms,
    }
