"""Task-incremental (TIL) multi-head training loop with strict per-step
matched-magnitude enforcement, used by e19.

Every arm at every step:
  1. Computes plain SGD delta on the pure task cross-entropy at the current θ,
     using the head belonging to the current task. Records r_t = ‖plain_delta‖.
  2. Builds `arm_delta` (arm-specific: SGD, EWC penalty, replay-augmented, CBP
     reset, or directional controller with coefficient c).
  3. Applies `applied = arm_delta * (r_t / ‖arm_delta‖)`.
  4. Logs (r_t, ‖applied‖, deviation).

Frozen in D-043, v1.4 §14. See configs.preregistration.TIL_REGIME,
STRICT_MATCH, E19_TRADEOFF.
"""
from __future__ import annotations
import math
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import flat_params, set_flat_params, flat_grad


# --------------- helpers ---------------

def _iter(x, y, batch, gen):
    n = x.shape[0]
    perm = torch.randperm(n, generator=gen)
    for i in range(0, n, batch):
        idx = perm[i:i + batch]
        yield x[idx], y[idx]


def _acc_task(model, x, y, task_id: int, device, batch: int = 256) -> float:
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for i in range(0, x.shape[0], batch):
            xb = x[i:i + batch].to(device); yb = y[i:i + batch].to(device)
            correct += (model(xb, task_id).argmax(-1) == yb).sum().item()
            total += xb.shape[0]
    model.train()
    return correct / max(total, 1)


def _plain_task_grad(model, xb, yb, task_id: int) -> torch.Tensor:
    """Plain task cross-entropy gradient at current θ (no penalty/replay)."""
    model.zero_grad(set_to_none=True)
    logits = model(xb, task_id)
    F.cross_entropy(logits, yb).backward()
    return flat_grad(model)


def _renorm_to(delta: torch.Tensor, target_norm: float) -> torch.Tensor:
    n = delta.norm() + 1e-12
    return delta * (target_norm / n)


# --------------- probe FTD (permuted-input) ---------------

def build_permuted_probes(x_pool, y_pool, n_probes: int, subset: int,
                          seed: int, dataset: str) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Build permuted-input probes on the full data pool. Labels are original
    class indices (0..n_classes-1) mapped through the probe's OWN 2-way head.

    We treat each probe as an independent 2-way task: draw two random classes
    from the pool, take a subset of examples from those two classes, remap
    labels to {0,1}, and permute pixels. The probe head is a separate 2-way
    head attached to a deepcopy of the model's backbone (see probe_ftd_til).
    """
    rng = np.random.default_rng(seed)
    probes = []
    y_np = y_pool.numpy() if isinstance(y_pool, torch.Tensor) else y_pool
    classes = sorted(list(np.unique(y_np)))
    for p_id in range(n_probes):
        pair = tuple(rng.choice(classes, size=2, replace=False).tolist())
        mask = np.isin(y_np, pair)
        idx_all = np.where(mask)[0]
        pick = rng.choice(idx_all, size=min(subset, len(idx_all)), replace=False)
        x_sub = x_pool[pick]
        if x_sub.dim() == 2:
            perm = rng.permutation(x_sub.shape[1])
            x = x_sub[:, perm]
        else:
            n_, c, h, w = x_sub.shape
            perm = rng.permutation(h * w)
            x = x_sub.reshape(n_, c, h * w)[:, :, perm].reshape(n_, c, h, w)
        y_raw = y_np[pick]
        y_bin = torch.from_numpy(
            np.where(y_raw == pair[0], 0, 1).astype(np.int64)
        )
        probes.append((x, y_bin))
    return probes


def probe_ftd_til(model, probes, steps: int, lr: float, batch: int, seed: int,
                  device) -> float:
    """FTD = mean AUC-of-loss over probe steps. Each probe is a fresh 2-way
    task; we attach a fresh probe head to a copy of the backbone (freshly
    initialized head, backbone weights inherited from `model`)."""
    aucs = []
    for p_id, (x, y) in enumerate(probes):
        m = deepcopy(model)
        # Attach a fresh 2-way probe head; keep only backbone + one head slot.
        # We simply re-init head[0] and route through it for the probe.
        # Reset head 0 weights so probes are equal-footed across arms.
        with torch.no_grad():
            head = m.heads[0]
            nn.init.kaiming_uniform_(head.weight, a=math.sqrt(5))
            if head.bias is not None:
                fan_in = head.weight.shape[1]
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(head.bias, -bound, bound)
        m.to(device)
        opt = torch.optim.SGD(m.parameters(), lr=lr)
        g = torch.Generator().manual_seed(seed + 7919 * (p_id + 1))
        losses, step = [], 0
        while step < steps:
            for xb_cpu, yb_cpu in _iter(x, y, batch, g):
                xb = xb_cpu.to(device); yb = yb_cpu.to(device)
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(m(xb, 0), yb)
                loss.backward(); opt.step()
                losses.append(loss.item())
                step += 1
                if step >= steps: break
        aucs.append(sum(losses) / max(len(losses), 1))
        del m
    return float(sum(aucs) / max(len(aucs), 1))


# --------------- Replay buffer ---------------

class ReplayMemoryTIL:
    def __init__(self, per_task: int):
        self.per_task = per_task
        self.buffers: List[Tuple[torch.Tensor, torch.Tensor, int]] = []

    def add_task(self, x, y, task_id: int, seed: int):
        n = min(self.per_task, x.shape[0])
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(x.shape[0], generator=g)[:n]
        self.buffers.append((x[idx].clone(), y[idx].clone(), task_id))

    def sample(self, n_total: int, seed: int, device):
        if not self.buffers or n_total <= 0:
            return []
        per = max(1, n_total // len(self.buffers))
        g = torch.Generator().manual_seed(seed)
        picks = []
        for xb, yb, tid in self.buffers:
            idx = torch.randint(0, xb.shape[0], (per,), generator=g)
            picks.append((xb[idx].to(device, non_blocking=True),
                          yb[idx].to(device, non_blocking=True), tid))
        return picks


# --------------- CBP tracker ---------------

class CBPTrackerBackbone:
    """Utility-based unit reset across the *backbone* Linear layers only (heads
    are protected). Simplified version of Dohare et al. 2021."""

    def __init__(self, model: nn.Module, reset_rate: float, ema: float):
        self.reset_rate = reset_rate; self.ema = ema
        self.step_count = 0
        self.linears = [m for m in model.backbone.modules() if isinstance(m, nn.Linear)]
        self.utility = {id(l): torch.zeros(l.out_features) for l in self.linears}
        self._acts: Dict[int, torch.Tensor] = {}
        self.hooks: list = []

    def register(self):
        for l in self.linears:
            def make_hook(lid):
                def h(_m, _in, out):
                    self._acts[lid] = out.detach().mean(0).abs().cpu()
                return h
            self.hooks.append(l.register_forward_hook(make_hook(id(l))))

    def close(self):
        for h in self.hooks: h.remove()
        self.hooks = []

    def update_after_step(self, seed: int):
        for l in self.linears:
            u = self.utility[id(l)]
            a = self._acts.get(id(l), torch.zeros_like(u))
            if a.shape == u.shape:
                self.utility[id(l)] = self.ema * u + (1 - self.ema) * a
        self.step_count += 1
        g = torch.Generator().manual_seed(seed + self.step_count)
        for i, l in enumerate(self.linears):
            if torch.rand(1, generator=g).item() < self.reset_rate:
                u = self.utility[id(l)]
                lo = int(u.argmin())
                with torch.no_grad():
                    nn.init.kaiming_uniform_(l.weight[lo:lo + 1], a=math.sqrt(5))
                    if l.bias is not None: l.bias[lo] = 0.0
                if i + 1 < len(self.linears):
                    with torch.no_grad():
                        self.linears[i + 1].weight[:, lo] = 0.0
                u[lo] = 0.0


# --------------- Directional axis estimator ---------------

def build_permuted_aux(train_x, train_y, n: int, seed: int,
                       subset: int = 512) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        idx = rng.choice(train_x.shape[0],
                         size=min(subset, train_x.shape[0]), replace=False)
        x_sub = train_x[idx]
        if x_sub.dim() == 2:
            perm = rng.permutation(x_sub.shape[1])
            x = x_sub[:, perm]
        else:
            n_, c, h, w = x_sub.shape
            perm = rng.permutation(h * w)
            x = x_sub.reshape(n_, c, h * w)[:, :, perm].reshape(n_, c, h, w)
        y = train_y[idx].long()
        out.append((x, y))
    return out


def build_semantic_aux(pool_x, pool_y, held_out_classes, n: int, seed: int,
                       subset: int = 512) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    rng = np.random.default_rng(seed)
    y_np = pool_y.numpy() if isinstance(pool_y, torch.Tensor) else pool_y
    idx_all = np.where(np.isin(y_np, list(held_out_classes)))[0]
    if len(idx_all) == 0:
        return []
    out = []
    for i in range(n):
        pick = rng.choice(idx_all, size=min(subset, len(idx_all)), replace=False)
        y_raw = y_np[pick]
        # Remap to binary label 0/1 within the held-out pair (2-class only)
        pair = tuple(sorted(set(held_out_classes)))[:2]
        y_bin = torch.from_numpy(
            np.where(y_raw == pair[0], 0, 1).astype(np.int64)
        )
        x = pool_x[pick]
        out.append((x, y_bin))
    return out


def estimate_u_hazard(model, probes, task_id: int, batch: int, seed: int,
                      device) -> Optional[torch.Tensor]:
    """Mean-normalized probe gradient across aux probes, routed through
    head[task_id] so directions are commensurable with training updates."""
    if not probes:
        return None
    grads = []
    for i, (x, y) in enumerate(probes):
        m = deepcopy(model).to(device)
        g_rng = torch.Generator().manual_seed(seed + i)
        xb_cpu, yb_cpu = next(_iter(x, y, batch, g_rng))
        xb = xb_cpu.to(device); yb = yb_cpu.to(device)
        m.zero_grad(set_to_none=True)
        F.cross_entropy(m(xb, task_id), yb).backward()
        g_i = flat_grad(m).detach()
        grads.append(g_i / (g_i.norm() + 1e-12))
        del m
    G = torch.stack(grads, dim=0)
    u = G.mean(dim=0)
    return u / (u.norm() + 1e-12)


# --------------- Main sequence trainer ---------------

def train_sequence_til(
    model: nn.Module,
    tasks: List[dict],
    lr: float,
    epochs: int,
    batch: int,
    seed: int,
    device: torch.device,
    method: str,                  # "sgd" | "ewc" | "replay" | "cbp" | "controller"
    probes: list,
    probe_steps: int,
    probe_lr: float,
    probe_batch: int,
    hparams: Optional[dict] = None,
    controller_c: float = 0.0,
    aux_probes_perm_n: int = 3,
    aux_probes_sem_n: int = 3,
    reestimate_every: int = 50,
    aux_pool_x: Optional[torch.Tensor] = None,
    aux_pool_y: Optional[torch.Tensor] = None,
    held_out_pair: Optional[Tuple[int, int]] = None,
    verbose: bool = True,
) -> dict:
    hparams = hparams or {}
    per_task_acc_end = []          # acc on task t evaluated right after training it
    per_task_probe_ftd = []
    per_step_r = []                # r_t
    per_step_applied = []          # ‖applied_delta‖
    per_step_dev = []              # relative deviation

    # method state
    ewc_fisher: Optional[torch.Tensor] = None
    ewc_ref: Optional[torch.Tensor] = None
    replay: Optional[ReplayMemoryTIL] = None
    cbp: Optional[CBPTrackerBackbone] = None
    if method == "replay":
        replay = ReplayMemoryTIL(per_task=hparams.get("replay_memory_per_task", 200))
    if method == "cbp":
        cbp = CBPTrackerBackbone(model,
                                 reset_rate=hparams.get("cbp_reset_rate", 1e-4),
                                 ema=hparams.get("cbp_utility_ema", 0.99))
        cbp.register()

    for t_idx, task in enumerate(tasks):
        x, y = task["train_x"], task["train_y"]
        n = x.shape[0]
        steps = epochs * max(1, n // batch)
        g_rng = torch.Generator().manual_seed(seed + 17 * t_idx)

        # For controller: build task-local aux probes; u_hazard re-estimated
        aux_perm, aux_sem = None, None
        u_hazard = None
        if method == "controller":
            aux_perm = build_permuted_aux(x, y, aux_probes_perm_n,
                                          seed=seed + 101 + 31 * t_idx)
            if aux_pool_x is not None and held_out_pair is not None:
                aux_sem = build_semantic_aux(aux_pool_x, aux_pool_y,
                                             held_out_pair,
                                             aux_probes_sem_n,
                                             seed=seed + 202 + 31 * t_idx)
            else:
                aux_sem = []

        step = 0
        while step < steps:
            for xb_cpu, yb_cpu in _iter(x, y, batch, g_rng):
                xb = xb_cpu.to(device); yb = yb_cpu.to(device)

                # 1. plain SGD reference delta at current θ (task-only loss)
                plain_grad = _plain_task_grad(model, xb, yb, task_id=t_idx)
                plain_delta = -lr * plain_grad
                r_t = float(plain_delta.norm().item())

                # 2. build arm_delta
                if method == "sgd":
                    arm_delta = plain_delta.clone()

                elif method == "ewc" and ewc_fisher is not None:
                    # SGD + EWC penalty grad
                    penalty_grad = (hparams.get("ewc_lambda", 100.0)
                                    * ewc_fisher * (flat_params(model) - ewc_ref))
                    arm_delta = -lr * (plain_grad + penalty_grad)

                elif method == "ewc":
                    # First task: no penalty yet
                    arm_delta = plain_delta.clone()

                elif method == "replay":
                    # Concatenate current batch with per-past-task replay samples
                    if replay is not None and len(replay.buffers) > 0:
                        picks = replay.sample(
                            int(round(batch * hparams.get("replay_mix_ratio", 0.5))),
                            seed=seed + 37 * step, device=device)
                        # Compute combined gradient: current task loss on head[t_idx]
                        # + past-task loss on head[tid] for each replayed batch.
                        model.zero_grad(set_to_none=True)
                        loss = F.cross_entropy(model(xb, t_idx), yb)
                        for xr, yr, tid in picks:
                            loss = loss + F.cross_entropy(model(xr, tid), yr)
                        loss.backward()
                        combined_grad = flat_grad(model)
                        arm_delta = -lr * combined_grad
                    else:
                        arm_delta = plain_delta.clone()

                elif method == "cbp":
                    arm_delta = plain_delta.clone()

                elif method == "controller":
                    # Re-estimate u_hazard if needed (state-dependent)
                    if u_hazard is None or (step % reestimate_every == 0):
                        aux_all = list(aux_perm or []) + list(aux_sem or [])
                        u_hazard = estimate_u_hazard(
                            model, aux_all, task_id=t_idx, batch=batch,
                            seed=seed + 1_000_000 + t_idx * 997 + step,
                            device=device)
                    if u_hazard is not None:
                        comp = float((plain_delta @ u_hazard).item())
                        # applied_delta = plain + c * comp * u  (before renorm)
                        # c=-1 steers TOWARD u (plasticity-preserving hypothesis)
                        # c=0  is plain SGD
                        # c=+1 steers AWAY (v1.3 alpha=1)
                        arm_delta = plain_delta + controller_c * comp * u_hazard
                    else:
                        arm_delta = plain_delta.clone()

                else:
                    raise ValueError(method)

                # 3. Renormalize applied_delta to r_t exactly
                applied = _renorm_to(arm_delta, r_t)
                applied_norm = float(applied.norm().item())

                # 4. Apply update
                set_flat_params(model, flat_params(model) + applied)

                per_step_r.append(r_t)
                per_step_applied.append(applied_norm)
                per_step_dev.append(abs(applied_norm - r_t) / max(r_t, 1e-12))

                if cbp is not None:
                    # CBP requires a forward pass to have populated activations;
                    # we did one via _plain_task_grad. Update utility now.
                    cbp.update_after_step(seed=seed + 111 * step)

                step += 1
                if step >= steps: break

        # End-of-task bookkeeping
        if method == "ewc":
            # Recompute Fisher on tasks 0..t_idx using head-consistent samples
            ewc_fisher = _compute_ewc_fisher_til(
                model, tasks[:t_idx + 1],
                batch=hparams.get("ewc_fisher_batch", 200),
                seed=seed + 33333 + t_idx, device=device)
            ewc_ref = flat_params(model).detach().clone()
        if method == "replay":
            replay.add_task(x, y, task_id=t_idx, seed=seed + 987 * (t_idx + 1))

        # Metrics: acc on task just trained; probe FTD after this task
        ta = _acc_task(model, task["test_x"], task["test_y"],
                       task_id=t_idx, device=device)
        pf = probe_ftd_til(model, probes, probe_steps, probe_lr,
                           probe_batch, seed=seed, device=device)
        per_task_acc_end.append(ta)
        per_task_probe_ftd.append(pf)
        if verbose:
            print(f"[{method}{'/c='+str(controller_c) if method=='controller' else ''}] "
                  f"task {t_idx} acc={ta:.3f} probeFTD={pf:.3f}")

    if cbp is not None:
        cbp.close()

    # Final acc on every task using the correct head
    final_accs = [
        _acc_task(model, t["test_x"], t["test_y"], task_id=i, device=device)
        for i, t in enumerate(tasks)
    ]
    n_tasks = len(tasks)
    bwt = float(sum(final_accs[i] - per_task_acc_end[i] for i in range(n_tasks - 1))
                / max(n_tasks - 1, 1))
    r_arr = np.asarray(per_step_r, dtype=np.float64)
    a_arr = np.asarray(per_step_applied, dtype=np.float64)
    dev_arr = np.asarray(per_step_dev, dtype=np.float64)
    return {
        "method": method,
        "controller_c": (float(controller_c) if method == "controller" else None),
        "per_task_acc_end": per_task_acc_end,
        "per_task_probe_ftd": per_task_probe_ftd,
        "final_accs": final_accs,
        "mean_final_acc": float(np.mean(final_accs)),
        "BWT": bwt,
        "matched_magnitude": {
            "r_mean": float(np.mean(r_arr)),
            "applied_mean": float(np.mean(a_arr)),
            "relative_deviation_max": float(np.max(dev_arr)) if len(dev_arr) else 0.0,
            "relative_deviation_mean": float(np.mean(dev_arr)) if len(dev_arr) else 0.0,
            "relative_deviation_p95": float(np.quantile(dev_arr, 0.95))
                                       if len(dev_arr) else 0.0,
        },
    }


def _compute_ewc_fisher_til(model, tasks, batch, seed, device) -> torch.Tensor:
    D = sum(p.numel() for p in model.parameters())
    fisher = torch.zeros(D, device=device)
    count = 0
    for t_idx, task in enumerate(tasks):
        m = deepcopy(model).to(device)
        g_rng = torch.Generator().manual_seed(seed + 999 * (t_idx + 1))
        for xb_cpu, yb_cpu in _iter(task["train_x"], task["train_y"], batch, g_rng):
            xb = xb_cpu.to(device); yb = yb_cpu.to(device)
            m.zero_grad(set_to_none=True)
            logits = m(xb, t_idx)
            with torch.no_grad():
                y_hat = logits.argmax(-1)
            F.nll_loss(F.log_softmax(logits, -1), y_hat).backward()
            fisher = fisher + flat_grad(m).pow(2)
            count += 1
            break
        del m
    return fisher / max(count, 1)
