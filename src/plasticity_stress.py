"""Plasticity-stress regime: long permuted-MNIST (the canonical loss-of-plasticity
benchmark, Dohare et al. 2024) + mechanism diagnostics + no-memory baselines +
a matched-magnitude directional controller.

Why this regime (D-044): the v1.4 TIL regime was too easy — SGD reached 91% with
6% forgetting, leaving no plasticity gap for any method to recover. Loss of
plasticity only appears in LONG sequences with a shared head and warm-started
weights. Here each task is a fresh random pixel-permutation of MNIST (labels
unchanged → shared 10-way head, no head race). A rigid network learns *later*
tasks worse than *earlier* ones; the per-task learning-accuracy curve decays.
That decay is the disease we try to cure.

Diagnostics recorded per task (so we can SEE the disease):
  - task_test_acc:  accuracy on this task's held-out permuted test set
  - online_train_acc: mean pre-update minibatch accuracy during the task
  - dormant_frac: fraction of hidden ReLU units ~never active on a probe batch
  - weight_l2: L2 norm of backbone weights (rigidity correlate)
  - eff_rank: stable rank of penultimate features = ‖F‖_F² / ‖F‖_2²

Update rules (all share one loop):
  - sgd        : plain SGD
  - l2_init    : SGD + λ‖θ-θ0‖²         (Kumar et al. 2023) — magnitude method
  - shrink_pert: each task θ ← s·θ + σ·noise (Ash & Adams 2020) — magnitude method
  - cbp        : Continual Backprop unit resets (Dohare et al. 2024)
  - dir_control: OUR method — at MATCHED ‖Δθ‖, steer the SGD step orthogonal to
                 the weight-norm-growth (radial) direction:
                     Δθ' = Δθ - c·(Δθ·θ̂)·θ̂ ,  θ̂ = θ/‖θ‖ ,  then renorm to ‖Δθ‖.
                 c=0 → SGD; c=1 → step stays tangent to the ‖θ‖-sphere (no norm
                 growth); this fights rigidity DIRECTIONALLY at fixed step size —
                 the project's core thesis, aimed at the real mechanism.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PlasticMLP(nn.Module):
    """Plain ReLU MLP with a single shared 10-way head. Exposes penultimate
    features and hidden activations for plasticity diagnostics."""

    def __init__(self, input_dim: int = 784, hidden: int = 128, depth: int = 2,
                 n_classes: int = 10):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        prev = input_dim
        for _ in range(depth):
            self.hidden_layers.append(nn.Linear(prev, hidden))
            prev = hidden
        self.head = nn.Linear(prev, n_classes)

    def features(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        acts = []
        h = x
        for lin in self.hidden_layers:
            h = F.relu(lin(h))
            acts.append(h)
        return h, acts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.features(x)
        return self.head(h)


class PlasticCNN(nn.Module):
    """Small CNN for 28x28x1 inputs: 2 conv blocks + one FC hidden + head.
    Exposes penultimate FC features and the FC-hidden activation for diagnostics.
    A genuinely different architecture from PlasticMLP (convolutional, spatial)."""

    def __init__(self, base: int = 16, fc: int = 64, n_classes: int = 10):
        super().__init__()
        self.c1 = nn.Conv2d(1, base, 3, padding=1)
        self.c2 = nn.Conv2d(base, 2 * base, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(2 * base * 7 * 7, fc)   # 28->14->7
        self.head = nn.Linear(fc, n_classes)

    def features(self, x):
        # x: (N, 784) flattened -> (N,1,28,28)
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        h = self.pool(F.relu(self.c1(x)))
        h = self.pool(F.relu(self.c2(h)))
        h = h.reshape(h.shape[0], -1)
        f = F.relu(self.fc(h))
        return f, [f]                       # penultimate feats, hidden acts list

    def forward(self, x):
        f, _ = self.features(x)
        return self.head(f)


def backbone_flat(model: PlasticMLP) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in _backbone_params(model)])


def _backbone_params(model: PlasticMLP):
    ps = []
    for lin in model.hidden_layers:
        ps += list(lin.parameters())
    return ps


def _all_flat(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def _flat_grad(model: nn.Module) -> torch.Tensor:
    gs = []
    for p in model.parameters():
        gs.append((p.grad.detach().reshape(-1) if p.grad is not None
                    else torch.zeros(p.numel(), device=p.device)))
    return torch.cat(gs)


def _set_flat(model: nn.Module, flat: torch.Tensor) -> None:
    off = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[off:off + n].view_as(p))
        off += n


# ---------------------------------------------------------------------------
# Data: permuted-MNIST tasks (on-the-fly permutation to save memory)
# ---------------------------------------------------------------------------

class PermutedMNIST:
    def __init__(self, train_x, train_y, test_x, test_y,
                 n_tasks: int, seed: int, train_subset: int = 10000):
        # Flatten + normalize once
        self.tx = torch.from_numpy(train_x.reshape(len(train_x), -1)
                                    .astype(np.float32) / 255.0)
        self.ty = torch.from_numpy(train_y.astype(np.int64))
        self.vx = torch.from_numpy(test_x.reshape(len(test_x), -1)
                                    .astype(np.float32) / 255.0)
        self.vy = torch.from_numpy(test_y.astype(np.int64))
        rng = np.random.default_rng(seed)
        self.perms = [rng.permutation(self.tx.shape[1]) for _ in range(n_tasks)]
        # Fixed train subset indices per task (keeps the budget bounded)
        self.train_idx = [rng.choice(self.tx.shape[0],
                                      size=min(train_subset, self.tx.shape[0]),
                                      replace=False)
                           for _ in range(n_tasks)]
        self.n_tasks = n_tasks

    def task(self, t: int):
        p = self.perms[t]
        idx = self.train_idx[t]
        return (self.tx[idx][:, p], self.ty[idx],
                self.vx[:, p], self.vy)


class RotatedMNIST:
    """Rotated-MNIST task stream: each task rotates all images by a fixed
    task-specific angle (dependency-free rotation via affine grid sampling).
    Complements permuted MNIST: it changes the *input geometry* smoothly rather
    than destroying spatial structure, so the sequential-adaptation stress is
    of a different character."""

    def __init__(self, train_x, train_y, test_x, test_y,
                 n_tasks: int, seed: int, train_subset: int = 6000,
                 test_subset: int = 2000):
        self.trx = torch.from_numpy(train_x.astype(np.float32) / 255.0)   # (N,28,28)
        # flattened view so run_arm can read input_dim via stream.tx.shape[1]
        self.tx = self.trx.reshape(self.trx.shape[0], -1)
        self.ty_full = torch.from_numpy(train_y.astype(np.int64))
        self.vx = torch.from_numpy(test_x.astype(np.float32) / 255.0)
        self.vy_full = torch.from_numpy(test_y.astype(np.int64))
        rng = np.random.default_rng(seed)
        self.angles = rng.uniform(0.0, 180.0, size=n_tasks)
        self.train_idx = [rng.choice(self.trx.shape[0],
                                     size=min(train_subset, self.trx.shape[0]),
                                     replace=False) for _ in range(n_tasks)]
        te = min(test_subset, self.vx.shape[0])
        self.test_idx = rng.choice(self.vx.shape[0], size=te, replace=False)
        self.n_tasks = n_tasks

    @staticmethod
    def _rotate(imgs: torch.Tensor, angle_deg: float) -> torch.Tensor:
        import math
        n = imgs.shape[0]
        a = math.radians(angle_deg); ca, sa = math.cos(a), math.sin(a)
        theta = torch.tensor([[ca, -sa, 0.0], [sa, ca, 0.0]],
                             dtype=torch.float32).unsqueeze(0).repeat(n, 1, 1)
        grid = F.affine_grid(theta, (n, 1, 28, 28), align_corners=False)
        out = F.grid_sample(imgs.unsqueeze(1), grid, align_corners=False,
                            padding_mode="zeros").squeeze(1)
        return out.reshape(n, -1)

    def task(self, t: int):
        ang = float(self.angles[t]); idx = self.train_idx[t]
        trx = self._rotate(self.trx[idx], ang)
        vx = self._rotate(self.vx[self.test_idx], ang)
        return trx, self.ty_full[idx], vx, self.vy_full[self.test_idx]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@torch.no_grad()
def diagnostics(model: PlasticMLP, x_probe: torch.Tensor,
                device: torch.device, dormant_thresh: float = 1e-3) -> dict:
    model.eval()
    xb = x_probe.to(device)
    feat, acts = model.features(xb)
    # dormant fraction across all hidden units
    dormant, total = 0, 0
    for a in acts:
        mean_act = a.abs().mean(0)            # (hidden,)
        dormant += int((mean_act < dormant_thresh).sum().item())
        total += mean_act.numel()
    # weight L2 (backbone)
    wl2 = float(torch.cat([p.reshape(-1) for p in _backbone_params(model)]).norm().item())
    # stable rank of penultimate features
    F_ = feat - feat.mean(0, keepdim=True)
    try:
        s = torch.linalg.svdvals(F_)
        eff_rank = float((s.pow(2).sum() / (s.max().pow(2) + 1e-12)).item())
    except Exception:
        eff_rank = float("nan")
    model.train()
    return {"dormant_frac": dormant / max(total, 1),
            "weight_l2": wl2, "eff_rank": eff_rank}


# ---------------------------------------------------------------------------
# CBP (backbone unit resets) — reused idea, self-contained here
# ---------------------------------------------------------------------------

class CBPReset:
    def __init__(self, model: PlasticMLP, reset_rate: float = 1e-4, ema: float = 0.99):
        self.linears = list(model.hidden_layers)
        self.reset_rate = reset_rate; self.ema = ema
        self.util = {id(l): torch.zeros(l.out_features) for l in self.linears}
        self._acts = {}; self.hooks = []; self.step = 0

    def register(self, model):
        for l in self.linears:
            def mk(lid):
                def h(_m, _in, out):
                    self._acts[lid] = out.detach().float().abs().mean(0).cpu()
                return h
            self.hooks.append(l.register_forward_hook(mk(id(l))))

    def after_step(self, model, seed):
        for l in self.linears:
            u = self.util[id(l)]
            a = self._acts.get(id(l), torch.zeros_like(u))
            self.util[id(l)] = self.ema * u + (1 - self.ema) * a
        self.step += 1
        g = torch.Generator().manual_seed(seed + self.step)
        for li, l in enumerate(self.linears):
            if torch.rand(1, generator=g).item() < self.reset_rate:
                lo = int(self.util[id(l)].argmin())
                with torch.no_grad():
                    nn.init.kaiming_uniform_(l.weight[lo:lo+1], a=5 ** 0.5)
                    if l.bias is not None: l.bias.data[lo] = 0.0
                self.util[id(l)][lo] = 0.0

    def close(self):
        for h in self.hooks: h.remove()


# ---------------------------------------------------------------------------
# Directional controller: orthogonalize the step against weight-norm growth
# ---------------------------------------------------------------------------

def dir_control_step(sgd_delta: torch.Tensor, theta: torch.Tensor,
                     c: float) -> torch.Tensor:
    """Δθ' = Δθ - c·(Δθ·θ̂)·θ̂, then renorm to ‖Δθ‖ (matched magnitude).

    NOTE (D-044 finding): this ORTHOGONALIZES the step against the radial
    direction. Because a fixed-length step orthogonal to θ still grows ‖θ‖
    (Pythagoras), this does NOT arrest weight-norm growth — kept only as the
    ablation that demonstrates the geometric obstacle."""
    ref = sgd_delta.norm()
    tn = theta.norm()
    if tn < 1e-12 or c == 0.0:
        return sgd_delta
    u = theta / tn
    d = sgd_delta - c * (sgd_delta @ u) * u
    return d * (ref / (d.norm() + 1e-12))


def dir_inward_step(sgd_delta: torch.Tensor, theta: torch.Tensor,
                    gamma: float) -> torch.Tensor:
    """Norm-regulated directional step at MATCHED magnitude:
        Δθ' = normalize( Δθ - γ·‖Δθ‖·θ̂ ) · ‖Δθ‖ ,   θ̂ = θ/‖θ‖.
    Steers the fixed-length step partly INWARD (γ>0) to counteract weight-norm
    growth while preserving ‖Δθ‖ exactly. γ=0 → SGD; larger γ → stronger inward
    bias. This is the corrected directional norm-regulator."""
    ref = sgd_delta.norm()
    tn = theta.norm()
    if tn < 1e-12 or gamma == 0.0:
        return sgd_delta
    u = theta / tn
    d = sgd_delta - gamma * ref * u
    return d * (ref / (d.norm() + 1e-12))


# ---------------------------------------------------------------------------
# Training loop over the task sequence
# ---------------------------------------------------------------------------

def _iter(x, y, batch, gen):
    n = x.shape[0]
    perm = torch.randperm(n, generator=gen)
    for i in range(0, n, batch):
        idx = perm[i:i + batch]
        yield x[idx], y[idx]


def run_arm(stream: PermutedMNIST, method: str, *, lr: float, epochs: int,
            batch: int, seed: int, device: torch.device,
            hidden: int = 128, depth: int = 2, steps_per_task: int = 0,
            l2_init_lambda: float = 0.0, shrink: float = 0.6, perturb: float = 0.01,
            cbp_reset_rate: float = 1e-4, dir_coeff: float = 0.0,
            anr_target_norm: float = 2.0, anr_dormant_thresh: float = 1e-3,
            anr_mode: str = "full",
            cos_target: float = 1.0, norm_project_target: float = 6.0,
            diag_probe_size: int = 1000, verbose: bool = True,
            progress_cb=None) -> dict:
    # progress_cb (optional, read-only): called with a dict after every step and
    # after every task's evaluation. It only receives values that are already
    # computed; with progress_cb=None this function is unchanged.
    torch.manual_seed(seed)
    model = PlasticMLP(input_dim=stream.tx.shape[1], hidden=hidden, depth=depth,
                        n_classes=10).to(device)
    theta0 = _all_flat(model).clone()
    # Per-parameter init references (LIVE-graph safe) for l2_init penalty.
    theta0_params = [p.detach().clone() for p in model.parameters()]
    cbp = None
    if method == "cbp":
        cbp = CBPReset(model, reset_rate=cbp_reset_rate); cbp.register(model)

    task_test_acc, online_train_acc = [], []
    diag_dormant, diag_wl2, diag_rank = [], [], []
    applied_norms, ref_norms = [], []
    grad_aligns, radial_aligns = [], []   # real per-step alignment diagnostics

    for t in range(stream.n_tasks):
        trx, tr_y, vx, vy = stream.task(t)
        g = torch.Generator().manual_seed(seed + 17 * t)
        n = trx.shape[0]
        steps = steps_per_task if steps_per_task > 0 else epochs * max(1, n // batch)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        step = 0
        s0 = len(applied_norms)
        correct_online, seen_online = 0, 0
        while step < steps:
            for xb_cpu, yb_cpu in _iter(trx, tr_y, batch, g):
                xb = xb_cpu.to(device); yb = yb_cpu.to(device)
                theta_before = _all_flat(model)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                # online accuracy (pre-update prediction)
                with torch.no_grad():
                    correct_online += int((logits.argmax(-1) == yb).sum().item())
                    seen_online += xb.shape[0]
                loss = F.cross_entropy(logits, yb)
                if method == "l2_init" and l2_init_lambda > 0:
                    # FIX (D-045): penalty must flow into the gradient. Compute it
                    # on LIVE parameters, not a detached flat copy (the old bug
                    # zeroed the penalty, making l2_init read identical to SGD).
                    pen = sum(((p - p0.to(device)) ** 2).sum()
                              for p, p0 in zip(model.parameters(), theta0_params))
                    loss = loss + l2_init_lambda * pen
                loss.backward()
                grad = _flat_grad(model)
                sgd_delta = -lr * grad
                ref = sgd_delta.norm()
                ref_norms.append(float(ref.item()))
                if method == "dir_control":
                    applied = dir_control_step(sgd_delta, theta_before, dir_coeff)
                elif method == "dir_inward":
                    applied = dir_inward_step(sgd_delta, theta_before, dir_coeff)
                elif method == "random_dir":
                    # matched-magnitude random direction on the sphere
                    gg = torch.Generator(device="cpu").manual_seed(
                        seed * 100003 + step)
                    r = torch.randn(sgd_delta.shape, generator=gg).to(sgd_delta.device)
                    applied = r * (ref / (r.norm() + 1e-12))
                elif method == "angle":
                    # matched-magnitude update with cos(applied, sgd_delta)=cos_target:
                    #   d(a) = a*ghat + sqrt(1-a^2)*u_perp ; applied = d(a)*||sgd_delta||
                    ghat = sgd_delta / (ref + 1e-12)
                    gg = torch.Generator(device="cpu").manual_seed(
                        seed * 100003 + step)
                    u = torch.randn(sgd_delta.shape, generator=gg).to(sgd_delta.device)
                    u = u - (u @ ghat) * ghat
                    u = u / (u.norm() + 1e-12)
                    a = float(cos_target)
                    d = a * ghat + (max(0.0, 1.0 - a * a) ** 0.5) * u
                    applied = d * ref
                else:  # sgd, l2_init, shrink_pert, cbp, adaptive_norm_revive, norm_project
                    applied = sgd_delta
                # gradient- and radial-alignment of the APPLIED update (real diagnostics)
                grad_aligns.append(float((applied @ sgd_delta)
                                         / ((applied.norm() * ref) + 1e-12)))
                radial_aligns.append(float((applied @ theta_before)
                                           / ((applied.norm() * theta_before.norm()) + 1e-12)))
                _set_flat(model, theta_before + applied)
                # direct norm control: project theta onto the ||theta||<=target ball
                if method == "norm_project":
                    with torch.no_grad():
                        cur = _all_flat(model)
                        cn = cur.norm()
                        if float(cn.item()) > norm_project_target:
                            _set_flat(model, cur * (norm_project_target / (cn + 1e-12)))
                applied_norms.append(float(applied.norm().item()))
                if cbp is not None:
                    cbp.after_step(model, seed=seed + 131 * step)
                step += 1
                if progress_cb is not None:
                    progress_cb({"type": "step", "task": t, "step": step, "steps": steps})
                if step >= steps: break

        # Shrink & Perturb between tasks
        if method == "shrink_pert":
            with torch.no_grad():
                flat = _all_flat(model)
                gnoise = torch.randn(flat.shape, generator=torch.Generator()
                                     .manual_seed(seed + 71 * t)).to(device)
                _set_flat(model, shrink * flat + perturb * gnoise)

        # ADAPTIVE-NORM-REVIVE (our method, D-046): per-unit TARGETED norm cap +
        # dormancy revival between tasks. Unlike S&P's global blind shrink, this
        # (a) shrinks only units whose incoming-weight norm exceeds a per-unit
        # target (healthy small units untouched), and (b) hard-resets only the
        # units that have gone dormant, zeroing their outgoing weights so revival
        # doesn't disrupt the current function.
        # `anr_mode` selects the component(s) applied, enabling a component
        # ablation: 'full' = shrink + revive; 'shrink' = per-unit cap only;
        # 'revive' = dormant reset only.
        if method == "adaptive_norm_revive":
            do_shrink = anr_mode in ("full", "shrink")
            do_revive = anr_mode in ("full", "revive")
            with torch.no_grad():
                # dormancy from a probe batch
                _, acts = model.features(vx[:diag_probe_size].to(device))
                for li, lin in enumerate(model.hidden_layers):
                    W = lin.weight.data                      # (out, in)
                    if do_shrink:
                        # (a) targeted per-unit norm cap (only shrink over-grown units)
                        row_norm = W.norm(dim=1, keepdim=True)   # (out, 1)
                        scale = torch.clamp(anr_target_norm / (row_norm + 1e-12), max=1.0)
                        lin.weight.data.mul_(scale)
                        if lin.bias is not None:
                            lin.bias.data.mul_(scale.squeeze(1))
                    if do_revive:
                        # (b) revive dormant units
                        mean_act = acts[li].abs().mean(0)        # (out,)
                        dormant = mean_act < anr_dormant_thresh
                        if bool(dormant.any()):
                            new_w = torch.empty_like(W[dormant])
                            nn.init.kaiming_uniform_(new_w, a=5 ** 0.5)
                            lin.weight.data[dormant] = new_w
                            if lin.bias is not None:
                                lin.bias.data[dormant] = 0.0
                            nxt = (model.hidden_layers[li + 1]
                                   if li + 1 < len(model.hidden_layers) else model.head)
                            nxt.weight.data[:, dormant] = 0.0

        # Metrics
        with torch.no_grad():
            xv = vx.to(device); yv = vy.to(device)
            # batch eval to bound memory
            preds = []
            for i in range(0, xv.shape[0], 2000):
                preds.append(model(xv[i:i+2000]).argmax(-1))
            acc = float((torch.cat(preds) == yv).float().mean().item())
        task_test_acc.append(acc)
        online_train_acc.append(correct_online / max(seen_online, 1))
        d = diagnostics(model, vx[:diag_probe_size], device)
        diag_dormant.append(d["dormant_frac"]); diag_wl2.append(d["weight_l2"])
        diag_rank.append(d["eff_rank"])
        if progress_cb is not None:
            progress_cb({"type": "task", "task": t, "n_tasks": stream.n_tasks,
                         "acc": acc, "online_acc": online_train_acc[-1],
                         "weight_l2": d["weight_l2"], "dormant": d["dormant_frac"],
                         "eff_rank": d["eff_rank"],
                         "applied_norms": applied_norms[s0:], "ref_norms": ref_norms[s0:],
                         "grad_aligns": grad_aligns[s0:], "radial_aligns": radial_aligns[s0:],
                         "model": model})
        if verbose and (t % 5 == 0 or t == stream.n_tasks - 1):
            print(f"  [{method}] task {t:3d} test_acc={acc:.3f} "
                  f"online={online_train_acc[-1]:.3f} dormant={d['dormant_frac']:.3f} "
                  f"wl2={d['weight_l2']:.1f} rank={d['eff_rank']:.1f}")

    if cbp is not None: cbp.close()

    # Plasticity decay = mean(first 20%) - mean(last 20%) of task_test_acc
    k = max(1, stream.n_tasks // 5)
    decay = float(np.mean(task_test_acc[:k]) - np.mean(task_test_acc[-k:]))
    max_dev = (float(np.max(np.abs(np.array(applied_norms) - np.array(ref_norms))
                             / (np.array(ref_norms) + 1e-12)))
               if applied_norms else 0.0)
    return {
        "method": method, "dir_coeff": dir_coeff if method == "dir_control" else None,
        "task_test_acc": task_test_acc,
        "online_train_acc": online_train_acc,
        "diag_dormant": diag_dormant, "diag_weight_l2": diag_wl2,
        "diag_eff_rank": diag_rank,
        "mean_acc": float(np.mean(task_test_acc)),
        "early_acc": float(np.mean(task_test_acc[:k])),
        "late_acc": float(np.mean(task_test_acc[-k:])),
        "plasticity_decay": decay,
        "matched_max_dev": max_dev,
        # --- Cor. 1 telescoping audit (read-only diagnostics; computed on the
        # SAME global _all_flat vector the matched step operates on). For an
        # exactly-orthogonal arm, thetaT_sq - theta0_sq == sum_applied_sq. ---
        "theta0_sq": float((theta0.double() ** 2).sum().item()),
        "thetaT_sq": float((_all_flat(model).double() ** 2).sum().item()),
        "sum_applied_sq": float(np.sum(np.asarray(applied_norms, dtype=np.float64) ** 2))
                           if applied_norms else 0.0,
        "grad_align_mean": float(np.mean(grad_aligns)) if grad_aligns else float("nan"),
        "radial_align_mean": float(np.mean(radial_aligns)) if radial_aligns else float("nan"),
        "layer_norms_final": [float(l.weight.data.norm().item()) for l in model.hidden_layers]
                             + [float(model.head.weight.data.norm().item())],
    }


# ---------------------------------------------------------------------------
# CNN runner (second architecture). Reuses the flat-parameter update rules;
# supports sgd, l2_init, shrink_pert, dir_control, dir_inward, random_dir,
# norm_project, angle. (CBP/ANR are MLP-per-unit specific and omitted here.)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _cnn_diag(model, x_probe, device, dormant_thresh=1e-3):
    model.eval()
    feat, acts = model.features(x_probe.to(device))
    a = acts[0].abs().mean(0)
    dormant = float((a < dormant_thresh).float().mean().item())
    wl2 = float(torch.cat([p.reshape(-1) for p in model.parameters()]).norm().item())
    Fc = feat - feat.mean(0, keepdim=True)
    try:
        s = torch.linalg.svdvals(Fc)
        eff = float((s.pow(2).sum() / (s.max().pow(2) + 1e-12)).item())
    except Exception:
        eff = float("nan")
    model.train()
    return dormant, wl2, eff


def run_arm_cnn(stream, method, *, lr, epochs, batch, seed, device,
                steps_per_task=0, base=16, fc=64, l2_init_lambda=0.0,
                shrink=0.7, perturb=0.01, dir_coeff=0.0, cos_target=1.0,
                norm_project_target=6.0, diag_probe_size=1000, verbose=False,
                progress_cb=None):
    # progress_cb: same read-only hook as in run_arm.
    torch.manual_seed(seed)
    model = PlasticCNN(base=base, fc=fc, n_classes=10).to(device)
    theta0 = [p.detach().clone() for p in model.parameters()]
    task_acc, diag_wl2, diag_dorm, diag_rank = [], [], [], []
    applied_norms, ref_norms, grad_aligns, radial_aligns = [], [], [], []
    for t in range(stream.n_tasks):
        trx, tr_y, vx, vy = stream.task(t)
        g = torch.Generator().manual_seed(seed + 17 * t)
        n = trx.shape[0]
        steps = steps_per_task if steps_per_task > 0 else epochs * max(1, n // batch)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        step = 0
        s0 = len(applied_norms)
        while step < steps:
            for xb_cpu, yb_cpu in _iter(trx, tr_y, batch, g):
                xb = xb_cpu.to(device); yb = yb_cpu.to(device)
                theta_before = _all_flat(model)
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(xb), yb)
                if method == "l2_init" and l2_init_lambda > 0:
                    loss = loss + l2_init_lambda * sum(
                        ((p - p0.to(device)) ** 2).sum()
                        for p, p0 in zip(model.parameters(), theta0))
                loss.backward()
                grad = _flat_grad(model)
                sgd_delta = -lr * grad
                ref = sgd_delta.norm()
                ref_norms.append(float(ref.item()))
                if method == "dir_control":
                    applied = dir_control_step(sgd_delta, theta_before, dir_coeff)
                elif method == "dir_inward":
                    applied = dir_inward_step(sgd_delta, theta_before, dir_coeff)
                elif method == "random_dir":
                    gg = torch.Generator(device="cpu").manual_seed(seed * 100003 + step)
                    r = torch.randn(sgd_delta.shape, generator=gg).to(sgd_delta.device)
                    applied = r * (ref / (r.norm() + 1e-12))
                elif method == "angle":
                    ghat = sgd_delta / (ref + 1e-12)
                    gg = torch.Generator(device="cpu").manual_seed(seed * 100003 + step)
                    u = torch.randn(sgd_delta.shape, generator=gg).to(sgd_delta.device)
                    u = u - (u @ ghat) * ghat; u = u / (u.norm() + 1e-12)
                    a = float(cos_target)
                    applied = (a * ghat + (max(0.0, 1 - a * a) ** 0.5) * u) * ref
                else:
                    applied = sgd_delta
                grad_aligns.append(float((applied @ sgd_delta) / ((applied.norm() * ref) + 1e-12)))
                radial_aligns.append(float((applied @ theta_before) / ((applied.norm() * theta_before.norm()) + 1e-12)))
                _set_flat(model, theta_before + applied)
                if method == "norm_project":
                    cur = _all_flat(model); cn = cur.norm()
                    if float(cn.item()) > norm_project_target:
                        _set_flat(model, cur * (norm_project_target / (cn + 1e-12)))
                applied_norms.append(float(applied.norm().item()))
                step += 1
                if progress_cb is not None:
                    progress_cb({"type": "step", "task": t, "step": step, "steps": steps})
                if step >= steps: break
        if method == "shrink_pert":
            flat = _all_flat(model)
            gn = torch.randn(flat.shape, generator=torch.Generator().manual_seed(seed + 71 * t)).to(device)
            _set_flat(model, shrink * flat + perturb * gn)
        with torch.no_grad():
            xv = vx.to(device); yv = vy.to(device)
            preds = torch.cat([model(xv[i:i+2000]).argmax(-1) for i in range(0, xv.shape[0], 2000)])
            acc = float((preds == yv).float().mean().item())
        task_acc.append(acc)
        dm, wl, ef = _cnn_diag(model, vx[:diag_probe_size], device)
        diag_dorm.append(dm); diag_wl2.append(wl); diag_rank.append(ef)
        if progress_cb is not None:
            progress_cb({"type": "task", "task": t, "n_tasks": stream.n_tasks,
                         "acc": acc, "online_acc": None,
                         "weight_l2": wl, "dormant": dm, "eff_rank": ef,
                         "applied_norms": applied_norms[s0:], "ref_norms": ref_norms[s0:],
                         "grad_aligns": grad_aligns[s0:], "radial_aligns": radial_aligns[s0:],
                         "model": model})
    import numpy as _np
    k = max(1, stream.n_tasks // 5)
    decay = float(_np.mean(task_acc[:k]) - _np.mean(task_acc[-k:]))
    md = (float(_np.max(_np.abs(_np.array(applied_norms) - _np.array(ref_norms)) / (_np.array(ref_norms) + 1e-12)))
          if applied_norms else 0.0)
    return {"method": method, "task_test_acc": task_acc, "mean_acc": float(_np.mean(task_acc)),
            "plasticity_decay": decay, "diag_weight_l2": diag_wl2, "diag_dormant": diag_dorm,
            "diag_eff_rank": diag_rank, "matched_max_dev": md,
            "grad_align_mean": float(_np.mean(grad_aligns)) if grad_aligns else float("nan"),
            "radial_align_mean": float(_np.mean(radial_aligns)) if radial_aligns else float("nan")}
