"""Canonical probe taxonomy for the Directional Plasticity project (v1.3).

Frozen alongside D-034. Six families, each with a `strength` parameter that
smoothly varies alignment `cos(g_ref_train, g_ref_probe)` — designed to close
W1 (probe inconsistency), W2 (two-cluster correlation), W3 (all-positive-β).

Every probe uses ORIGINAL class labels on the shared classification head. No
binary remapping (that was e10's bug — see D-034). Head consistency is checked
by the `labels_are_original` assertion.

The six families:

  1. structural_permute (8 probes)  — permute a fraction `p` ∈ [0.05, 1.0] of pixels
                                       within each image. p=0 is identity (highest align);
                                       p=1 is full permutation (lowest positive align).
  2. semantic_novel_class (4)       — draw images from held-out classes; keep original
                                       labels. Align ≈ +1 (shared head, similar geometry).
  3. noise (4)                      — add Gaussian noise σ ∈ [0.02, 0.5] to inputs.
  4. input_scale (4)                — multiplicative scaling s ∈ [0.5, 1.5].
  5. label_shift (6)                — target = (correct + k) mod C for k ∈ 1..C-1
                                       (subset of 6). Some k produce anti-aligned g_probe.
  6. anti_class (3)                 — target = argmin_c p(c | x, θ_ref); the label that the
                                       network is MOST WRONG about. g_probe points AGAINST
                                       current-task gradient by construction.

All probe (x, y) tensors are on CPU; move to device inside probe_ftd.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

from .models import flat_grad


@dataclass
class Probe:
    x: torch.Tensor       # (N, ...) inputs, CPU
    y: torch.Tensor       # (N,) int64 labels
    name: str             # e.g. "structural_permute_p=0.5"
    family: str           # taxonomy family name
    strength: float       # scalar characterizing the probe within its family


# ---------------- Image-shape-aware helpers ----------------

def _permute_pixels(x: torch.Tensor, fraction: float, rng: np.random.Generator
                    ) -> torch.Tensor:
    """Permute a `fraction` of pixel positions (per-image structure). fraction=1
    is a full permutation; fraction=0 is identity.

    Works for both flat (N, D) and image (N, C, H, W) tensors.
    """
    if fraction <= 0:
        return x.clone()
    if x.dim() == 2:
        n_feat = x.shape[1]
        n_perm = max(1, int(round(fraction * n_feat)))
        idx = rng.choice(n_feat, size=n_perm, replace=False)
        perm_of_idx = rng.permutation(idx)
        out = x.clone()
        out[:, idx] = out[:, perm_of_idx]
        return out
    elif x.dim() == 4:
        n, c, h, w = x.shape
        n_feat = h * w
        n_perm = max(1, int(round(fraction * n_feat)))
        idx = rng.choice(n_feat, size=n_perm, replace=False)
        perm_of_idx = rng.permutation(idx)
        xf = x.reshape(n, c, n_feat).clone()
        xf[:, :, idx] = xf[:, :, perm_of_idx]
        return xf.reshape(n, c, h, w)
    else:
        raise ValueError(f"unexpected x.dim()={x.dim()}")


# ---------------- Family builders ----------------

def _build_structural_permute(tx, ty, n: int, rng: np.random.Generator,
                               subset: int) -> List[Probe]:
    probes = []
    fractions = np.linspace(0.05, 1.0, n).tolist()
    for i, f in enumerate(fractions):
        idx = rng.choice(tx.shape[0], size=min(subset, tx.shape[0]), replace=False)
        x_sub = tx[idx]
        x = _permute_pixels(x_sub, fraction=float(f), rng=rng)
        y = ty[idx].long()
        probes.append(Probe(x=x, y=y,
                            name=f"structural_permute_f={f:.2f}",
                            family="structural_permute", strength=float(f)))
    return probes


def _build_semantic_novel_class(tx, ty, n: int, rng: np.random.Generator,
                                 subset: int, held_out_classes: Tuple[int, ...]
                                 ) -> List[Probe]:
    y_np = ty.numpy() if isinstance(ty, torch.Tensor) else ty
    pool = np.where(np.isin(y_np, list(held_out_classes)))[0]
    probes = []
    for i in range(n):
        pick = rng.choice(pool, size=min(subset, len(pool)), replace=False)
        x = tx[pick]
        y_pick_np = y_np[pick].astype(np.int64)
        y = torch.from_numpy(y_pick_np)   # ORIGINAL labels — no remap (D-034)
        probes.append(Probe(x=x, y=y,
                            name=f"semantic_novel_class_i={i}",
                            family="semantic_novel_class", strength=1.0))
    return probes


def _build_noise(tx, ty, n: int, rng: np.random.Generator, subset: int
                  ) -> List[Probe]:
    stds = np.linspace(0.02, 0.5, n).tolist()
    probes = []
    for i, s in enumerate(stds):
        idx = rng.choice(tx.shape[0], size=min(subset, tx.shape[0]), replace=False)
        x_sub = tx[idx]
        g = torch.Generator().manual_seed(int(rng.integers(1, 10**9)))
        noise = torch.empty(x_sub.shape).normal_(std=float(s), generator=g)
        x = x_sub + noise
        y = ty[idx].long()
        probes.append(Probe(x=x, y=y,
                            name=f"noise_std={s:.2f}",
                            family="noise", strength=float(s)))
    return probes


def _build_input_scale(tx, ty, n: int, rng: np.random.Generator, subset: int
                        ) -> List[Probe]:
    scales = np.linspace(0.5, 1.5, n).tolist()
    probes = []
    for i, s in enumerate(scales):
        idx = rng.choice(tx.shape[0], size=min(subset, tx.shape[0]), replace=False)
        x = tx[idx] * float(s)
        y = ty[idx].long()
        probes.append(Probe(x=x, y=y,
                            name=f"input_scale_s={s:.2f}",
                            family="input_scale", strength=float(s)))
    return probes


def _build_label_shift(tx, ty, n_classes: int, n: int,
                        rng: np.random.Generator, subset: int) -> List[Probe]:
    """target = (correct + k) mod n_classes for a spread of k values."""
    # Choose k values that spread across [1, n_classes-1]
    if n <= n_classes - 1:
        ks = np.linspace(1, n_classes - 1, n).round().astype(int).tolist()
    else:
        ks = list(range(1, n_classes))
    ks = list(dict.fromkeys(ks))    # dedupe
    probes = []
    for k in ks[:n]:
        idx = rng.choice(tx.shape[0], size=min(subset, tx.shape[0]), replace=False)
        x = tx[idx]
        y_orig = ty[idx].numpy() if isinstance(ty, torch.Tensor) else ty[idx]
        y_shift = ((y_orig.astype(np.int64) + int(k)) % n_classes).astype(np.int64)
        y = torch.from_numpy(y_shift)
        probes.append(Probe(x=x, y=y,
                            name=f"label_shift_k={k}",
                            family="label_shift", strength=float(k)))
    return probes


def _build_anti_class(tx, ty, n: int, model, device,
                       rng: np.random.Generator, subset: int) -> List[Probe]:
    """target = argmin over classes ≠ correct of p(c|x, model). This label is
    what the network currently thinks is LEAST likely, so g_probe wants to push
    predictions toward the *wrong* class → strongly anti-aligned with g_train.
    """
    from .models import set_flat_params, flat_params  # noqa
    m = deepcopy(model).to(device)
    m.eval()
    probes = []
    for i in range(n):
        idx = rng.choice(tx.shape[0], size=min(subset, tx.shape[0]), replace=False)
        x = tx[idx]
        with torch.no_grad():
            logits = m(x.to(device)).cpu()
        # Force minimum class ≠ correct: mask out the true label, pick argmin
        probs = torch.softmax(logits, dim=-1)
        # Set correct-class prob to +inf so argmin ignores it
        n_classes = probs.shape[1]
        y_true = ty[idx].long()
        probs[torch.arange(len(y_true)), y_true] = float("inf")
        y_anti = probs.argmin(dim=-1).long()
        probes.append(Probe(x=x, y=y_anti,
                            name=f"anti_class_i={i}",
                            family="anti_class", strength=float(i)))
    return probes


# ---------------- Public API ----------------

def build_canonical_probes(
    tx: torch.Tensor,
    ty: torch.Tensor,
    families_and_counts: List[Tuple[str, int]],
    seed: int,
    subset_size: int = 3000,
    n_classes: int = 10,
    held_out_classes: Optional[Tuple[int, ...]] = None,
    model_for_anticlass: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
) -> List[Probe]:
    """Build the full canonical probe list per D-034.

    `families_and_counts` is a list like [("structural_permute", 8),
    ("semantic_novel_class", 4), ...]. Values not listed are skipped.
    """
    rng = np.random.default_rng(seed)
    out: List[Probe] = []
    for family, count in families_and_counts:
        if count <= 0: continue
        if family == "structural_permute":
            out.extend(_build_structural_permute(tx, ty, count, rng, subset_size))
        elif family == "semantic_novel_class":
            assert held_out_classes is not None
            out.extend(_build_semantic_novel_class(tx, ty, count, rng,
                                                   subset_size, held_out_classes))
        elif family == "noise":
            out.extend(_build_noise(tx, ty, count, rng, subset_size))
        elif family == "input_scale":
            out.extend(_build_input_scale(tx, ty, count, rng, subset_size))
        elif family == "label_shift":
            out.extend(_build_label_shift(tx, ty, n_classes, count, rng, subset_size))
        elif family == "anti_class":
            assert model_for_anticlass is not None and device is not None
            out.extend(_build_anti_class(tx, ty, count, model_for_anticlass, device,
                                          rng, subset_size))
        else:
            raise ValueError(f"unknown probe family: {family}")
    return out


def probe_grad_align(model, probe: Probe, g_ref_train: torch.Tensor,
                     batch: int, device: torch.device,
                     is_classification: bool = True) -> Tuple[float, float]:
    """Return (align, ||g_probe||) at the pre-update checkpoint.

    align = cos(g_ref_train, g_ref_probe), where g_ref_probe is the mean gradient
    of the probe loss over one batch. Returns 0-magnitude gracefully.
    """
    m = deepcopy(model).to(device)
    m.zero_grad(set_to_none=True)
    # One representative batch
    n = probe.x.shape[0]
    take = min(batch, n)
    idx = torch.randperm(n)[:take]
    xb = probe.x[idx].to(device); yb = probe.y[idx].to(device)
    loss = F.cross_entropy(m(xb), yb)
    loss.backward()
    g_probe = flat_grad(m).detach().cpu()
    n_train = g_ref_train.norm() + 1e-12
    n_probe = g_probe.norm() + 1e-12
    align = float((g_ref_train @ g_probe / (n_train * n_probe)).item())
    return align, float(g_probe.norm().item())


def normalized_ftd(auc_after: float, auc_ref: float, eps: float = 1e-6) -> float:
    """Pre-registered normalization: (AUC_after - AUC_ref) / AUC_ref.

    Positive → update hurt probe learning-cost; negative → helped.
    """
    return (auc_after - auc_ref) / max(auc_ref, eps)
