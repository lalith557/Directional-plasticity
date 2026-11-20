"""Candidate A: projection-based directional constraint at matched ‖Δθ‖.

Given a *harmful direction* û_bad (estimated from prior updates whose FTD was
worst), we project each incoming SGD update to remove its component along û_bad
and then exactly renormalize back to the original ‖Δθ‖.
"""
from __future__ import annotations
import numpy as np
import torch
from typing import List


def estimate_harmful_direction(
    deltas: List[torch.Tensor],
    ftds: List[float],
    top_k: int = 1,
) -> torch.Tensor:
    """Return unit vector along the top-k principal direction of the
    FTD-weighted delta set.

    We rank updates by FTD and take the *worst* n_harmful via PCA. Weighting
    by (ftd - median(ftd)) puts weight on damaging directions and negative
    weight on beneficial ones, sharpening the estimate.
    """
    n = len(deltas)
    assert n == len(ftds) and n > 1
    weights = np.array(ftds) - np.median(ftds)
    D = deltas[0].numel()
    # Stack as (n, D)
    A = torch.stack([d / (d.norm() + 1e-12) for d in deltas], dim=0)  # unit vectors
    # Weighted covariance in the D-dim space is too big; use the Gram trick:
    # If A' = W^{1/2} A, then top eigenvector of A'^T A' equals A'^T of top
    # eigenvector of A' A'^T (which is n×n and cheap).
    w = torch.tensor(weights, dtype=A.dtype).clamp(min=0)  # keep only damaging
    if w.sum() < 1e-8:
        w = torch.ones(n, dtype=A.dtype)
    Aw = A * w.sqrt().unsqueeze(1)  # (n, D)
    G = Aw @ Aw.T                    # (n, n)
    eigvals, eigvecs = torch.linalg.eigh(G)
    # Top eigenvector -> direction in D
    top = eigvecs[:, -top_k:]         # (n, k)
    dirs = Aw.T @ top                 # (D, k)
    # Normalize each column
    dirs = dirs / (dirs.norm(dim=0, keepdim=True) + 1e-12)
    # Return the leading direction as a flat vector for k=1
    return dirs[:, 0].contiguous()


def project_out(delta: torch.Tensor, u_bad: torch.Tensor,
                exact_renorm: bool = True) -> torch.Tensor:
    """Remove the component of `delta` along `u_bad` and (optionally) rescale
    to preserve ‖delta‖ exactly (Control A)."""
    u = u_bad / (u_bad.norm() + 1e-12)
    comp = (delta @ u) * u
    d_perp = delta - comp
    if exact_renorm:
        orig_norm = delta.norm()
        cur_norm = d_perp.norm() + 1e-12
        d_perp = d_perp * (orig_norm / cur_norm)
    return d_perp


def project_out_multi(delta: torch.Tensor, U_bad: torch.Tensor,
                      exact_renorm: bool = True) -> torch.Tensor:
    """Multi-direction projection: U_bad has shape (D, k)."""
    Q, _ = torch.linalg.qr(U_bad)  # orthonormalize
    comp = Q @ (Q.T @ delta)
    d_perp = delta - comp
    if exact_renorm:
        orig_norm = delta.norm()
        cur_norm = d_perp.norm() + 1e-12
        d_perp = d_perp * (orig_norm / cur_norm)
    return d_perp
