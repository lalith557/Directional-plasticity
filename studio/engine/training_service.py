"""Bridge to the research training code (reused unchanged).

Builds the existing data stream and calls ``run_arm`` / ``run_arm_cnn`` with the
parameters of a normalized Studio configuration. No training logic lives here.
"""
from __future__ import annotations

import functools
import sys

from . import config_loader as cl
from .paths import DATASET_DIR, ROOT

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import load_mnist                       # noqa: E402
from src.device import pick_device                        # noqa: E402
from src.plasticity_stress import (PermutedMNIST, RotatedMNIST,  # noqa: E402
                                   run_arm, run_arm_cnn)


@functools.lru_cache(maxsize=1)
def mnist():
    """MNIST arrays from the repository's IDX files (cached per process)."""
    return load_mnist(str(DATASET_DIR))


def resolve_device(pref: str):
    """Use the repository's device picker; 'cuda' falls back to CPU only if unavailable."""
    return pick_device(pref)


def build_stream(cfg: dict):
    args = cl.to_run_args(cfg)
    tx, ty, vx, vy = mnist()
    cls = PermutedMNIST if args["stream"] == "permuted_mnist" else RotatedMNIST
    return cls(tx, ty, vx, vy, **args["stream_kwargs"])


def run(cfg: dict, device, progress_cb=None) -> dict:
    """Execute one configuration exactly as the experiment scripts do."""
    args = cl.to_run_args(cfg)
    stream = build_stream(cfg)
    if args["runner"] == "run_arm":
        return run_arm(stream, device=device, progress_cb=progress_cb, **args["run_kwargs"])
    return run_arm_cnn(stream, args["method"], device=device, progress_cb=progress_cb,
                       **args["run_kwargs"])
