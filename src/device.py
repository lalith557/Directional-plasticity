"""Device auto-detect + hardware-profile helpers.

Every experiment should call `pick_device()` once and pass the result through
all model/tensor creation. The returned profile string is written into every
result JSON so cross-runs can be filtered by hardware.
"""
from __future__ import annotations
import os
import torch


def pick_device(prefer: str = "auto") -> tuple[torch.device, str]:
    """Return (device, profile_string)."""
    if prefer in ("cpu",):
        return torch.device("cpu"), "cpu"
    if prefer in ("cuda", "gpu") and torch.cuda.is_available():
        return torch.device("cuda"), f"gpu:{torch.cuda.get_device_name(0)}"
    if torch.cuda.is_available():
        return torch.device("cuda"), f"gpu:{torch.cuda.get_device_name(0)}"
    return torch.device("cpu"), "cpu"


def gpu_free_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.mem_get_info(0)[0] / 1e9


def env_summary() -> dict:
    return {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": (torch.version.cuda if torch.cuda.is_available() else None),
        "device_name": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else "cpu"),
        "gpu_free_gb": gpu_free_gb(),
        "hardware_profile": ("gpu" if torch.cuda.is_available() else "cpu"),
    }
