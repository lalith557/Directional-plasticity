"""Environment, code and dataset fingerprints recorded with every Studio run."""
from __future__ import annotations

import functools
import hashlib
import platform
import sys
from pathlib import Path

from .paths import DATASET_DIR, ROOT

# Files whose content determines a run's computation.
CODE_FILES = [
    "src/plasticity_stress.py",
    "src/datasets.py",
    "src/device.py",
    "studio/engine/config_loader.py",
    "studio/engine/training_service.py",
    "studio/engine/metrics_collector.py",
]
# Same candidate names, in the same order, as src/datasets.load_mnist.
DATASET_FILES = {
    "train_images": ["train-images-idx3-ubyte", "train-images.idx3-ubyte"],
    "train_labels": ["train-labels-idx1-ubyte", "train-labels.idx1-ubyte"],
    "test_images": ["t10k-images-idx3-ubyte", "t10k-images.idx3-ubyte"],
    "test_labels": ["t10k-labels-idx1-ubyte", "t10k-labels.idx1-ubyte"],
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def code_fingerprint() -> dict:
    out = {}
    for rel in CODE_FILES:
        p = ROOT / rel
        out[rel] = sha256_file(p) if p.exists() else None
    return out


@functools.lru_cache(maxsize=1)
def dataset_fingerprint() -> dict:
    """SHA-256 of exactly the files load_mnist reads (resolved with its own _find)."""
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.datasets import _find
    out = {}
    for key, options in DATASET_FILES.items():
        try:
            p = Path(_find(str(DATASET_DIR), options))
            out[key] = {"file": p.name, "sha256": sha256_file(p)}
        except FileNotFoundError:
            out[key] = None
    return out


def environment() -> dict:
    """Software/hardware environment (no personal data)."""
    import numpy as np
    import torch
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return env


def snapshot() -> dict:
    return {"environment": environment(), "code_sha256": code_fingerprint(),
            "dataset_sha256": dataset_fingerprint()}


NOTES = [
    "MLP runs on CUDA are bit-reproducible from the seed (verified by test_engine_repro).",
    "CNN training trajectories (weights, accuracy, dormancy) are bit-reproducible; the CNN "
    "effective-rank diagnostic jitters by ~1e-5 between runs on CUDA because the cuDNN forward "
    "pass on the probe batch is not bitwise deterministic. It does not affect training.",
    "Resume from checkpoint is not supported: run_arm keeps CBP utilities, the global torch RNG "
    "state and per-step lists internally, so a resumed run would not equal an uninterrupted one.",
]
