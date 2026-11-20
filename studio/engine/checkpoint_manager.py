"""Checkpoints.

The existing pipeline saves no checkpoints (audit §3.6), so the default policy is
``"none"``. ``"final"`` stores the final state_dict once training has finished; it
reads the model only (``state_dict`` copy to CPU) and therefore cannot alter the run.
"""
from __future__ import annotations

import json
from pathlib import Path

from .reproducibility import sha256_file

POLICIES = {"none": "no checkpoint (matches the existing pipeline)",
            "final": "final model weights after the last task"}


def save_final(model, run_path: Path, config_hash: str) -> dict:
    import torch
    ck_dir = Path(run_path) / "checkpoints"
    ck_dir.mkdir(exist_ok=True)
    path = ck_dir / "final.pt"
    state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "config_hash": config_hash,
                "model_class": type(model).__name__}, path)
    info = {"name": "final.pt", "path": str(path.relative_to(Path(run_path))),
            "bytes": path.stat().st_size, "sha256": sha256_file(path),
            "config_hash": config_hash, "model_class": type(model).__name__}
    (ck_dir / "final.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def list_checkpoints(run_path: Path) -> list[dict]:
    ck_dir = Path(run_path) / "checkpoints"
    out = []
    for meta in sorted(ck_dir.glob("*.json")) if ck_dir.exists() else []:
        try:
            out.append(json.loads(meta.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return out
