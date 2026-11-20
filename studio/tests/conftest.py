import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def runs_tmp(tmp_path, monkeypatch):
    """Redirect the Studio runs directory to a temporary folder (real runs are untouched)."""
    d = tmp_path / "runs"
    d.mkdir()
    monkeypatch.setenv("STUDIO_RUNS_DIR", str(d))
    return d


def tiny_config(**over):
    from studio.engine import config_loader as cl
    raw = {"dataset": {"name": "permuted_mnist", "n_tasks": 3, "train_subset": 1000},
           "optimization": {"steps_per_task": 15}, "evaluation": {"diag_probe_size": 500},
           "seed": 11}
    for k, v in over.items():
        raw[k] = v
    return cl.normalize(raw)
