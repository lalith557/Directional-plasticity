"""Filesystem locations used by the Studio (all derived from the repo root)."""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = ROOT / "results" / "tables"          # legacy artifacts (read-only)
DATASET_DIR = ROOT / "Deep Learning Dataset1"     # MNIST IDX files
PAPER_TEX = ROOT / "paper" / "main(iconcise).tex"

# Studio runs live here unless overridden (tests use a temporary directory).
RUNS_DIR = Path(os.environ.get("STUDIO_RUNS_DIR", ROOT / "studio" / "experiments"))

RUN_ID_RE = re.compile(r"^run-\d{8}-\d{6}-[0-9a-f]{6}$")
LEGACY_ID_RE = re.compile(r"^legacy\.[A-Za-z0-9_.+=\-]+$")


def runs_dir() -> Path:
    """Resolve the runs directory at call time so tests can redirect it."""
    d = Path(os.environ.get("STUDIO_RUNS_DIR", RUNS_DIR))
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_dir(run_id: str) -> Path:
    """Path of a Studio run; refuses anything that is not a well-formed run id."""
    if not RUN_ID_RE.match(run_id):
        raise ValueError(f"invalid run id: {run_id!r}")
    p = (runs_dir() / run_id).resolve()
    if p.parent != runs_dir().resolve():
        raise ValueError("run path escapes the runs directory")
    return p
