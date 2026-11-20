"""Small filesystem helpers that are safe when the API server and a runner process
touch the same files concurrently.

On Windows, ``os.replace`` fails with PermissionError while another process has the
destination open (Python opens files without FILE_SHARE_DELETE), and a reader can see
a file mid-write. Writers therefore retry briefly; readers retry on transient errors.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def atomic_write(path: Path, text: str, attempts: int = 100, wait_s: float = 0.01) -> bool:
    """Write ``text`` to ``path`` atomically; returns False only if every attempt failed."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    for _ in range(attempts):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:
            time.sleep(wait_s)
    try:
        tmp.unlink()
    except OSError:
        pass
    return False


def read_json(path: Path, attempts: int = 20, wait_s: float = 0.01):
    """Read JSON, retrying transient sharing / partial-write errors. Missing file → None."""
    path = Path(path)
    last = None
    for _ in range(attempts):
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (PermissionError, json.JSONDecodeError) as e:
            last = e
            time.sleep(wait_s)
    raise last  # type: ignore[misc]
