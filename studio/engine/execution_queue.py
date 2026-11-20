"""Execution queue with a single-GPU lock.

* At most ``max_concurrent`` (default 1) run executes at a time; the rest wait in FIFO.
* Each run executes in a subprocess (``experiment_runner``); stdout → ``log.txt``.
* Cancel: queued → removed; running → ``CANCEL`` file (graceful, checked every few
  steps), then terminate after a grace period.
* Recovery on start: queued runs are re-queued; runs that were initializing/running when
  the server stopped are marked failed ("interrupted") — they are never resumed.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections import deque

from . import experiment_registry as reg
from .paths import ROOT, run_dir, runs_dir

GRACE_S = 20.0


class ExecutionManager:
    def __init__(self, max_concurrent: int = 1, poll_s: float = 0.3):
        self.max_concurrent = max_concurrent
        self.poll_s = poll_s
        self.queue: deque[str] = deque()
        self.active: dict[str, subprocess.Popen] = {}
        self.cancel_deadline: dict[str, float] = {}
        self.lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ------------------------------------------------------------------
    def start(self) -> None:
        """(Re)start the worker. Safe to call after stop() — e.g. across app lifespans."""
        self.recover()
        if self._thread is None or not self._thread.is_alive():
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._loop, name="studio-queue", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None

    def recover(self) -> None:
        with self.lock:
            for rid in sorted(reg.list_run_ids()):
                m = reg.read_meta(rid)
                if m["status"] == "queued" and rid not in self.queue:
                    self.queue.append(rid)
                elif m["status"] in reg.ACTIVE and rid not in self.active:
                    reg.update_meta(rid, status="failed", finished_at=reg.now_iso(),
                                    error="interrupted: the Studio server stopped while this run was "
                                          "active. Runs are not resumed; reproduce it as a new run.")

    # -- operations -----------------------------------------------------------------
    def submit(self, run_ids: list[str]) -> None:
        with self.lock:
            for rid in run_ids:
                m = reg.read_meta(rid)
                if m["status"] not in ("proposed", "queued"):
                    raise ValueError(f"{rid} is {m['status']}; only proposed runs can be queued")
                reg.update_meta(rid, status="queued", queued_at=reg.now_iso())
                if rid not in self.queue:
                    self.queue.append(rid)

    def cancel(self, rid: str) -> str:
        with self.lock:
            m = reg.read_meta(rid)
            if m["status"] in reg.TERMINAL:
                return m["status"]
            if rid in self.queue or m["status"] in ("queued", "proposed"):
                if rid in self.queue:
                    self.queue.remove(rid)
                reg.update_meta(rid, status="cancelled", finished_at=reg.now_iso(),
                                error="cancelled before execution")
                return "cancelled"
            (run_dir(rid) / "CANCEL").write_text(reg.now_iso(), encoding="utf-8")
            self.cancel_deadline[rid] = time.time() + GRACE_S
            return "cancelling"

    def state(self) -> dict:
        with self.lock:
            return {"max_concurrent": self.max_concurrent, "active": list(self.active),
                    "queued": list(self.queue),
                    "cancelling": [r for r in self.cancel_deadline if r in self.active]}

    # -- worker -----------------------------------------------------------------------
    def _launch(self, rid: str) -> None:
        log = open(run_dir(rid) / "log.txt", "a", encoding="utf-8")
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "STUDIO_RUNS_DIR": str(runs_dir())}
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen([sys.executable, "-m", "studio.engine.experiment_runner", rid],
                                cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env, **kwargs)
        log.close()
        self.active[rid] = proc
        reg.update_meta(rid, pid=proc.pid)

    def _reap(self) -> None:
        for rid, proc in list(self.active.items()):
            code = proc.poll()
            if code is None:
                dl = self.cancel_deadline.get(rid)
                if dl and time.time() > dl:
                    proc.terminate()
                continue
            del self.active[rid]
            killed = self.cancel_deadline.pop(rid, None) is not None
            m = reg.read_meta(rid)
            if m["status"] not in reg.TERMINAL:      # process died without recording a result
                status = "cancelled" if killed else "failed"
                tail = "\n".join(reg.read_log(rid, 15))
                reg.update_meta(rid, status=status, finished_at=reg.now_iso(), exit_code=code,
                                error=("terminated after the cancellation grace period" if killed
                                       else f"runner exited with code {code} without a result.\n{tail}"))
            else:
                reg.update_meta(rid, exit_code=code)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                with self.lock:
                    self._reap()
                    while self.queue and len(self.active) < self.max_concurrent:
                        rid = self.queue.popleft()
                        if reg.read_meta(rid)["status"] == "queued":
                            self._launch(rid)
            except Exception as e:  # pragma: no cover — keep the worker alive, log the error
                print(f"[studio-queue] error: {e}", file=sys.stderr)
            time.sleep(self.poll_s)
