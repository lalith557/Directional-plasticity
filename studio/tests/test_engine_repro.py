"""Engine tests on tiny real runs (MNIST, a few tasks). Uses CUDA when available."""
import json
import math

import pytest
import torch

from studio.engine import config_loader as cl
from studio.engine import experiment_registry as reg
from studio.engine import experiment_runner as runner
from studio.engine import training_service as ts
from studio.engine.metrics_collector import MetricsCollector
from studio.engine.paths import run_dir

from .conftest import tiny_config

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KEYS = ("task_test_acc", "diag_weight_l2", "diag_dormant", "diag_eff_rank", "matched_max_dev")


@pytest.mark.parametrize("method,params", [
    ("sgd", {}), ("cbp", {"cbp_reset_rate": 1e-2}), ("l2_init", {"l2_init_lambda": 1e-2}),
    ("shrink_pert", {}), ("dir_control", {}), ("dir_inward", {}), ("adaptive_norm_revive", {}),
])
def test_hook_does_not_change_mlp_results(method, params):
    cfg = tiny_config(method={"name": method, "params": params})
    a = ts.run(cfg, DEV, progress_cb=None)
    events = []
    b = ts.run(cfg, DEV, progress_cb=events.append)
    for k in KEYS:
        assert a[k] == b[k], k
    assert sum(e["type"] == "task" for e in events) == 3
    assert sum(e["type"] == "step" for e in events) == 3 * 15


def test_hook_does_not_change_cnn_training():
    cfg = cl.normalize({"dataset": {"name": "rotated_mnist", "n_tasks": 2, "train_subset": 800, "test_subset": 300},
                        "architecture": {"name": "cnn"}, "method": {"name": "dir_inward"},
                        "optimization": {"steps_per_task": 8}, "evaluation": {"diag_probe_size": 300},
                        "seed": 3})
    a = ts.run(cfg, DEV, progress_cb=None)
    b = ts.run(cfg, DEV, progress_cb=lambda e: None)
    for k in ("task_test_acc", "diag_weight_l2", "diag_dormant"):
        assert a[k] == b[k], k
    # documented: CNN eff-rank diagnostic jitters ~1e-5 on CUDA even without the hook
    for x, y in zip(a["diag_eff_rank"], b["diag_eff_rank"]):
        assert abs(x - y) < 1e-3


def test_runner_end_to_end_with_checkpoint(runs_tmp):
    cfg = tiny_config(checkpoint_policy="final")
    rid = reg.create_run(cfg, metadata={"name": "e2e", "hypothesis": "pipeline works"}, status="queued")
    assert runner.execute(rid) == "completed"
    meta = reg.read_meta(rid)
    assert meta["status"] == "completed" and meta["steps_done"] == 45 and meta["duration_s"] > 0
    res = json.loads((run_dir(rid) / "result.json").read_text(encoding="utf-8"))
    s = res["summary"]
    assert len(s["per_task"]["acc"]) == 3 and s["n_tasks_completed"] == 3
    assert 0 <= s["mean_acc"] <= 1 and s["decay"] >= 0
    assert s["cor1_relative_error"] is None or s["cor1_relative_error"] >= 0
    events = reg.read_events(rid)
    assert [e["task"] for e in events] == [0, 1, 2]
    assert res["checkpoint"]["sha256"] and (run_dir(rid) / "checkpoints" / "final.pt").exists()
    env = json.loads((run_dir(rid) / "environment.json").read_text(encoding="utf-8"))
    assert env["code_sha256"]["src/plasticity_stress.py"] and env["dataset_sha256"]["train_images"]["sha256"]
    assert runner.execute(rid) == "completed"            # refuses to re-execute a finished run


def test_seed_reproducibility_across_runs(runs_tmp):
    cfg = tiny_config(method={"name": "dir_control"})
    r1, r2 = reg.create_run(cfg, status="queued"), reg.create_run(cfg, status="queued")
    runner.execute(r1); runner.execute(r2)
    s1 = reg.read_json(r1, "result.json")["summary"]; s2 = reg.read_json(r2, "result.json")["summary"]
    for k in ("mean_acc", "decay", "final_weight_norm", "final_dormant", "final_eff_rank"):
        assert s1[k] == s2[k], k
    assert s1["per_task"] == s2["per_task"]
    assert reg.read_meta(r1)["config_hash"] == reg.read_meta(r2)["config_hash"]
    assert r1 != r2                                       # a reproduction never reuses an id


def test_cancellation_is_honoured(runs_tmp):
    rid = reg.create_run(tiny_config(dataset={"name": "permuted_mnist", "n_tasks": 50, "train_subset": 1000}),
                         status="queued")
    (run_dir(rid) / "CANCEL").write_text("now", encoding="utf-8")
    assert runner.execute(rid) == "cancelled"
    m = reg.read_meta(rid)
    assert m["status"] == "cancelled" and "cancelled by user" in m["error"]
    assert not (run_dir(rid) / "result.json").exists()


def test_invalid_device_request_fails_loudly(runs_tmp, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rid = reg.create_run(tiny_config(device="cuda"), status="queued")
    assert runner.execute(rid) == "failed"
    assert "not silently falling back" in reg.read_meta(rid)["error"]


def test_collector_running_decay_is_labelled_estimate(tmp_path):
    col = MetricsCollector(tmp_path, n_tasks=2, steps_per_task=1)
    base = {"type": "task", "weight_l2": 1.0, "dormant": 0.0, "eff_rank": 1.0, "applied_norms": [1.0],
            "ref_norms": [1.0], "grad_aligns": [1.0], "radial_aligns": [0.0], "model": None}
    col({**base, "task": 0, "acc": 0.9}); col({**base, "task": 1, "acc": 0.6})
    assert col.records[-1]["decay_running_estimate"] == pytest.approx(0.3)
    assert math.isclose(json.loads((tmp_path / "progress.json").read_text())["tasks_done"], 2)
