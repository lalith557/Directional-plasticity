import json
import os
import stat

import pytest

from studio.engine import experiment_registry as reg
from studio.engine import experiment_validator as val
from studio.engine import legacy_artifacts as la
from studio.engine import config_loader as cl
from studio.engine.paths import RUN_ID_RE, run_dir

from .conftest import tiny_config


def test_create_run_unique_ids_and_immutable_snapshot(runs_tmp):
    c = tiny_config()
    ids = {reg.create_run(c) for _ in range(5)}
    assert len(ids) == 5 and all(RUN_ID_RE.match(i) for i in ids)
    rid = next(iter(ids))
    p = run_dir(rid) / "config.json"
    assert not os.access(p, os.W_OK)                                   # read-only file
    meta = reg.read_meta(rid)
    assert meta["status"] == "proposed" and meta["config_hash"] == cl.config_hash(c)
    assert val.verify_snapshot(run_dir(rid))["ok"]


def test_tampered_snapshot_is_detected(runs_tmp):
    rid = reg.create_run(tiny_config())
    p = run_dir(rid) / "config.json"
    os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
    cfg = json.loads(p.read_text(encoding="utf-8"))
    cfg["optimization"]["lr"] = 0.5
    p.write_text(json.dumps(cfg), encoding="utf-8")
    chk = val.verify_snapshot(run_dir(rid))
    assert not chk["ok"] and "modified" in chk["reason"]


def test_run_ids_are_validated_against_traversal(runs_tmp):
    for bad in ("../etc", "run-1", "legacy.e20.sgd", "run-20260101-000000-zzzzzz"):
        with pytest.raises(ValueError):
            run_dir(bad)


def test_legacy_index_is_complete_and_honest():
    idx = la.index()
    ids = {r["id"] for r in idx}
    for must in ("legacy.e20.sgd", "legacy.e20.dir_inward_g0.6", "legacy.e22_lambda.lambda=0.01",
                 "legacy.e27.gamma=0.2", "legacy.e22_anr.anr_revive_only", "legacy.e25_cnn.random_dir",
                 "legacy.e24_rotmnist.sgd", "legacy.cor1.orthogonal", "legacy.e18.scale_mechanism"):
        assert must in ids, must
    rd = la.get("legacy.e25_cnn.random_dir")
    assert rd["status"] == "completed_diverged" and rd["metrics"]["final_weight_norm"]["mean"] is None
    assert la.get("legacy.e18.scale_mechanism")["status"] == "incomplete"
    assert la.get("legacy.e21.pilot")["status"] == "superseded"
    assert all(r["validation"]["status"] in ("artifact", "missing") for r in idx)


def test_per_seed_reference_requires_exact_configuration():
    c = cl.normalize({"method": {"name": "sgd"}, "seed": 20260825})
    rec, row = la.per_seed_reference(c)
    assert rec["id"] == "legacy.e20.sgd" and row["seed"] == 20260825
    assert la.per_seed_reference(cl.normalize({"seed": 1})) is None                 # seed not in artifact
    assert la.per_seed_reference(cl.normalize({"seed": 20260825, "optimization": {"lr": 0.1}})) is None


def test_research_category_inference():
    assert reg.research_category(cl.normalize({})) == "A"
    assert reg.research_category(cl.normalize({"method": {"name": "l2_init", "params": {"l2_init_lambda": 0.1}}})) == "B"
    assert reg.research_category(cl.normalize({"method": {"name": "dir_inward", "params": {"dir_coeff": 0.2}}})) == "C"
    assert reg.research_category(cl.normalize({"architecture": {"name": "mlp", "hidden": 64}})) == "D"
    assert reg.research_category(cl.normalize({"dataset": {"name": "rotated_mnist"}})) == "E"
    assert reg.research_category(cl.normalize({"method": {"name": "adaptive_norm_revive",
                                                          "params": {"anr_mode": "revive"}}})) == "F"


def test_registry_never_writes_legacy_tables(runs_tmp):
    from studio.engine.paths import TABLES_DIR
    before = {p.name: p.stat().st_mtime for p in TABLES_DIR.iterdir()}
    reg.create_run(tiny_config())
    reg.all_records(); reg.summary_counts()
    after = {p.name: p.stat().st_mtime for p in TABLES_DIR.iterdir()}
    assert before == after


def test_atomic_write_survives_concurrent_reader(tmp_path):
    """Windows: os.replace fails while another handle is open; writers must retry, readers too."""
    import threading
    from studio.engine import fsutil
    target = tmp_path / "progress.json"
    fsutil.atomic_write(target, json.dumps({"i": -1}))
    stop, errors, naive_collisions = threading.Event(), [], []

    def contender():                 # raw handles: creates the sharing conflict on Windows
        while not stop.is_set():
            try:
                with open(target, encoding="utf-8") as f:
                    f.read()
            except (PermissionError, FileNotFoundError) as e:
                naive_collisions.append(e)    # expected on Windows — the hazard being guarded

    def reader():                    # the production read path used by the API server
        while not stop.is_set():
            try:
                fsutil.read_json(target)
            except Exception as e:   # noqa: BLE001
                errors.append(e)

    ts = [threading.Thread(target=contender), threading.Thread(target=reader)]
    for t in ts:
        t.start()
    try:
        for i in range(300):
            assert fsutil.atomic_write(target, json.dumps({"i": i}))   # writer must never give up
    finally:
        stop.set()
        for t in ts:
            t.join()
    assert not errors
    assert fsutil.read_json(target) == {"i": 299}
