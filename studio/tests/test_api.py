import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(runs_tmp):
    from studio.api import server
    server.MANAGER.queue.clear(); server.MANAGER.active.clear()
    with TestClient(server.app) as c:
        yield c


SMALL = {"dataset": {"name": "permuted_mnist", "n_tasks": 3, "train_subset": 1000},
         "optimization": {"steps_per_task": 15}, "evaluation": {"diag_probe_size": 500}}


def _wait(client, rid, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = client.get(f"/experiments/{rid}/status").json()
        if st["status"] in ("completed", "failed", "cancelled"):
            return st
        time.sleep(0.5)
    raise AssertionError(f"{rid} did not finish: {st}")


def test_health_schema_presets(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["legacy_records"] > 80
    s = client.get("/schema").json()
    assert set(s["methods"]) >= {"sgd", "cbp", "l2_init", "shrink_pert", "dir_control", "dir_inward",
                                 "adaptive_norm_revive"}
    pr = client.get("/presets").json()
    assert pr[0]["id"] == "smoke" and any(p["id"] == "legacy.e20.sgd" for p in pr)


def test_list_filter_and_legacy_detail(client):
    a = client.get("/experiments", params={"category": "A"}).json()
    assert len(a) == 7 and all(r["source"] == "legacy_artifact" for r in a)
    q = client.get("/experiments", params={"q": "radial"}).json()
    assert q
    d = client.get("/experiments/legacy.e25_cnn.random_dir").json()
    assert d["metrics"]["final_weight_norm"]["mean"] is None               # NaN → null, not 0
    assert client.get("/experiments/legacy.nope").status_code == 404
    m = client.get("/experiments/legacy.e20.sgd/metrics").json()
    assert m["kind"] == "aggregate" and "not persisted" in m["note"]


def test_configure_validation(client):
    bad = client.post("/experiments/configure", json={"config": {"architecture": {"name": "cnn"}}}).json()
    assert not bad["valid"] and bad["errors"]
    r = client.post("/experiments/configure", json={"config": {"method": {"name": "sgd", "params": {"x": 1}}}})
    assert r.status_code == 400
    ok = client.post("/experiments/configure", json={"config": {}, "seeds": [20260824, 20260825]}).json()
    assert ok["valid"] and ok["n_seeds"] == 2 and len(set(ok["config_hashes"])) == 2
    assert ok["legacy_counterpart"] and ok["legacy_counterpart"][0]["record"] == "legacy.e20.sgd"
    assert "schema_version" in ok["yaml"]
    saved = client.post("/experiments/configure", json={"config": SMALL, "save": True,
                                                        "metadata": {"name": "proposal"}}).json()
    rid = saved["saved_ids"][0]
    assert client.get(f"/experiments/{rid}").json()["status"] == "proposed"


def test_long_runs_require_confirmation(client):
    # 200 tasks × 2000 steps (= 400k steps per run) is far above the automatic-launch limit
    # whether or not a timing measurement exists for this device.
    long_cfg = {"dataset": {"name": "permuted_mnist", "n_tasks": 200}, "optimization": {"steps_per_task": 2000}}
    r = client.post("/experiments/run", json={"config": long_cfg, "seeds": [1, 2], "metadata": {"name": "too long"}})
    assert r.status_code == 409 and r.json()["detail"]["requires_confirmation"]
    assert client.get("/experiments", params={"source": "studio"}).json() == []      # nothing was created


def test_run_monitor_results_and_cancel(client):
    r = client.post("/experiments/run", json={"config": SMALL, "seeds": [5, 6],
                                              "metadata": {"name": "api e2e"}}).json()
    a, b = r["queued"]
    assert r["group_id"] == a
    st = _wait(client, a)
    assert st["status"] == "completed", st
    ev = client.get(f"/experiments/{a}/metrics").json()
    assert [e["task"] for e in ev["events"]] == [0, 1, 2]
    assert "radial_component_mean" in ev["events"][0]
    assert client.get(f"/experiments/{a}/logs").json()["lines"]
    res = client.get(f"/experiments/{a}/results").json()
    assert res["summary"]["n_tasks_completed"] == 3
    y = client.get(f"/experiments/{a}/config", params={"format": "yaml"})
    assert y.status_code == 200 and "attachment" in y.headers["content-disposition"]
    _wait(client, b)
    g = client.get(f"/experiments/group:{a}").json()
    assert g["n_seeds"] == 2 and g["metrics"]["mean_acc"]["n"] == 2 and g["curves"]["acc"]["n"] == [2, 2, 2]
    # cancel a queued run that never starts
    q = client.post("/experiments/configure", json={"config": SMALL, "save": True}).json()["saved_ids"][0]
    assert client.post(f"/experiments/{q}/cancel").json()["result"] == "cancelled"


def test_websocket_streams_real_events(client):
    rid = client.post("/experiments/run", json={"config": SMALL, "metadata": {"name": "ws"}}).json()["queued"][0]
    tasks, end = [], None
    with client.websocket_connect(f"/ws/experiments/{rid}") as ws:
        for _ in range(400):
            msg = ws.receive_json()
            if msg["type"] == "events":
                tasks += [e["task"] for e in msg["events"]]
            if msg["type"] == "end":
                end = msg
                break
    assert end and end["status"] == "completed" and tasks == [0, 1, 2]


def test_compare_pairs_only_compatible_records(client):
    c = client.post("/experiments/compare", json={"ids": ["legacy.e20.l2_init", "legacy.e20.sgd",
                                                          "legacy.e25_cnn.sgd"]}).json()
    pairs = {(p["a"], p["b"]): p for p in c["pairs"]}
    ok = pairs[("legacy.e20.l2_init", "legacy.e20.sgd")]
    assert ok["valid"] and round(ok["results"]["mean_acc"]["descriptive"]["mean_difference"], 3) == 0.074
    bad = pairs[("legacy.e20.sgd", "legacy.e25_cnn.sgd")]
    assert not bad["valid"] and any("protocols differ" in x for x in bad["reasons"])
    md = client.post("/experiments/compare", json={"ids": ["legacy.e20.l2_init", "legacy.e20.sgd"],
                                                   "format": "markdown"})
    assert md.status_code == 200 and "# Method comparison report" in md.text


def test_correlation_regimes_paper_and_gpu(client):
    ids = [r["id"] for r in client.get("/experiments", params={"category": "A"}).json()]
    c = client.post("/analysis/correlation", json={"ids": ids}).json()
    assert c["n"] == 7 and round(c["pearson_r"], 2) == 0.97 and c["ci95"] is None
    reg = client.get("/analysis/regimes").json()
    assert [round(x["pearson_r"], 2) for x in reg] == [0.97, 0.45, -0.52, -0.94, -0.82]
    p = client.get("/paper-results").json()
    assert p["consistency"]["checks"] == p["consistency"]["artifact_consistent"] > 50
    gpu = client.get("/resources/gpu").json()
    assert "cuda_available" in gpu and "queue" in gpu


def test_bad_ids_rejected(client):
    assert client.post("/experiments/legacy.e20.sgd/cancel").status_code == 400
    assert client.get("/experiments/run-20990101-000000-abcdef").status_code == 404
    assert client.get("/experiments/run-x/logs").json()["lines"] == []
