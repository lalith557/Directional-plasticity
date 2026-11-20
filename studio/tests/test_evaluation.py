import math

import numpy as np
import pytest

from studio.engine import evaluation_service as ev
from studio.engine import legacy_artifacts as la


def _e20():
    return [r for r in la.index() if r["group"] == "e20"]


def test_method_level_correlation_reproduces_paper_r_and_withholds_ci():
    recs = _e20()
    c = ev.correlation([r["metrics"]["final_weight_norm"]["mean"] for r in recs],
                       [r["metrics"]["decay"]["mean"] for r in recs])
    assert c["n"] == 7 and round(c["pearson_r"], 2) == 0.97 and round(c["spearman_rho"], 2) == 0.86
    assert c["ci95"] is None and "n = 7" in c["ci_note"]
    assert "not 7 independent seed-level" in c["scope_note"]


def test_ci_reported_only_with_enough_points():
    rng = np.random.default_rng(0)
    x = rng.normal(size=12); y = x + rng.normal(scale=0.5, size=12)
    c = ev.correlation(x, y)
    assert c["ci95"] is not None and c["ci95"][0] < c["pearson_r"] < c["ci95"][1]


def test_non_finite_points_excluded_and_reported():
    c = ev.correlation([1, 2, 3, float("nan"), 5], [1, 2, 3, 4, None], labels=list("abcde"))
    assert c["n"] == 3 and c["excluded"] == ["d", "e"]


def test_cnn_regime_excludes_diverged_arm():
    recs = [r for r in la.index() if r["group"] == "e25_cnn"]
    c = ev.correlation([r["metrics"]["final_weight_norm"]["mean"] for r in recs],
                       [r["metrics"]["decay"]["mean"] for r in recs], labels=[r["arm_label"] for r in recs])
    assert c["n"] == 6 and c["excluded"] == ["Random Direction"] and round(c["pearson_r"], 2) == -0.82


def test_paired_differences_match_paper_and_sign_test():
    e20 = {r["arm_key"]: r for r in _e20()}
    a = {x["seed"]: x["mean_acc"] for x in e20["l2_init"]["per_seed"]}
    b = {x["seed"]: x["mean_acc"] for x in e20["sgd"]["per_seed"]}
    p = ev.paired(a, b, "mean_acc")
    assert p["ok"] and p["n"] == 5 and round(p["descriptive"]["mean_difference"], 3) == 0.074
    assert p["descriptive"]["a_better_on"] == 5
    assert p["tests"]["sign_test_p_two_sided"] == pytest.approx(0.0625)
    assert "no significance claim" in p["interpretation_note"]
    lower = ev.paired({1: 0.1, 2: 0.2}, {1: 0.3, 2: 0.4}, "decay", higher_is_better=False)
    assert lower["descriptive"]["a_better_on"] == 2


def test_aggregation_matches_numpy_population_std_and_curves():
    agg = ev.aggregate([1.0, 2.0, None, float("nan"), 4.0])
    assert agg["n"] == 3 and agg["std"] == pytest.approx(np.std([1, 2, 4]))
    cur = ev.aggregate_curves([[1, 2, 3], [3, 4], [5, None, 7]])
    assert cur["mean"] == [3.0, 3.0, 5.0] and cur["n"] == [3, 2, 2]


def test_legacy_aggregates_equal_per_seed_means():
    for r in _e20():
        vals = [row["mean_acc"] for row in r["per_seed"]]
        assert r["metrics"]["mean_acc"]["mean"] == pytest.approx(np.mean(vals), abs=1e-12)
        assert r["metrics"]["mean_acc"]["std"] == pytest.approx(np.std(vals), abs=1e-12)
