import math

import numpy as np
import pytest
import torch

from studio.engine import metrics_collector as mc
from studio.engine.training_service import mnist  # noqa: F401  (ensures src import path)
from src.plasticity_stress import (PlasticMLP, _all_flat, _backbone_params, diagnostics,
                                   dir_control_step, dir_inward_step)


def test_peak_to_tail_decay_matches_paper_definition():
    acc = [0.5, 0.9, 0.8, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3]
    peak, end, d = mc.peak_to_tail_decay(acc)
    assert peak == 0.9 and end == pytest.approx(np.mean(acc[-2:])) and d == pytest.approx(0.9 - 0.4)
    assert mc.peak_to_tail_decay([0.7])[2] == 0.0
    assert mc.internal_first_last_decay(acc) == pytest.approx(np.mean(acc[:4]) - np.mean(acc[-4:]))


def test_fixed_step_identity_and_corollaries():
    g = torch.Generator().manual_seed(0)
    theta = torch.randn(500, generator=g, dtype=torch.float64)
    d = torch.randn(500, generator=g, dtype=torch.float64); d /= d.norm()
    eta = 0.3
    lhs = (theta + eta * d).norm() ** 2
    rhs = theta.norm() ** 2 + 2 * eta * (theta @ d) + eta ** 2
    assert float(lhs) == pytest.approx(float(rhs), rel=1e-12)                      # Prop. 1
    u = theta / theta.norm()
    d_orth = d - (d @ u) * u; d_orth /= d_orth.norm()
    assert float((theta + eta * d_orth).norm() ** 2) == pytest.approx(float(theta.norm() ** 2 + eta ** 2), rel=1e-12)
    # Cor. 2: the threshold <θ,d> = −η/2 leaves the norm unchanged
    tangent = d_orth
    a = -eta / (2 * theta.norm())          # inward radial component
    d_thr = a * u + math.sqrt(1 - float(a) ** 2) * tangent
    assert float((theta + eta * d_thr).norm()) == pytest.approx(float(theta.norm()), rel=1e-12)


def test_research_rules_keep_matched_step_and_geometry():
    g = torch.Generator().manual_seed(1)
    theta = torch.randn(300, generator=g); step = 0.05 * torch.randn(300, generator=g)
    orth = dir_control_step(step, theta, 1.0)
    assert float(orth.norm()) == pytest.approx(float(step.norm()), rel=1e-6)
    assert abs(float(orth @ theta)) / (float(orth.norm()) * float(theta.norm())) < 1e-6
    inw = dir_inward_step(step, theta, 0.6)
    assert float(inw.norm()) == pytest.approx(float(step.norm()), rel=1e-6)
    assert float(inw @ theta) < float(step @ theta)                  # pushed inward


def test_task_record_radial_tangential_decomposition():
    ev = {"task": 0, "acc": 0.5, "weight_l2": 3.0, "dormant": 0.1, "eff_rank": 2.0,
          "applied_norms": [2.0, 2.0, 1.0], "ref_norms": [2.0, 2.0, 1.0],
          "grad_aligns": [1.0, 0.5, 1.0], "radial_aligns": [0.0, -1.0, 0.6]}
    r = mc.task_record(ev)
    assert r["radial_component_mean"] == pytest.approx((0 - 2.0 + 0.6) / 3)
    assert r["tangential_component_mean"] == pytest.approx((2.0 + 0.0 + 0.8) / 3)
    assert r["radial_energy_frac"] == pytest.approx((0 + 1 + 0.36) / 3)
    assert r["matched_dev_max"] == pytest.approx(0.0, abs=1e-9)
    ev["applied_norms"] = [2.0, 2.2, 1.0]
    assert mc.task_record(ev)["matched_dev_max"] == pytest.approx(0.1, rel=1e-6)


def test_missing_values_are_none_not_zero():
    ev = {"task": 0, "acc": float("nan"), "weight_l2": float("inf"), "dormant": 0.0, "eff_rank": float("nan"),
          "applied_norms": [], "ref_norms": [], "grad_aligns": [], "radial_aligns": []}
    r = mc.task_record(ev)
    assert r["acc"] is None and r["weight_norm"] is None and r["eff_rank"] is None
    assert r["dormant"] == 0.0 and r["step_norm_mean"] is None


def test_diagnostics_norm_dormancy_and_rank_definitions():
    torch.manual_seed(0)
    m = PlasticMLP(784, 16, 2, 10)
    x = torch.rand(200, 784)
    d = diagnostics(m, x, torch.device("cpu"))
    manual = torch.cat([p.detach().reshape(-1) for p in _backbone_params(m)]).norm()
    assert d["weight_l2"] == pytest.approx(float(manual), rel=1e-6)                 # backbone only
    assert d["weight_l2"] < float(_all_flat(m).norm())                              # head excluded
    with torch.no_grad():                                                           # kill layer 2
        m.hidden_layers[1].weight.zero_(); m.hidden_layers[1].bias.fill_(-1.0)
    d2 = diagnostics(m, x, torch.device("cpu"))
    with torch.no_grad():                    # independent count of already-dead layer-1 units
        n1 = int((torch.relu(m.hidden_layers[0](x)).abs().mean(0) < 1e-3).sum())
    assert d2["dormant_frac"] == pytest.approx((n1 + 16) / 32)                      # pooled over layers
    assert d2["eff_rank"] == 0.0 or math.isnan(d2["eff_rank"]) or d2["eff_rank"] < 1e-6
