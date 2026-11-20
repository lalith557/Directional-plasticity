import json

import pytest

from studio.engine import config_loader as cl
from studio.engine import experiment_validator as val


def test_defaults_are_paper_values_not_signature_defaults():
    c = cl.default_config()
    assert c["architecture"] == {"name": "mlp", "hidden": 16, "depth": 2}
    assert c["optimization"] == {"lr": 0.2, "batch": 128, "steps_per_task": 60}
    assert c["dataset"]["n_tasks"] == 100 and c["dataset"]["train_subset"] == 6000
    assert cl.default_config(method="shrink_pert")["method"]["params"] == {"shrink": 0.7, "perturb": 0.01}
    assert cl.default_config(method="cbp")["method"]["params"] == {"cbp_reset_rate": 1e-2}
    assert cl.default_config(method="adaptive_norm_revive")["method"]["params"]["anr_target_norm"] == 3.0
    assert cl.default_config("rotated_mnist", "cnn")["optimization"]["lr"] == 0.1   # e25


def test_normalize_rejects_unsupported_parameters():
    with pytest.raises(cl.ConfigError):
        cl.normalize({"method": {"name": "sgd", "params": {"momentum": 0.9}}})
    with pytest.raises(cl.ConfigError):
        cl.normalize({"optimization": {"weight_decay": 1e-4}})
    with pytest.raises(cl.ConfigError):
        cl.normalize({"dataset": {"name": "cifar10"}})
    with pytest.raises(cl.ConfigError):
        cl.normalize({"optimizer": "adam"})


def test_hash_stable_seed_sensitive_metadata_insensitive():
    a = cl.normalize({"seed": 1})
    b = cl.normalize({"seed": 1, "metadata": {"name": "x"}, "device": "cpu"})
    c = cl.normalize({"seed": 2})
    assert cl.config_hash(a) == cl.config_hash(b)
    assert cl.config_hash(a) != cl.config_hash(c)
    assert cl.protocol_key(a) == cl.protocol_key(c)
    d = cl.normalize({"seed": 1, "method": {"name": "l2_init"}})
    assert cl.protocol_key(a) == cl.protocol_key(d)            # method excluded
    e = cl.normalize({"seed": 1, "optimization": {"lr": 0.1}})
    assert cl.protocol_key(a) != cl.protocol_key(e)


def test_json_yaml_roundtrip():
    c = cl.normalize({"method": {"name": "dir_inward", "params": {"dir_coeff": 0.4}}, "seed": 5})
    assert cl.parse_text(cl.to_json(c), "json") == c
    assert cl.parse_text(cl.to_yaml(c), "yaml") == c
    assert cl.config_hash(cl.parse_text(cl.to_yaml(c), "yaml")) == cl.config_hash(c)


def test_compatibility_rules_from_code():
    errs, _ = val.validate_config(cl.normalize({"architecture": {"name": "cnn"}}))
    assert any("Rotated MNIST" in e for e in errs)                       # CNN × permuted
    errs, _ = val.validate_config(cl.normalize({"dataset": {"name": "rotated_mnist"},
                                                "architecture": {"name": "cnn"},
                                                "method": {"name": "cbp"}}))
    assert any("MLP-only" in e for e in errs)                            # CBP not in run_arm_cnn
    errs, _ = val.validate_config(cl.normalize({"dataset": {"name": "rotated_mnist"},
                                                "architecture": {"name": "cnn"},
                                                "method": {"name": "dir_inward"}}))
    assert errs == []


def test_ranges_and_paper_deviation_warnings():
    errs, _ = val.validate_config(cl.normalize({"optimization": {"lr": 0}}))
    assert errs
    errs, _ = val.validate_config(cl.normalize({"method": {"name": "adaptive_norm_revive",
                                                           "params": {"anr_mode": "both"}}}))
    assert errs
    errs, warns = val.validate_config(cl.normalize({"optimization": {"lr": 0.1}}))
    assert not errs and any("paper" in w for w in warns)
    errs, warns = val.validate_config(cl.default_config())
    assert errs == [] and warns == []


def test_to_run_args_maps_onto_existing_signatures():
    import inspect
    from studio.engine.training_service import run_arm, run_arm_cnn
    for ds, ar, me in [("permuted_mnist", "mlp", "adaptive_norm_revive"), ("rotated_mnist", "cnn", "norm_project"),
                       ("permuted_mnist", "mlp", "cbp"), ("rotated_mnist", "cnn", "angle")]:
        args = cl.to_run_args(cl.default_config(ds, ar, me))
        fn = run_arm if args["runner"] == "run_arm" else run_arm_cnn
        params = set(inspect.signature(fn).parameters)
        assert set(args["run_kwargs"]) <= params, set(args["run_kwargs"]) - params
    assert json.loads(cl.canonical_json({"b": 1, "a": 2})) == {"a": 2, "b": 1}
