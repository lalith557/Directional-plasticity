"""Experiment configuration: schema, defaults, normalization, hashing, I/O.

Every field here maps one-to-one onto an argument that already exists in
``src/plasticity_stress.py`` (``PermutedMNIST``, ``RotatedMNIST``, ``run_arm``,
``run_arm_cnn``). Nothing the implementation does not support is exposed.

Defaults are the *paper* values (e20 / e25 scripts), NOT the function-signature
defaults of ``run_arm`` (which differ: hidden=128, shrink=0.6, cbp_reset_rate=1e-4,
anr_target_norm=2.0). Every run passes its parameters explicitly.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import yaml

SCHEMA_VERSION = 1
PAPER_SEEDS = [20260824, 20260825, 20260826, 20260827, 20260828]

# --------------------------------------------------------------------------- #
# Schema                                                                       #
# --------------------------------------------------------------------------- #
P = dict  # readability alias for parameter specs

DATASETS: dict[str, dict] = {
    "permuted_mnist": {
        "label": "Permuted MNIST",
        "archs": ["mlp"],
        "test_size": 10000,
        "description": "Each task applies a fixed random pixel permutation (labels unchanged); "
                       "evaluated on the full 10 000-image test set under that permutation.",
        "params": {
            "n_tasks": P(type="int", default=100, min=1, max=200, label="Number of tasks"),
            "train_subset": P(type="int", default=6000, min=128, max=60000,
                              label="Training subset per task"),
        },
    },
    "rotated_mnist": {
        "label": "Rotated MNIST",
        "archs": ["mlp", "cnn"],
        "test_size": None,  # = test_subset
        "description": "Each task rotates all images by a fixed angle ~ U[0°, 180°); "
                       "evaluated on a fixed test subset under that rotation.",
        "params": {
            "n_tasks": P(type="int", default=100, min=1, max=200, label="Number of tasks"),
            "train_subset": P(type="int", default=6000, min=128, max=60000,
                              label="Training subset per task"),
            "test_subset": P(type="int", default=2000, min=100, max=10000,
                             label="Test subset"),
        },
    },
}

ARCHITECTURES: dict[str, dict] = {
    "mlp": {
        "label": "MLP 784–h–…–h–10 (ReLU, shared head)",
        "runner": "run_arm",
        "params": {
            "hidden": P(type="int", default=16, min=1, max=1024, label="Hidden width h"),
            "depth": P(type="int", default=2, min=1, max=4, label="Hidden layers"),
        },
    },
    "cnn": {
        "label": "Small CNN (conv16 → conv32 → FC → head)",
        "runner": "run_arm_cnn",
        "params": {
            "base": P(type="int", default=16, min=4, max=64, label="Base channels"),
            "fc": P(type="int", default=64, min=8, max=512, label="FC width"),
        },
    },
}

OPTIMIZATION: dict[str, dict] = {
    "lr": P(type="float", default=0.2, min=1e-5, max=5.0, label="Learning rate η₀"),
    "batch": P(type="int", default=128, min=1, max=6000, label="Batch size"),
    "steps_per_task": P(type="int", default=60, min=1, max=2000, label="SGD steps per task"),
}

EVALUATION: dict[str, dict] = {
    "diag_probe_size": P(type="int", default=1000, min=10, max=10000,
                         label="Diagnostic probe size"),
}

METHODS: dict[str, dict] = {
    "sgd": dict(label="SGD", family="control", matched_step="reference",
                archs=["mlp", "cnn"], group="primary", params={},
                equation="Δθ' = Δθ = −η₀∇L"),
    "cbp": dict(label="Continual Backprop", family="reset", matched_step="na",
                archs=["mlp"], group="primary",
                params={"cbp_reset_rate": P(type="float", default=1e-2, min=0.0, max=1.0,
                                            label="Reset probability ρ per step")},
                equation="per step, w.p. ρ re-init the lowest-utility hidden unit "
                         "(utility EMA 0.99, no maturity threshold)"),
    "l2_init": dict(label="L2-init", family="magnitude", matched_step="na",
                    archs=["mlp", "cnn"], group="primary",
                    params={"l2_init_lambda": P(type="float", default=1e-2, min=0.0, max=10.0,
                                                label="λ (norm-control strength)")},
                    equation="L + λ Σ‖p − p₀‖² on live parameters"),
    "shrink_pert": dict(label="Shrink & Perturb", family="magnitude", matched_step="na",
                        archs=["mlp", "cnn"], group="primary",
                        params={"shrink": P(type="float", default=0.7, min=1e-6, max=1.0,
                                            label="Shrink factor s"),
                                "perturb": P(type="float", default=0.01, min=0.0, max=1.0,
                                             label="Noise scale σ")},
                        equation="at each task boundary θ ← sθ + σε"),
    "dir_control": dict(label="Orthogonal Steering", family="directional",
                        matched_step="applicable", archs=["mlp", "cnn"], group="primary",
                        params={"dir_coeff": P(type="float", default=1.0, min=0.0, max=1.0,
                                               label="c (radial removal)")},
                        equation="d = Δθ − c(Δθ·θ̂)θ̂,  Δθ' = d·η/‖d‖"),
    "dir_inward": dict(label="Inward Steering", family="directional",
                       matched_step="applicable", archs=["mlp", "cnn"], group="primary",
                       params={"dir_coeff": P(type="float", default=0.6, min=0.0, max=1.0,
                                              label="γ (inward radial coefficient)")},
                       equation="d = Δθ − γηθ̂,  Δθ' = d·η/‖d‖"),
    "adaptive_norm_revive": dict(
        label="Adaptive-Norm-Revive", family="magnitude", matched_step="na",
        archs=["mlp"], group="primary",
        params={"anr_target_norm": P(type="float", default=3.0, min=1e-3, max=100.0,
                                     label="Per-unit target τ"),
                "anr_dormant_thresh": P(type="float", default=1e-3, min=0.0, max=1.0,
                                        label="Dormancy threshold"),
                "anr_mode": P(type="choice", default="full", choices=["full", "shrink", "revive"],
                              label="Components")},
        equation="at each boundary: cap per-unit incoming norm at τ; re-init dormant units, "
                 "zero their outgoing weights"),
    # Existing control arms (e23 / e25) -----------------------------------------
    "norm_project": dict(label="Norm Projection", family="magnitude", matched_step="na",
                         archs=["mlp", "cnn"], group="control",
                         params={"norm_project_target": P(type="float", default=6.0, min=1e-3,
                                                          max=1000.0, label="Ball radius")},
                         equation="after the SGD step, if ‖θ‖ > r: θ ← θ·r/‖θ‖"),
    "random_dir": dict(label="Random Direction", family="directional",
                       matched_step="applicable", archs=["mlp", "cnn"], group="control",
                       params={}, equation="Δθ' = r·η/‖r‖, r ~ N(0, I)"),
    "angle": dict(label="Angle-controlled", family="directional", matched_step="applicable",
                  archs=["mlp", "cnn"], group="control",
                  params={"cos_target": P(type="float", default=1.0, min=0.0, max=1.0,
                                          label="cos(Δθ', Δθ)")},
                  equation="Δθ' = (aĝ + √(1−a²)u)η, u ⊥ ĝ"),
}

DEVICE_CHOICES = ["auto", "cuda", "cpu"]
CHECKPOINT_POLICIES = ["none", "final"]
SCIENTIFIC_KEYS = ("schema_version", "dataset", "architecture", "method",
                   "optimization", "evaluation", "seed")


class ConfigError(ValueError):
    """Raised when a configuration cannot be normalized."""


# --------------------------------------------------------------------------- #
# Defaults / normalization                                                     #
# --------------------------------------------------------------------------- #
def _defaults(spec: dict[str, dict]) -> dict[str, Any]:
    return {k: v["default"] for k, v in spec.items()}


def default_config(dataset: str = "permuted_mnist", architecture: str = "mlp",
                   method: str = "sgd", seed: int = PAPER_SEEDS[0]) -> dict:
    """Paper primary configuration for the chosen dataset/architecture/method."""
    cfg = {
        "schema_version": SCHEMA_VERSION,
        "dataset": {"name": dataset, **_defaults(DATASETS[dataset]["params"])},
        "architecture": {"name": architecture, **_defaults(ARCHITECTURES[architecture]["params"])},
        "method": {"name": method, "params": _defaults(METHODS[method]["params"])},
        "optimization": _defaults(OPTIMIZATION),
        "evaluation": _defaults(EVALUATION),
        "seed": seed,
        "device": "auto",
        "checkpoint_policy": "none",
    }
    if architecture == "cnn":           # e25 used η₀ = 0.1 for the CNN
        cfg["optimization"]["lr"] = 0.1
    return cfg


def _coerce(name: str, spec: dict, value: Any) -> Any:
    t = spec["type"]
    try:
        if t == "int":
            if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
                raise ValueError
            return int(value)
        if t == "float":
            if isinstance(value, bool):
                raise ValueError
            return float(value)
        if t == "choice":
            return str(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{name}: expected {t}, got {value!r}") from None
    raise ConfigError(f"{name}: unknown type {t}")


def _merge_block(name: str, spec: dict[str, dict], raw: dict | None, extra_keys=()) -> dict:
    raw = dict(raw or {})
    out = {}
    unknown = set(raw) - set(spec) - set(extra_keys)
    if unknown:
        raise ConfigError(f"{name}: unsupported parameter(s) {sorted(unknown)} — "
                          "not implemented by the training code")
    for k, s in spec.items():
        out[k] = _coerce(f"{name}.{k}", s, raw[k]) if k in raw else s["default"]
    return out


def normalize(raw: dict) -> dict:
    """Fill defaults, coerce types, reject unsupported keys. Does not range-check
    (see experiment_validator.validate_config)."""
    if not isinstance(raw, dict):
        raise ConfigError("configuration must be an object")
    allowed_top = set(SCIENTIFIC_KEYS) | {"device", "checkpoint_policy", "metadata"}
    unknown = set(raw) - allowed_top
    if unknown:
        raise ConfigError(f"unsupported top-level key(s): {sorted(unknown)}")

    ds = (raw.get("dataset") or {}).get("name", "permuted_mnist")
    ar = (raw.get("architecture") or {}).get("name", "mlp")
    me = (raw.get("method") or {}).get("name", "sgd")
    if ds not in DATASETS:
        raise ConfigError(f"unknown dataset {ds!r}; supported: {sorted(DATASETS)}")
    if ar not in ARCHITECTURES:
        raise ConfigError(f"unknown architecture {ar!r}; supported: {sorted(ARCHITECTURES)}")
    if me not in METHODS:
        raise ConfigError(f"unknown method {me!r}; supported: {sorted(METHODS)}")

    base = default_config(ds, ar, me)
    opt_raw = raw.get("optimization")
    cfg = {
        "schema_version": SCHEMA_VERSION,
        "dataset": {"name": ds, **_merge_block("dataset", DATASETS[ds]["params"],
                                              raw.get("dataset"), ("name",))},
        "architecture": {"name": ar, **_merge_block("architecture", ARCHITECTURES[ar]["params"],
                                                    raw.get("architecture"), ("name",))},
        "method": {"name": me, "params": _merge_block(f"method.{me}", METHODS[me]["params"],
                                                      (raw.get("method") or {}).get("params"))},
        "optimization": ({**base["optimization"], **_merge_block("optimization", OPTIMIZATION, opt_raw)}
                         if opt_raw else base["optimization"]),
        "evaluation": _merge_block("evaluation", EVALUATION, raw.get("evaluation")),
        "seed": _coerce("seed", {"type": "int"}, raw.get("seed", PAPER_SEEDS[0])),
        "device": str(raw.get("device", "auto")),
        "checkpoint_policy": str(raw.get("checkpoint_policy", "none")),
    }
    if opt_raw and "lr" not in opt_raw:          # keep the architecture-specific default lr
        cfg["optimization"]["lr"] = base["optimization"]["lr"]
    if raw.get("metadata"):
        md = raw["metadata"]
        cfg["metadata"] = {k: str(md[k]) for k in ("name", "hypothesis", "notes") if md.get(k)}
    return cfg


# --------------------------------------------------------------------------- #
# Hashing                                                                      #
# --------------------------------------------------------------------------- #
def scientific_part(cfg: dict) -> dict:
    """The fields that determine the computation (execution/metadata excluded)."""
    return {k: copy.deepcopy(cfg[k]) for k in SCIENTIFIC_KEYS}


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(cfg: dict) -> str:
    """SHA-256 of the canonical scientific configuration."""
    return hashlib.sha256(canonical_json(scientific_part(cfg)).encode("utf-8")).hexdigest()


def protocol_key(cfg: dict) -> str:
    """Hash of the configuration *without* method and seed — two runs with the same
    protocol key differ only in update rule / seed, which is what pairing requires."""
    sci = scientific_part(cfg)
    sci.pop("method"); sci.pop("seed")
    return hashlib.sha256(canonical_json(sci).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Serialization                                                                #
# --------------------------------------------------------------------------- #
def to_json(cfg: dict) -> str:
    return json.dumps(cfg, indent=2, sort_keys=False, ensure_ascii=False)


def to_yaml(cfg: dict) -> str:
    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)


def parse_text(text: str, fmt: str = "json") -> dict:
    data = yaml.safe_load(text) if fmt in ("yaml", "yml") else json.loads(text)
    return normalize(data)


# --------------------------------------------------------------------------- #
# Mapping onto the research code                                               #
# --------------------------------------------------------------------------- #
def to_run_args(cfg: dict) -> dict:
    """Translate a normalized config into stream + runner keyword arguments."""
    ds, ar, me, opt = cfg["dataset"], cfg["architecture"], cfg["method"], cfg["optimization"]
    stream_kwargs = {"n_tasks": ds["n_tasks"], "seed": cfg["seed"],
                     "train_subset": ds["train_subset"]}
    if ds["name"] == "rotated_mnist":
        stream_kwargs["test_subset"] = ds["test_subset"]
    run_kwargs = {"lr": opt["lr"], "epochs": 1, "batch": opt["batch"], "seed": cfg["seed"],
                  "steps_per_task": opt["steps_per_task"],
                  "diag_probe_size": cfg["evaluation"]["diag_probe_size"],
                  **me["params"]}
    if ar["name"] == "mlp":
        run_kwargs.update(method=me["name"], hidden=ar["hidden"], depth=ar["depth"], verbose=False)
    else:
        run_kwargs.update(base=ar["base"], fc=ar["fc"])
    return {"runner": ARCHITECTURES[ar["name"]]["runner"], "stream": ds["name"],
            "stream_kwargs": stream_kwargs, "method": me["name"], "run_kwargs": run_kwargs}


def schema() -> dict:
    """Machine-readable schema for the configuration UI."""
    return {"schema_version": SCHEMA_VERSION, "datasets": DATASETS,
            "architectures": ARCHITECTURES, "optimization": OPTIMIZATION,
            "evaluation": EVALUATION, "methods": METHODS, "devices": DEVICE_CHOICES,
            "checkpoint_policies": CHECKPOINT_POLICIES, "paper_seeds": PAPER_SEEDS}


def describe(cfg: dict) -> str:
    """One-line human description, e.g. 'Inward Steering (γ=0.6) · MLP h=16 · Permuted MNIST'."""
    me = METHODS[cfg["method"]["name"]]
    pr = ", ".join(f"{k}={v}" for k, v in cfg["method"]["params"].items())
    ar = cfg["architecture"]
    arch = f"MLP h={ar['hidden']}" + (f" d={ar['depth']}" if ar["depth"] != 2 else "") \
        if ar["name"] == "mlp" else f"CNN {ar['base']}/{2 * ar['base']}/fc{ar['fc']}"
    return (f"{me['label']}{f' ({pr})' if pr else ''} · {arch} · "
            f"{DATASETS[cfg['dataset']['name']]['label']}")
