"""Reproduce the full Directional Plasticity project pipeline.

Runs in dependency order. Each stage is a Python subprocess so a failure in
one experiment doesn't wedge the interpreter for the next.

  python run_all.py                    # runs all stages
  python run_all.py --only e03 e04     # runs only listed experiments
  python run_all.py --skip e02         # runs everything except the listed
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time


ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


# (stage_id, description, script_path)
STAGES = [
    ("e01", "Scaffold smoke test",              "experiments/e01_scaffold_smoke.py"),
    ("e02", "Identifiability pre-check (kill #1 gate)",
                                                "experiments/e02_identifiability.py"),
    ("e03", "Candidate B synthetic (pre-registered)",
                                                "experiments/e03_candidate_b_synthetic.py"),
    ("e03b","Candidate B robustness re-run w/ convex_mix source",
                                                "experiments/e03b_candidate_b_convexmix.py"),
    ("e03c","Candidate B in harder plasticity regime",
                                                "experiments/e03c_candidate_b_hard_regime.py"),
    ("e04", "Split-MNIST replication (kill #5 gate)",
                                                "experiments/e04_split_mnist_replication.py"),
    ("e05", "Candidate A projection method",    "experiments/e05_candidate_a_method.py"),
    ("e06", "Novelty extensions (multi-descriptor, stability, empirical u_bad)",
                                                "experiments/e06_novelty_extensions.py"),
    # ----- v1.1 (post-D-018) additions -----
    ("e04b","MNIST bridged (VIF fix; Phase 2)", "experiments/e04b_split_mnist_bridged.py"),
    ("e08", "Past-gradient projection method (Phase 1)",
                                                "experiments/e08_past_gradient_method.py"),
    ("e10", "Probe-family robustness sweep (Phase 3)",
                                                "experiments/e10_probe_robustness.py"),
    ("e11", "Fisher-eigenvector axis stability (Phase 5)",
                                                "experiments/e11_hessian_descriptor.py"),
    ("e09_cifar10",  "Split-CIFAR-10 replication (Phase 4)",
                                                "experiments/e09_split_cifar_replication.py --dataset cifar10"),
    ("e09_cifar100", "Split-CIFAR-100 replication (Phase 4)",
                                                "experiments/e09_split_cifar_replication.py --dataset cifar100"),
    # ----- v1.2 (post-D-026) additions -----
    ("e09_cifar10_novel",  "Split-CIFAR-10 novel-class probe (Phase A2)",
                                                "experiments/e09_split_cifar_replication.py --dataset cifar10 --probe novel_class --device cuda"),
    ("e12", "Probe-geometry predictive theory (Phase A1)",
                                                "experiments/e12_probe_geometry.py"),
    ("e13", "Forgetting-plasticity Pareto controller (Phase B1+B2)",
                                                "experiments/e13_tradeoff_controller.py"),
    ("e14", "Better curvature axes (Phase C1)",
                                                "experiments/e14_curvature_axes.py"),
    ("e15", "FTD-scale-invariance investigation (Phase D1)",
                                                "experiments/e15_scale_invariance.py"),
    # ----- v1.3 (post-D-033) additions -----
    ("e16", "Predictive theory v1.3 (canonical probes + normalized FTD)",
                                                "experiments/e16_predictive_theory.py"),
    ("e17", "Adaptive controller vs SGD/EWC/replay/CBP (ORIGINAL MOTIVE)",
                                                "experiments/e17_adaptive_controller.py"),
    ("e18", "Scale/mechanism check (larger CNN)",
                                                "experiments/e18_scale_mechanism.py"),
    ("e07_final","Cross-experiment forest plot (regenerated)",
                                                "experiments/e07_cross_experiment_forest.py"),
    # ----- v1.4 (post-D-043) — task-incremental regime fix + tradeoff controller -----
    ("e19_mnist",  "Tradeoff controller sweep on Split-MNIST TIL (Pivots 1+2)",
                                                "experiments/e19_tradeoff_controller.py --datasets split_mnist"),
    ("e19_cifar",  "Tradeoff controller sweep on Split-CIFAR-10 TIL (Pivots 1+2)",
                                                "experiments/e19_tradeoff_controller.py --datasets split_cifar10 --device cuda --force_controller"),
    # ----- v1.5 (post-D-044) — plasticity-stress regime: mechanism + impossibility + method -----
    ("e20",  "Plasticity-stress: mechanism + impossibility + method attempt (5 seeds)",
                                                "experiments/e20_plasticity_stress.py"),
    # ----- paper experiments (e21-e27) -----
    ("e21",  "Norm-strength (lambda) pilot, 3 seeds (superseded by e22)",
                                                "experiments/e21_norm_strength_sweep.py"),
    ("e22",  "Width 64, lambda sweep and ANR ablation (5 seeds)",
                                                "experiments/e22_reviewer_experiments.py"),
    ("e23",  "Update-geometry controls (random direction, angle sweep)",
                                                "experiments/e23_update_geometry.py"),
    ("e24",  "Rotated MNIST and width-128 generalization (5 seeds)",
                                                "experiments/e24_generalization.py"),
    ("e25",  "Small CNN on rotated MNIST (3 seeds)",
                                                "experiments/e25_cnn.py"),
    ("e26",  "Layer-wise norms and alternative metrics (3 seeds)",
                                                "experiments/e26_diagnostics.py"),
    ("e27",  "Inward radial-coefficient sweep (5 seeds)",
                                                "experiments/e27_radial_sweep.py"),
    ("web_data", "Rebuild the Plasticity Lab dataset from results/tables",
                                                "web/scripts/build_data.py"),
]


def run_stage(stage_id: str, script_rel: str) -> int:
    print(f"\n{'=' * 72}\n[run_all] STAGE {stage_id}: {script_rel}\n{'=' * 72}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    # Support "script.py --flag value" entries by splitting on whitespace
    parts = script_rel.split()
    cmd = [PY, os.path.join(ROOT, parts[0])] + parts[1:]
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    dt = time.time() - t0
    print(f"[run_all] {stage_id} finished in {dt:.1f}s with exit={proc.returncode}")
    return proc.returncode


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", default=None,
                   help="Only run these stage IDs")
    p.add_argument("--skip", nargs="+", default=[],
                   help="Skip these stage IDs")
    args = p.parse_args()

    to_run = [s for s in STAGES
              if (args.only is None or s[0] in args.only)
              and s[0] not in args.skip]
    print(f"[run_all] running {len(to_run)} stages: {[s[0] for s in to_run]}")
    failures = []
    for stage_id, desc, script in to_run:
        rc = run_stage(stage_id, script)
        if rc != 0:
            failures.append((stage_id, rc))
            print(f"[run_all] WARN: {stage_id} exited with {rc}; continuing")
    print(f"\n[run_all] done. failures = {failures}")


if __name__ == "__main__":
    main()
