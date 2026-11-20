"""Frozen pre-registration for the Directional Plasticity project.

Every threshold, seed, and kill condition below is committed BEFORE any outcome is
observed. Do not edit the numeric constants once experiments have run against them;
if a value must change, append a note to decision.md and add a new SUPERSEDED_* block.

All experiment scripts import these constants; they never redefine them.
"""

from __future__ import annotations
from dataclasses import dataclass, field


# =============================================================================
# 1. Seeds and hardware profile
# =============================================================================
PRIMARY_SEED: int = 20260824
BOOTSTRAP_SEEDS = (11, 23, 37, 47, 59)     # multi-seed novelty extension
HARDWARE_PROFILE: str = "cpu"              # {"cpu", "gpu"}; annotates every result


# =============================================================================
# 2. Synthetic teacher-student architecture
# =============================================================================
@dataclass(frozen=True)
class TeacherStudentConfig:
    input_dim: int = 20
    output_dim: int = 10
    hidden_width: int = 64
    hidden_depth: int = 2      # number of hidden layers
    activation: str = "relu"
    # Task family: teacher-weight rotation between tasks
    task_rotation_rad: float = 0.10
    n_tasks: int = 5           # continual sequence length
    train_samples_per_task: int = 4000
    eval_samples_per_task: int = 1000
    input_std: float = 1.0     # Gaussian input distribution


TEACHER_STUDENT = TeacherStudentConfig()


# =============================================================================
# 3. FTD probe protocol
# =============================================================================
@dataclass(frozen=True)
class ProbeConfig:
    n_probes: int = 3                     # number of held-out probe tasks
    probe_seed_offset: int = 10_000       # separates probe tasks from training tasks
    probe_steps: int = 200                # SGD steps per probe (short budget)
    probe_batch: int = 64
    probe_lr: float = 1e-3
    # FTD scalar: area-under-loss curve over probe_steps
    # (higher AUC == slower learning == more damage)
    # Reported as (AUC_updated - AUC_reference) so positive = damage.


PROBE = ProbeConfig()


# =============================================================================
# 4. Update generator
# =============================================================================
@dataclass(frozen=True)
class UpdateGeneratorConfig:
    # Three sources, each contributing equal counts per magnitude bucket
    sources: tuple = ("sgd_seed", "orth_perturb", "random_sphere")
    per_source_per_norm: int = 25         # 25 * 3 sources * 4 norms = 300 samples
    target_norms: tuple = (0.05, 0.10, 0.20, 0.40)
    sgd_inner_steps: int = 50             # steps to accumulate Δθ before renormalize
    sgd_inner_lr: float = 5e-3
    sgd_inner_batch: int = 128
    orth_noise_std: float = 1.0           # scale of perturbation before orthogonalize
    # After generation each Δθ is exactly renormalized to a target norm (L2).


UPDATE_GEN = UpdateGeneratorConfig()


# =============================================================================
# 5. Statistical thresholds and gates
# =============================================================================
@dataclass(frozen=True)
class StatsConfig:
    # Regression: FTD = b0 + b1*|dθ| + b2*CTB + b3*PTD + b4*D_dir
    covariates: tuple = ("mag", "ctb", "ptd", "d_dir")
    vif_threshold: float = 5.0            # kill #1 if any covariate VIF > this
    ptd_calipers: tuple = (0.01, 0.02, 0.05)   # Control C sweep
    ptd_primary_caliper: float = 0.02
    ols_cov_type: str = "HC3"             # robust standard errors
    bootstrap_iterations: int = 1000
    bootstrap_alpha: float = 0.05         # 95% CI
    beta4_significance_alpha: float = 0.05


STATS = StatsConfig()


# =============================================================================
# 6. Pre-committed kill conditions (§13 of the spec)
# =============================================================================
@dataclass(frozen=True)
class KillConditions:
    # #1 Identifiability failure: any covariate's VIF > threshold after decorrelation
    max_vif: float = STATS.vif_threshold
    # #2 Null residual: |β4| CI overlaps 0 across all robustness sweeps
    residual_significance_alpha: float = STATS.beta4_significance_alpha
    # #3 Control C failure: variance of PTD *within* a caliper > this fraction of
    #    across-group variance means the matching didn't actually match.
    within_caliper_ratio_max: float = 0.25
    # #4 Non-robustness: fraction of subsamples in which β4 sign flips
    max_sign_flip_fraction: float = 0.10
    # #5 Synthetic-only artifact: β4 significant on synthetic but CI overlaps 0
    #    on Split-MNIST
    require_mnist_replication: bool = True
    # #6 Trivial-magnitude Candidate A: method must beat baseline at MATCHED |dθ|
    matched_magnitude_tolerance: float = 0.02  # relative tolerance on ‖Δθ‖


KILL = KillConditions()


# =============================================================================
# 7. Split-MNIST configuration
# =============================================================================
@dataclass(frozen=True)
class SplitMNISTConfig:
    # Standard 5-task split
    task_class_pairs: tuple = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
    mnist_dir: str = r"Deep Learning Dataset1"
    train_batch: int = 128
    train_lr: float = 1e-3
    train_epochs_per_task: int = 3
    student_hidden_width: int = 128
    student_hidden_depth: int = 2
    # For the update generator in the MNIST regime:
    updates_per_task: int = 60


SPLIT_MNIST = SplitMNISTConfig()


# =============================================================================
# 8. Candidate A method
# =============================================================================
@dataclass(frozen=True)
class CandidateAConfig:
    # Estimate harmful direction from B updates with worst FTD, take top eigenvector
    n_harmful_updates: int = 40
    harmful_top_k: int = 1                # number of harmful directions to project out
    # After projection, EXACTLY renormalize to original ‖Δθ‖ (Control A)
    exact_renorm: bool = True
    # Adaptive variant (novelty): re-estimate harmful direction each K updates
    adaptive_reestimate_every: int = 20


CAND_A = CandidateAConfig()


# =============================================================================
# 9. Success / failure summary criteria
# =============================================================================
@dataclass(frozen=True)
class SuccessCriteria:
    # Clean success
    beta4_ci_excludes_zero_synthetic: bool = True
    beta4_ci_excludes_zero_mnist: bool = True
    candidate_a_beats_baseline_at_matched_mag: bool = True
    # Partial success
    only_synthetic_ok: str = "PARTIAL"
    # Clean negative
    beta4_ci_includes_zero_all: str = "NEGATIVE_PUBLISHABLE"


SUCCESS = SuccessCriteria()


# =============================================================================
# 10. Convenience: version + date the pre-registration was frozen
# =============================================================================
PREREG_VERSION: str = "1.0.0"
PREREG_FROZEN_DATE: str = "2026-08-24"


# =============================================================================
# 11. POST-PREREG ADDENDA (v1.1 onward)
# The blocks below were added AFTER the initial pre-registration was frozen.
# They introduce new configurations for experiments not covered by the initial
# scope (Split-CIFAR, past-gradient method, probe robustness, Hessian axis).
# NONE of them edit v1.0 constants; each is tied to a decision.md entry that
# documents motivation. See D-018+ for details.
# =============================================================================

PREREG_ADDENDA_VERSION: str = "1.1.0"
PREREG_ADDENDA_FROZEN_DATE: str = "2026-08-24"  # frozen when GPU became available


@dataclass(frozen=True)
class SplitCIFARConfig:
    """Split-CIFAR-10 (5 x 2-class) and Split-CIFAR-100 (10 x 10-class) configs.

    Frozen alongside D-019 (Phase 4 pre-registration). Numeric values chosen to fit
    an RTX 4070 (8 GB) at fp32; CPU fallback runs at reduced size via `cpu_scale`.
    """
    # CIFAR-10 protocol
    cifar10_class_pairs: tuple = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
    cifar10_dir: str = "cifar-10"
    # CIFAR-100 protocol (10 sequential 10-class superclass groups)
    cifar100_task_size: int = 10       # 10 classes per task -> 10 tasks total
    cifar100_dir: str = "cifar100"
    # Student CNN
    cnn_base_channels: int = 32
    cnn_hidden_fc: int = 128
    cnn_dropout: float = 0.0           # keep deterministic for beta4 measurements
    # Training
    train_epochs_per_task: int = 3
    train_lr: float = 5e-3
    train_batch: int = 128
    # Update generator (matched-|dtheta|)
    updates_per_source_per_norm: int = 15
    target_norms: tuple = (0.05, 0.10, 0.20, 0.40)
    sgd_inner_steps: int = 40
    sgd_inner_lr: float = 5e-3
    sgd_inner_batch: int = 128
    # Probe protocol
    probe_steps: int = 200
    probe_batch: int = 128
    probe_lr: float = 5e-3
    probe_subset: int = 3000
    n_probes: int = 3
    # CPU fallback
    cpu_scale_updates: float = 0.25    # reduce per-source count to 25%
    cpu_scale_probe_steps: float = 0.5


SPLIT_CIFAR = SplitCIFARConfig()


@dataclass(frozen=True)
class ProbeRobustnessConfig:
    """Phase 3: probe-family robustness. Frozen alongside D-021.

    Two probe families crossed with two budgets.
    """
    families: tuple = ("permuted_mnist", "novel_class_pair")
    probe_budgets: tuple = (100, 200)      # short + long budget
    n_probes: int = 3


PROBE_ROBUSTNESS = ProbeRobustnessConfig()


@dataclass(frozen=True)
class PastGradientMethodConfig:
    """Phase 1: past-gradient projection method. Frozen alongside D-020."""
    project_top_k: int = 1                 # rank of the projection basis
    # EWC baseline (fair anti-forgetting comparison)
    ewc_lambda: float = 100.0
    ewc_fisher_batch: int = 200
    # Number of past-task-gradient samples averaged to form the past axis
    n_past_grad_samples: int = 3


PAST_GRAD_METHOD = PastGradientMethodConfig()


@dataclass(frozen=True)
class HessianDescriptorConfig:
    """Phase 5: Hessian/Fisher-eigenvector directional descriptor. Frozen alongside D-022."""
    # We use Fisher information as a Hessian surrogate (Gauss-Newton block).
    top_k: int = 3
    lanczos_iters: int = 20
    fisher_batch: int = 128
    n_fisher_batches: int = 4


# =============================================================================
# 12. POST-PREREG ADDENDA v1.2 (frozen 2026-08-24 by D-027..D-031)
# These blocks add NEW configurations for Phase A/B/C/D of the v1.2 round.
# NONE of them edit v1.0 or v1.1 constants; each is tied to a decision.md
# entry that documents motivation.
# =============================================================================

PREREG_V12_VERSION: str = "1.2.0"
PREREG_V12_FROZEN_DATE: str = "2026-08-24"


@dataclass(frozen=True)
class ProbeGeometryConfig:
    """Phase A1 (e12): probe-geometry predictive model. Frozen alongside D-027.

    Acceptance bar (D-027):
      - Sign of beta_d_dir per probe is predicted from probe_grad_alignment
        with accuracy >= 80% on held-out probes, AND
      - Pearson correlation between probe_grad_alignment and beta_d_dir >= 0.6
        with 95% CI excluding 0 across held-out probes.
    """
    # Probe families to construct (grouped): each element is (family_kind, n_probes)
    probe_families: tuple = (
        ("permuted",       6),   # 6 different pixel permutations of the same class set
        ("novel_class",    5),   # 5 different held-out class subsets
        ("noise_augment",  4),   # 4 different additive-noise scales on the training class set
        ("input_scale",    3),   # 3 different input-scaling probes on the training set
    )
    n_updates_per_probe: int = 60       # matched-|dtheta| updates for each probe measurement
    probe_train_steps: int = 200
    probe_batch: int = 128
    probe_lr: float = 5e-3
    holdout_fraction: float = 0.30      # 30% of probes held out for the predictive test
    sign_accuracy_threshold: float = 0.80
    correlation_threshold: float = 0.60
    n_seeds: int = 3


PROBE_GEOMETRY = ProbeGeometryConfig()


@dataclass(frozen=True)
class TradeoffControllerConfig:
    """Phase B1 (e13): forgetting/plasticity Pareto controller. Frozen alongside D-028.

    Acceptance bar (D-028):
      - Controller produces at least one operating point that strictly Pareto-dominates
        EWC on (BWT, probe FTD) at matched-‖Δθ‖ tolerance <= 2%.
      - Also strictly Pareto-dominates plain SGD in the same sense.
    """
    # Sweep the projection coefficient c ∈ [-1, 1] along û_past
    # (c = -1 pure avoid; c = 0 orthogonal projection; c = +1 pure toward)
    coefficient_sweep: tuple = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)
    n_seeds: int = 3
    # Fair baselines to include
    baselines: tuple = ("sgd", "ewc", "replay", "cbp")
    # EWC params (mirrors v1.1 D-020 config)
    ewc_lambda: float = 100.0
    ewc_fisher_batch: int = 200
    # Replay: keep a small memory of past-task samples and mix into each minibatch
    replay_memory_per_task: int = 200
    replay_mix_ratio: float = 0.5       # fraction of each minibatch drawn from memory
    # Continual Backprop: probability of re-initializing "dead" units per step
    cbp_reset_rate: float = 1e-4
    cbp_utility_ema: float = 0.99
    # Datasets
    datasets: tuple = ("split_mnist", "split_cifar10")


TRADEOFF = TradeoffControllerConfig()


@dataclass(frozen=True)
class CurvatureAxesConfig:
    """Phase C1 (e14): better curvature-based directional descriptors. Frozen alongside D-029.

    Acceptance bar (D-029): each axis is evaluated on:
      (a) mean pairwise |cos(u_i, u_j)| across 5 seeds >= 0.5, AND
      (b) beta_d_dir_axis significant (CI excludes 0) with sign consistent across all 5 seeds.
    If ANY axis passes BOTH, we say a stable directional geometric axis exists.
    """
    axes: tuple = ("hessian_joint_lanczos", "kfac_natural_gradient",
                   "jacobian_svd_top1", "update_covariance_svd_top1")
    n_seeds: int = 5
    lanczos_iters: int = 30
    lanczos_batch: int = 128
    lanczos_n_batches: int = 3
    kfac_damping: float = 1e-3
    jacobian_n_samples: int = 128
    update_covariance_n_updates: int = 40
    axis_stability_threshold: float = 0.5


CURVATURE_AXES = CurvatureAxesConfig()


@dataclass(frozen=True)
class ScaleInvarianceConfig:
    """Phase D1 (e15): FTD-scale-invariance investigation. Frozen alongside D-030.

    Acceptance bar (D-030): report BOTH raw and normalized β4 on MNIST/CIFAR-10/CIFAR-100.
    If β4/FTD_scale collapses to a common value, the invariance is a scale artifact,
    not a property of the effect - and we say so plainly in the paper's limitations.
    """
    normalizations: tuple = ("raw", "by_ref_ftd", "by_ftd_std", "by_max_source_ftd")
    probe_difficulty_multipliers: tuple = (0.5, 1.0, 2.0)  # probe budget scaling


SCALE_INV = ScaleInvarianceConfig()


@dataclass(frozen=True)
class NovelClassCIFARConfig:
    """Phase A2 (e09 --probe novel_class): novel-class probe on CIFAR-10.

    Warmup on 4 class pairs, hold out one pair as the probe.
    Frozen alongside D-031.
    """
    warmup_pairs: tuple = ((0, 1), (2, 3), (4, 5), (6, 7))
    held_out_pair: tuple = (8, 9)
    n_probes: int = 3
    probe_steps: int = 200
    probe_batch: int = 128
    probe_lr: float = 5e-3
    probe_subset: int = 3000


NOVEL_CLASS_CIFAR = NovelClassCIFARConfig()


# =============================================================================
# 13. POST-PREREG ADDENDA v1.3 (frozen 2026-08-25 by D-034..D-040)
# Final round targeting the ORIGINAL motive: a matched-magnitude directional
# CONTROLLER that reduces FTD (plasticity), or a clean proof it cannot exist.
# Every constant here was frozen BEFORE any v1.3 experiment ran.
# =============================================================================

PREREG_V13_VERSION: str = "1.3.0"
PREREG_V13_FROZEN_DATE: str = "2026-08-25"


@dataclass(frozen=True)
class CanonicalProbesConfig:
    """Phase 0.2 (src/probes.py): canonical probe taxonomy with continuous-alignment
    parameters. Designed to close W1 (probe inconsistency), W2 (two-cluster
    correlation), W3 (all-positive-beta artifact), W5 (compressed CIFAR alignment).

    Frozen alongside D-034. Every family here has a `strength` parameter that
    smoothly interpolates between low- and high-alignment probes so the resulting
    beta_d_dir vs align curve is continuous (not clustered).
    """
    # Four taxa; sizes chosen to give total >= 24 probes so n_holdout at 0.30
    # gives ~7-8 held-out probes (enough for an informative bootstrap CI).
    families: tuple = (
        # (family_kind, count, notes)
        ("structural_permute",  8),    # permute a fraction of pixels
        ("semantic_novel_class", 4),   # held-out classes, ORIGINAL labels (head-consistent)
        ("noise",                4),   # additive Gaussian at varying std
        ("input_scale",          4),   # multiplicative input scale
        # NEW families that produce genuinely negative-align probes:
        ("label_shift",          6),   # target = (correct + k) mod C for k=1..C-1
        ("anti_class",           3),   # target = argmin_c p(c | x) — actively opposes training
    )
    subset_size: int = 3000
    # Probe head/labels invariant: ALWAYS use original class labels; no binary
    # remapping. Head-consistent probes are the ONLY valid probes (see D-034).
    labels_are_original: bool = True


CANONICAL_PROBES = CanonicalProbesConfig()


@dataclass(frozen=True)
class NormalizedFTDConfig:
    """Phase 0.3 (D-035): principled FTD normalization to remove W10 scale artifact.

    normalized_FTD = (AUC_probe(θ_after) - AUC_probe(θ_ref)) / AUC_probe(θ_ref)

    Dimensionless fractional change. Positive = update HURT probe learning-cost;
    negative = update HELPED probe learning-cost. All v1.3 betas reported on
    this normalization; raw FTD still logged as a nuisance column.
    """
    normalization: str = "relative_to_ref"    # (AUC_after - AUC_ref) / AUC_ref
    fallback_eps: float = 1e-6                # avoid /0 for degenerate probes


NORMALIZED_FTD = NormalizedFTDConfig()


@dataclass(frozen=True)
class PredictiveTheoryV3Config:
    """Phase 1 (e16): power-adequate rerun of the predictive theory.

    Fixes: n_holdout large enough for informative bootstrap CI (supersedes D-032).
    Frozen alongside D-036.

    Acceptance:
      - Sign-of-beta predicted on >= 80% of held-out probes, AND
      - Slope-only bootstrap 95% CI (B=1000) excludes 0 on held-out probes,
        with holdout_fraction * n_probes >= 8. Bootstrap CI is authoritative;
        NO parametric fallback (D-032 explicitly superseded by D-037).
    """
    holdout_fraction: float = 0.30
    n_bootstrap: int = 1000
    sign_accuracy_threshold: float = 0.80
    slope_ci_alpha: float = 0.05
    updates_per_probe: int = 80         # 5 per source per norm x 16 = 80
    probe_train_steps: int = 200
    probe_batch: int = 128
    probe_lr: float = 5e-3
    # Datasets - CIFAR-10 only for canonical probes (CIFAR-100 optional in Phase 4)
    datasets: tuple = ("mnist", "cifar10")


PREDICTIVE_V3 = PredictiveTheoryV3Config()


@dataclass(frozen=True)
class AdaptiveControllerConfig:
    """Phase 2 (e17): state-dependent adaptive controller for plasticity (ORIGINAL MOTIVE).

    The controller estimates the LOCALLY-harmful direction at each step from the
    mean of K auxiliary probe gradients (sampled from a plausible-future-task
    distribution), projects the SGD update to remove its component along that
    direction, and exact-renorms to ||delta|| to satisfy kill #6 by construction.

    Frozen alongside D-038.

    Acceptance (STRICT — the ORIGINAL goal):
      - Adaptive controller reduces final probe FTD vs plain SGD at matched
        ||delta|| within tolerance 2%, reproduced over 3 seeds on >= 1 real dataset.
      - Is not Pareto-dominated by EWC/replay/CBP on (probe FTD, mean_final_acc).
    """
    # Auxiliary probe count sampled each estimation step
    n_aux_probes: int = 6
    # Estimation cadence: recompute u_hazard every N optimizer steps
    reestimate_every: int = 50
    # Projection strength alpha in [0, 1]; 0 = SGD, 1 = full projection
    projection_alpha_sweep: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
    n_seeds: int = 3
    # Multi-directional variant: project out top-k principal directions of past aux gradients
    top_k_project: int = 1
    # Fair baselines
    baselines: tuple = ("sgd", "ewc", "replay", "cbp")
    ewc_lambda: float = 100.0
    replay_memory_per_task: int = 200
    replay_mix_ratio: float = 0.5
    cbp_reset_rate: float = 1e-4


ADAPTIVE = AdaptiveControllerConfig()


@dataclass(frozen=True)
class NonDegenerateRegimeConfig:
    """Phase 2 pre-req (D-039): fix W7 — pre-registered hyperparameters that
    make baselines reach non-trivial accuracy so the Pareto comparison is real.

    On the previous CIFAR-10 run baselines collapsed to ~11% mean_final_acc
    with [0,0,0,0,0.56] per-task accs (only the last task was learned). The
    diagnosis: 3 epochs/task × lr=5e-3 on the SmallCNN was too little to reach
    a plateau; the shared 10-way head + all-tasks-see-all-classes was fine but
    without warm training even the last task was under-trained.
    """
    train_epochs_per_task: int = 8       # up from 3
    train_lr: float = 3e-3               # down from 5e-3 for stability
    train_batch: int = 128
    # Larger CNN base width for stability
    cnn_base_channels: int = 32
    cnn_hidden_fc: int = 128
    # Split-MNIST regime — MLP already OK but bump epochs
    mnist_train_epochs_per_task: int = 5    # up from 3
    mnist_train_lr: float = 1e-3            # down from 5e-3
    mnist_train_batch: int = 128


NON_DEG = NonDegenerateRegimeConfig()


@dataclass(frozen=True)
class ScaleMechanismConfig:
    """Phase 4 (e18): larger-CNN / longer-sequence CIFAR to test the state-dependent
    directional prediction at the scale the original spec cared about.

    Frozen alongside D-040.
    """
    cnn_base_channels: int = 64          # up from 32 (4x parameters)
    cnn_hidden_fc: int = 256             # up from 128
    dataset: str = "cifar10"             # 5-task sequential
    train_epochs_per_task: int = 8
    train_lr: float = 3e-3
    updates_per_source_per_norm: int = 15
    n_seeds: int = 2                     # reduced for GPU budget


SCALE_MECH = ScaleMechanismConfig()


# =============================================================================
# 14. POST-PREREG ADDENDA v1.4 (frozen 2026-09-08 by D-043)
# Task-incremental multi-head fix for the degenerate CL regime + strict per-step
# matched-magnitude enforcement + two-sided controller sweep.
# NONE of v1.0/v1.1/v1.2/v1.3 constants are edited.
# =============================================================================

PREREG_V14_VERSION: str = "1.4.0"
PREREG_V14_FROZEN_DATE: str = "2026-09-08"


@dataclass(frozen=True)
class TILRegimeConfig:
    """Task-incremental protocol used by e19.

    5 tasks of 2 classes each on Split-MNIST and Split-CIFAR-10; the network has
    a shared backbone and PER-TASK 2-way heads; training and evaluation always
    route through the head belonging to the current task. This is the standard
    TIL benchmark and is the correct regime for isolating backbone plasticity
    from head-race dynamics.
    """
    n_tasks: int = 5
    n_classes_per_task: int = 2
    # Split-MNIST (MLP backbone, CPU-fast)
    mnist_hidden: int = 128
    mnist_depth: int = 2
    mnist_lr: float = 3e-3
    mnist_epochs_per_task: int = 4
    mnist_batch: int = 128
    # Split-CIFAR-10 (small CNN backbone, GPU-preferred)
    cifar_cnn_base: int = 32
    cifar_cnn_fc: int = 128
    cifar_lr: float = 5e-3
    cifar_epochs_per_task: int = 6
    cifar_batch: int = 128
    # Probes (permuted-input) for FTD
    n_probes: int = 3
    probe_steps: int = 200
    probe_batch: int = 128
    probe_lr: float = 5e-3
    probe_subset: int = 3000


TIL_REGIME = TILRegimeConfig()


@dataclass(frozen=True)
class StrictMatchedMagnitudeConfig:
    """Every step: applied_delta is renormalized to ‖-lr · ∇plain_task_loss(θ)‖
    at the SAME θ. relative_deviation is the max over the trajectory of
    |‖applied‖ − r_t| / r_t and MUST be ≤ 2% for the arm to keep the
    matched-magnitude label. By construction under the renorm rule this
    condition is met to numerical precision (~1e-6). Any arm exceeding 2% is
    excluded from the Pareto comparison.
    """
    per_step_tol: float = 0.02
    # For each arm we log per-step (r_t, ‖applied‖, deviation) and the summary
    # is (max_dev, mean_dev, dev_p95). Report all three.


STRICT_MATCH = StrictMatchedMagnitudeConfig()


@dataclass(frozen=True)
class E19TradeoffControllerConfig:
    """e19 controller sweep — Pivot 1 (stability–plasticity dial) and Pivot 2
    (sign-reversed plasticity-preserving direction).

    Acceptance bars pre-registered in D-043.
    """
    coefficient_sweep: tuple = (-1.0, -0.75, -0.5, -0.25, 0.0,
                                 +0.25, +0.5, +0.75, +1.0)
    n_seeds: int = 3
    datasets: tuple = ("split_mnist", "split_cifar10")
    baselines: tuple = ("sgd", "ewc", "replay", "cbp")
    # Baseline hyperparameters (mirror v1.3 D-038)
    ewc_lambda: float = 100.0
    ewc_fisher_batch: int = 200
    replay_memory_per_task: int = 200
    replay_mix_ratio: float = 0.5
    cbp_reset_rate: float = 1e-4
    cbp_utility_ema: float = 0.99
    # Directional controller — reuses AdaptiveDirectionalController's u_hazard
    n_permuted_aux: int = 3
    n_semantic_aux: int = 3
    reestimate_every: int = 50
    top_k_project: int = 1
    # Sanity gate (D-043)
    sanity_mnist_acc: float = 0.60
    sanity_cifar_acc: float = 0.35
    sanity_min_tasks_above_chance: int = 3
    sanity_task_chance: float = 0.55           # 2-way TIL binary chance + margin
    # Pivot 1 acceptance
    p1_pareto_tolerance: float = 0.01
    # Pivot 2 acceptance
    p2_min_ftd_reduction_norm: float = 0.005
    p2_min_ftd_reduction_raw: float = 0.010
    p2_max_acc_drop: float = 0.05


E19_TRADEOFF = E19TradeoffControllerConfig()
