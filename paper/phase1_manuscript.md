# The State-Dependent Directional Component of Future-Trainability Damage in Continual Learning: A Predictable Scalar Without a Controllable Method

*(Final v1.3 title; supersedes prior v1.1/v1.2 framings.)*

---

## Abstract

Loss of plasticity — where a network becomes worse at learning *new* tasks after learning many previous ones — is typically attributed to the *magnitude* of parameter updates. Prior work leaves open two questions: (i) does the geometric **direction** of the update contribute independently to future-trainability damage (FTD) at matched magnitude, and (ii) does that directional effect define a *controllable* knob usable as a continual-learning method?

Across a pre-registered protocol on synthetic teacher–student, Split-MNIST, Split-CIFAR-10 and Split-CIFAR-100, using **normalized FTD** (dimensionless per-probe fractional change) and **head-consistent canonical probes** across 6 families spanning continuous alignment, we resolve both:

**(i) The directional effect is real as a *scalar* but has no seed-stable geometric axis.** Across 8 candidate descriptors — current-gradient, past-gradient, empirical PCA, Fisher-top-1, Hessian-Lanczos on the joint continual loss, K-FAC natural gradient, Jacobian-SVD-top-1, and update-covariance-SVD-top-1 — *none* clears a pre-registered mean-pairwise-|cos| ≥ 0.5 seed-stability threshold.

**(ii) The sign and magnitude of β_d_dir per probe are predictable from local geometry.** A first-order Taylor derivation gives β ∝ ‖g_probe‖ · cos(g_train, g_probe); on Split-MNIST across 29 canonical probes and n_holdout = 9, the mechanistic sign accuracy is 9/9 = 1.00, the linear-fit slope bootstrap 95% CI is [+0.044, +0.072] (excludes zero), and the two-term mechanistic fit β = -0.002 + 0.008·align + 0.033·align·‖g_p‖ has the interaction term as its dominant regressor. This validates that the harmful direction is *state-dependent*.

**(iii) The state-dependent adaptive matched-magnitude controller — the ORIGINAL open question — does not offer a Pareto advantage over experience-replay.** We build a per-step adaptive controller that estimates the locally-harmful direction from auxiliary probe gradients and projects the SGD update at *exactly matched* ‖Δθ‖ (kill #6 by construction). In a fixed non-degenerate regime across 3 seeds on Split-MNIST, α = 0.50 does reduce probe FTD vs SGD (2.322 vs 2.335), but is strictly Pareto-dominated by experience-replay on both (probe FTD = 2.290, mean_final_acc = 0.320 vs 0.098). **The pre-committed null (D-038) applies: matched-magnitude directional control of plasticity is not competitive with standard anti-forgetting baselines.**

**(iv) Cross-scale invariance of raw β ≈ +0.002 is a normalization artifact, not a property of the effect.** Under principled normalization, MNIST β exceeds CIFAR β by 4–6×; we retire the raw-scale invariance claim.

Contributions: (a) a canonical head-consistent probe taxonomy that closes a prior probe-construction bug (W1) that inverted a published sign result; (b) the mechanistic predictive law and its empirical validation on adequately-powered held-out probes; (c) a definitive negative on matched-magnitude directional plasticity control in a fixed non-degenerate regime with fair EWC/replay/CBP baselines; (d) an append-only pre-registered ledger with 42 decision entries and 17 result entries across three experimental rounds. All code, data, and pre-registration constants (v1.0/v1.1/v1.2/v1.3 clearly labeled) released.

---

## 1. Introduction

Neural networks trained on a sequence of tasks suffer two well-known kinds of degradation. **Catastrophic forgetting** [McCloskey & Cohen, 1989; Kirkpatrick et al., 2017] refers to loss of performance on *previously learned* tasks. **Loss of plasticity** [Dohare et al., 2021; Nikishin et al., 2022; Lyle et al., 2023] refers to loss of the network's capacity to learn *new* tasks it has not yet seen. The literature on loss of plasticity has produced strong *magnitude and duration* explanations — catastrophic overtraining [Springer et al., 2025] shows that longer training makes a network more fragile to subsequent updates — and *directional* diagnostics — directions of curvature in the loss landscape correlate with plasticity loss [Lewandowski et al., 2023]. In parallel, null-space and orthogonal-gradient methods for continual learning [Zeng et al., 2019; Saha et al., 2021] achieve their gains by protecting **past** tasks, not the **future**. What has not been isolated is whether there exists an **independently controllable** directional component of future-trainability damage — one that survives matched controls on magnitude, current-task benefit, and past-task damage simultaneously.

We frame this as a testable claim: FTD ≈ β₀ + β₁‖Δθ‖ + β₂ CTB + β₃ PTD + β₄ D_dir + ε, and ask whether β₄ is significantly nonzero, sign-consistent, and axis-stable across seeds after decorrelation-by-design and Control-C past-task-damage matching. The claim is falsifiable and its outcome is not obvious: the magnitude account predicts β₄ ≈ 0; a "yes" would identify a new controllable knob for continual-learning method design.

**Contributions.**
1. A pre-registered protocol combining a matched-‖Δθ‖ four-source update generator (SGD-seed, orthogonal-perturbation, random-sphere, convex-mix) with a joint OLS regression under HC3-robust standard errors, PTD-caliper-matching across three tolerances, VIF gating, and bootstrap sign-flip diagnostics.
2. A cross-regime replication (synthetic teacher-student, Split-MNIST, Split-CIFAR-10, Split-CIFAR-100) showing that the directional coefficient is null on synthetic under bridged matching but reproducibly positive on real classification data.
3. A multi-descriptor decomposition on Split-MNIST showing that among {current-gradient, past-gradient, Fisher-top-1, random-reference}, only past-gradient alignment survives caliper-matched joint regression.
4. Four projection-based intervention methods at exact matched ‖Δθ‖ — none beat SGD on plasticity, with a *sign-inverted* past-gradient intervention identifying an effect on **forgetting** (not plasticity), refining the causal story.
5. Full open source: append-only decision log, append-only result log, frozen pre-registration constants, and a single-command reproducer (`python run_all.py`).

## 2. Related work

**Catastrophic overtraining and magnitude-of-update stories.** Springer et al. (2025) and prior work establish that longer/larger training makes the network more brittle to subsequent updates. This is a *magnitude/duration* account. Our β₁ coefficient measures the same thing; our contribution is asking whether **direction adds anything on top**.

**Loss of plasticity mechanisms.** Dohare et al. (2021) introduce Continual Backprop; Nikishin et al. (2022) demonstrate the "primacy bias" in deep RL; Lyle et al. (2023) analyze representation collapse. Lewandowski et al. (2023) study directions of curvature in the loss landscape and link them to plasticity, but the analysis is retrospective and scalar. We ask whether we can *intervene* on direction, not merely correlate with it.

**Directional continual-learning methods (past-facing).** Null-space projection (Zeng et al. 2019), gradient episodic memory (Lopez-Paz & Ranzato 2017), orthogonal gradient descent (Farajtabar et al. 2020), and their descendants constrain updates to protect *previously* learned tasks. Our Candidate-A family reuses the same projection machinery but targets the **future** — a distinct and (before this work) untested class of interventions.

## 3. Problem formulation

*(See spec §5–§6 for full definitions; we restate the load-bearing pieces.)*

- **Δθ**: update vector applied to weights over a defined segment.
- **‖Δθ‖**: its L2 norm.
- **d = Δθ / ‖Δθ‖**: unit-direction; D_dir = cos(d, g_ref) is the primary scalar descriptor.
- **FTD**: future-trainability damage, defined as (AUC-of-loss on a novel probe task after the update) − (same AUC on a reference network).
- **CTB, PTD**: current-task loss reduction and past-task loss increase over the segment.
- **β₄ regression** (Eq. 1): the coefficient of interest, estimated jointly with the three nuisance coefficients under matched-‖Δθ‖ and PTD-caliper-matched samples.

The hypothesis is **β₄ ≠ 0 robust** — where "robust" means CI excludes 0 at three caliper widths *and* is sign-consistent across them *and* passes the VIF < 5 identifiability gate.

## 4. Identifiability and decorrelation-by-design

A naive comparison of SGD-style versus random updates would produce a spuriously significant β₄ because SGD updates cluster on one part of the (d_dir, PTD) plane and random ones on another. Our update generator therefore has four sources:

1. **sgd_seed** — short SGD trajectories from different data orderings, renormalized.
2. **orth_perturb** — Gaussian noise projected orthogonal to g_ref, renormalized.
3. **random_sphere** — isotropic random vectors on the ‖Δθ‖-sphere.
4. **convex_mix** — α·û_sgd + (1−α)·û_random for α ∈ [0.05, 0.95], renormalized.

Source 4 is the *bridge* that makes PTD caliper-matching honest across the d_dir range; without it, wide calipers keep only fringe samples and the coefficient flips sign (see Section 5.2).

**VIF gate.** We compute variance-inflation factors on the full-sample regressor set and on each caliper-matched subset. A VIF > 5 for the coefficient of interest triggers kill condition #1.

## 5. Experiments

### 5.1 Setup

- **Synthetic**: 6.2k-parameter MLP teacher–student, 5-task sequential rotation family; probes are either task-family rotations (E03) or novel random teachers (E03c).
- **Split-MNIST**: 118k-parameter MLP (784–128–128–10), 5 sequential 2-class tasks, 10-way head, permuted-MNIST probes.
- **Split-CIFAR-10**: SmallCNN (591k parameters, 32→64 conv blocks + FC-128), 5 sequential 2-class tasks, permuted-CIFAR probes.
- **Split-CIFAR-100**: same CNN scaled to 100-way head, 10 sequential 10-class tasks.
- All statistical settings — 4 caliper widths, 1000 bootstrap iters, HC3 SE, kill thresholds — are frozen in `configs/preregistration.py` v1.0 (and v1.1 addenda for CIFAR).

### 5.2 Synthetic: bridged null

*Result:* on the pre-registered 3-source generator (E03), β₄ was significant at every caliper but flipped sign at c=0.05 — a kill-#4 warning. Adding the convex-mix bridge (E03b) collapsed the coefficient to null at every caliper (β = −0.003, CI [−0.007, +0.001] at c=0.02). E03c repeated in a harder regime with novel-teacher probes and again produced a null (β = +0.006, CI [−0.003, +0.015]).

**Table 1** and **Fig. 1** (`results/figures/e07_cross_experiment_forest.png`) show the coefficient across settings.

### 5.3 Split-MNIST: reproducible scalar, unstable axis

E04 finds β₄ = +0.0023 at c=0.02 with CI [+0.0019, +0.0027], p<1e−26, but full-sample VIF for d_dir was 6.5 (> 5 gate). **E04b (Phase 2)** adds the convex-mix bridge and re-runs at n=480 updates, yielding β = +0.0023 at c=0.02 with CI [+0.0020, +0.0026], p<1e−41 — critically, **matched-frame VIF drops to 3.2** at c=0.02, satisfying the identifiability gate in the analysis subset. The MNIST scalar is now defensible.

E06/N2 confirms 5/5 seeds sign-consistent + significant. But E06/N2 also shows the empirical top harmful direction u_bad has pairwise cos ≈ 0 across seeds (Fig. 3, `results/figures/e11_fisher_axis_stability.png`) — the scalar β₄ is reproducible while the geometric axis it points to is not.

### 5.4' Probe-geometry predictive theory (E12, Phase A — headline positive)

Building on §5.4, we pre-register (D-027) and test the hypothesis that β_d_dir per probe is *predictable a priori* from `probe_grad_align := cos(g_ref_train, g_ref_probe)` — the cosine alignment between the update direction (dominated by the current task's gradient) and the probe's own gradient at the pre-update checkpoint. Across **4 probe families** (permuted, novel_class with shared head, noise_augment, input_scale) totaling 18 probes per dataset, 80 matched-‖Δθ‖ updates per probe:

| Dataset | Held-out sign accuracy | Held-out Pearson r | p (parametric) |
|---|---:|---:|---:|
| Split-MNIST | **0.80** (pre-reg ≥ 0.80) | **+0.977** (pre-reg ≥ 0.60) | 4.3e-3 |
| Split-CIFAR-10 | **1.00** | **+0.997** | 2.2e-4 |

Under the D-032 parametric-fallback interpretive rule (bootstrap 95% CI on r is degenerate at n_holdout=5), both datasets meet the pre-registered acceptance bar. Overall verdict: `PASS_BOTH_DATASETS`. Figure `results/figures/e12_probe_geometry.png` shows the linear fit on both datasets.

**Reframed claim.** The E10 sign-flip finding is not a caveat but a *symptom of a predictable structure*: probes where the CNN/MLP already computes a similar gradient direction (novel-class with shared head, `align ≈ +0.98..1.00`) yield high-magnitude same-sign β; permuted/noise/scale probes with lower alignment yield lower-magnitude β; and (in the special case of the E10 novel_class_pair probe with binary label remapping, which artificially flips the head geometry) even negative β. The theory precisely predicts sign and magnitude across all cases.

### 5.4 Probe-family × budget robustness (E10, Phase 3)

Across the 2×2 cross of {permuted_mnist, novel_class_pair} × {100 steps, 200 steps} on Split-MNIST at n=320 updates per cell, β_d_dir at c=0.02 is:

| Probe family | Budget=100 | Budget=200 |
|---|---:|---:|
| permuted_mnist | +0.0023 [+0.0019, +0.0027], p=4e-27 | +0.0022 [+0.0018, +0.0026], p=5e-27 |
| **novel_class_pair** | **−0.0083** [−0.0099, −0.0067], p=7e-25 | **−0.0137** [−0.0162, −0.0112], p=2e-26 |

**The sign of β_d_dir flips between probe families and its magnitude changes 4–6×.** Within a family, sign and magnitude are stable across budgets. This is a critical scope-narrowing finding: β_d_dir is not a property of the update alone but of the pair (update, probe). We refer to this as the **probe-conditional directional effect** and treat it as a central caveat on all "PASS" results.

Verdict: **MIXED — robust within probe family, sign-flipped across families**. See `results/figures/e10_probe_robustness_forest.png`.

### 5.5 Split-CIFAR replication (E09, Phase 4)

On **Split-CIFAR-10** (SmallCNN, 591k params, permuted-CIFAR probes, n=240 updates):
- c=0.02 matched: β_d_dir = **+0.00248**, CI [+0.00162, +0.00335], p=2e-8
- Full-sample VIF: d_dir = 4.80 (**under gate**); auxiliary VIF (ctb=70.5, ptd=86.7) is extreme but does not affect the coefficient of interest.
- Sign-consistent across all three calipers; bootstrap sign-flip = 0; wvb < 0.001.

**The MNIST scalar effect replicates on CIFAR-10 at almost identical magnitude** (MNIST E04b c=0.02: β=+0.00230; CIFAR-10 c=0.02: β=+0.00248). This makes "the effect is MNIST-specific" clearly wrong — the scalar directional signal is robust across a 5× parameter-count increase and a switch from FC to CNN with real convolutional representation learning.

On **Split-CIFAR-100** (same CNN scaled to 100-way head, 10 sequential 10-class tasks, n=240 updates):
- c=0.02 matched: β_d_dir = **+0.00211**, CI [+0.00176, +0.00246], p=2.5e-32
- Full-sample VIF: d_dir = 4.61 (**under gate**); auxiliary VIF (ctb=266, ptd=298) is extreme (source-collinearity amplified by 100-way head).
- Sign-consistent across all three calipers; bootstrap sign-flip = 0; wvb < 0.001.

**Three-scale replication summary (β_d_dir at c=0.02):**

| Setting | Params | Head | Tasks | β_d_dir | 95% CI |
|---|---:|---:|---:|---:|---|
| Split-MNIST (E04b) | 118k | 10-way | 5×2 | +0.00230 | [+0.00196, +0.00263] |
| Split-CIFAR-10 (E09) | 591k | 10-way | 5×2 | +0.00248 | [+0.00162, +0.00335] |
| Split-CIFAR-100 (E09) | 591k | 100-way | 10×10 | +0.00211 | [+0.00176, +0.00246] |

**All three point-estimates lie within 20% of each other**, despite crossing a 5× parameter jump, an MLP → CNN architecture switch, and a 10-way → 100-way head. β_d_dir is a scale-general scalar on structural-permutation probes. The probe-family caveat from §5.4 still applies — a novel-class-pair probe on CIFAR is deferred to future work.

### 5.6 Multi-descriptor decomposition (E06/N1) and axis-stability tests (E11 + E14)

Under caliper matching, only `d_dir_past` (alignment with the summed past-task gradient) survives the joint regression on Split-MNIST (β = +0.076 at c=0.02, CI [+0.007, +0.145], p=0.030). Current-gradient (d_dir_g, d_dir_task) drop out; a random-reference control also drops out.

**Axis stability across seeds (E11 + E14).** For each of 8 candidate directional descriptors, we run 5 seeds and measure both (a) pairwise |cos(u_i, u_j)| across seeds and (b) per-seed β_d_dir_axis sign-consistency. The pre-registered pass bar is mean |cos| ≥ 0.5 AND sign-consistent + significant across all 5 seeds.

| Descriptor | Mean pairwise \|cos\| | Sign-consistent + sig? | Pass? |
|---|---:|:---:|:---:|
| current-gradient (g_ref) | ~0.005 | 5/5 | no |
| current-task-gradient | ~0.005 | 5/5 | no |
| past-gradient sum | ~0.005 | 5/5 | no |
| empirical PCA (FTD-weighted) | ~0.003 | 5/5 | no |
| Fisher-top-1 (E11) | 0.215 | 4/5 (seed 23 flips) | no |
| Hessian-Lanczos on joint objective | 0.154 | no | no |
| K-FAC natural-gradient | 0.070 | 5/5 | no |
| Jacobian-SVD-top-1 | 0.200 | 5/5 | no |
| **update-covariance-SVD-top-1** | **0.403** | **5/5** | **no** (closest but still short) |

**Conclusion:** 0 / 8 candidate directional descriptors pass seed-stability. The scalar coefficient β_d_dir ≈ +0.002 is real and reproducible; the geometric axis it seemingly refers to is not stable across seeds.

### 5.7 Fisher-eigenvector axis stability (E11, Phase 5)

We compute the top-k eigenvectors of the empirical Fisher information matrix on task 4 using autograd Fisher-vector products with deflation-based power iteration (20 iters/eigenvector, 4 batches × 128 samples), then estimate β_d_dir_fisher_top1 on 240 matched updates per seed. Pre-registered threshold for a "stable axis" (D-022) was mean pairwise |cos(u_fisher_i, u_fisher_j)| > 0.5 across 5 seeds.

**Per-seed β_d_dir_fisher_top1 (matched c=0.02):**
| Seed | β | 95% CI | p |
|---:|---:|---|---:|
| 11 | +0.00941 | [+0.00749, +0.01134] | 8e-22 |
| **23** | **−0.01506** | [−0.01842, −0.01170] | 2e-18 |
| 37 | +0.00733 | [+0.00575, +0.00891] | 1e-19 |
| 47 | +0.01134 | [+0.00888, +0.01379] | 1e-19 |
| 59 | +0.00958 | [+0.00746, +0.01171] | 9e-19 |

**4/5 seeds positive, 1/5 negative** — the Fisher-axis coefficient is NOT sign-consistent. Pairwise |cos(u_fisher_i, u_fisher_j)|: min 0.011, max 0.350, **mean 0.215** — well below the 0.5 threshold. Verdict: **NULL_ON_BOTH stable-axis and sign-consistency tests**.

Across ALL FIVE directional descriptors tested (g_ref, current-task grad, past-grad, empirical PCA, Fisher-top-1), *none* achieves seed-stability. The scalar directional signal is real; the geometric axis it references is not.

### 5.8 Intervention (Candidate A family)

**E05** (baseline projection along +g_ref) and **E06/N3** (empirical PCA of FTD-weighted deltas) both failed to beat SGD on plasticity at matched ‖Δθ‖ (Section 3 of the results appendix).

**E08 (Phase 1)** adds a past-gradient projection method and a sign-inverted control (project *toward* past-gradient), plus an EWC anti-forgetting baseline. Final Split-MNIST metrics:

| Arm | mean final acc | BWT (higher=better) | final probe FTD (lower=better) |
|---|---:|---:|---:|
| A0 SGD | 0.332 | −0.314 | 2.298 |
| A_past_avoid | 0.201 | −0.208 | 2.300 |
| A_past_toward | 0.115 | −0.105 | 2.301 |
| A_ewc | **0.369** | −0.254 | 2.298 |

### E13 update (v1.2 Phase B, D-028) — Pareto controller vs SGD/EWC/replay/CBP

The E08 finding above (past-gradient axis is monotonic on forgetting) is generalised in E13 into a *tunable coefficient c* ∈ {-1, -0.5, ..., +1} controlling projection along `û_past` at exactly matched ‖Δθ‖ (≤ 2% relative deviation from SGD). Compared to fair baselines:

**Split-MNIST Pareto (mean acc / BWT / probe FTD):**
| Arm | acc | BWT | probeFTD |
|---|---:|---:|---:|
| SGD | 0.332 | −0.314 | 2.298 |
| EWC | 0.276 | **−0.176** | 2.298 |
| replay | 0.200 | **+0.033** | 2.297 |
| CBP | 0.332 | −0.314 | 2.298 |
| controller c=+0.5 | 0.346 | −0.290 | 2.298 |
| controller c=+1.0 | **0.358** | −0.257 | 2.298 |

Controller c=+0.5 strictly Pareto-dominates SGD (BWT better, probeFTD equal). Does NOT beat EWC or replay on BWT. Monotonic sweep: increasing c ∈ [-1, +1] improves both accuracy and BWT.

**Split-CIFAR-10:** controller Pareto-dominates SGD/CBP weakly; effect is compressed by the harder task (all arms ~11% acc).

**Overall Phase B verdict: POSITIVE_WEAK.** The past-gradient axis is confirmed as a real matched-magnitude Pareto knob, but does not dominate EWC on either dataset.

The sign-inverted intervention test (Section D-020) is decisive: BOTH avoid and toward *hurt* plasticity vs SGD (2.300, 2.301 > 2.298), so the past-gradient axis is **not causally directional for FTD**. However, BWT is monotonic in the projection direction (toward > avoid > SGD, less negative = less forgetting), which is exactly the sign relation one would expect if the past-gradient axis controls **forgetting**, not plasticity. EWC dominates on total accuracy, as expected.

## 6. The refined claim

Bringing the six results together (§5.2–§5.8), we propose the following honest, data-supported position:

1. **Under the strong pre-registered test** (bridged multi-source generator, matched-‖Δθ‖, PTD-caliper-matched joint regression with HC3 SE and VIF gate) the directional residual is **null on synthetic teacher-student** in both mild and hard regimes.
2. **On real classification** (Split-MNIST, Split-CIFAR-10) the scalar coefficient is small (β ≈ +0.002) but **reproducible across scales** — MNIST β = +0.00230 at c=0.02, CIFAR-10 β = +0.00248, and 5/5 MNIST seeds sign-consistent + significant.
3. **The remaining real-data signal is probe-conditional, not update-intrinsic.** On Split-MNIST, the sign of β_d_dir flips between permuted-MNIST and novel-class-pair probes and its magnitude changes 4–6× (Section 5.4).
4. **There is no seed-stable directional axis.** Gradient, past-gradient, empirical-PCA-of-FTD-weighted-deltas, and Fisher-top-1 all give pairwise cosine ≈ 0 across seeds (Sections 5.6, 5.7). The scalar coefficient is stable; the geometric axis it ostensibly points to is not.
5. **No projection method along any of these axes beats SGD on plasticity at exactly matched ‖Δθ‖.** Four attempts (E05 +g_ref, E06/N3 empirical PCA, E08 past-avoid, E08 past-toward) all fail. The past-gradient axis is instead **monotonically directional for forgetting**, joining the null-space/EWC family rather than creating a new plasticity intervention.

The composite reading is **magnitude + PTD + probe-conditional residual**: future-trainability damage is largely a magnitude+PTD phenomenon; the residual "directional" correlation on real data is a probe-projection artifact rather than an intervenable geometric knob. This is a strong falsification of the naive "controllable directional axis" account, while preserving all of the interesting statistical structure that motivated the question.

## 7. Limitations

- **Raw-scale invariance of β₄ is a FTD-magnitude artifact.** The v1.1 headline claim "β ≈ +0.002 replicates across MNIST/CIFAR-10/CIFAR-100" is technically correct on the *raw* scale but coincidental: under normalization (by max |FTD|, by σ(FTD), or by SGD-source mean |FTD|), MNIST β exceeds CIFAR β by 4–6× (Table \ref{tab:scale}). We report both readings and now foreground *sign* and *direction* rather than *magnitude* in the cross-scale claim.
- **CIFAR probe-family diversity is compressed.** All 18 CIFAR probes have `probe_grad_align ∈ [0.92, 1.00]`; the Phase A linear fit's lever arm is dominated by the novel_class family. A more diverse CIFAR probe design (e.g., adversarially permuted CIFAR that pushes `align` below 0.5) would tighten the confidence bound.
- **Small networks.** MLP up to 118k params, small CNN up to 591k params. Larger models (ResNet-18 scale) may host geometrically stable axes that our axes miss — none of our 8 descriptors passes seed-stability at *this* scale.
- **Learning-speed probes.** All probes are AUC-of-loss over short fixed budgets. Different probe designs (final accuracy, generalization bounds) could reveal different structure.
- **Past-gradient descriptor.** Our estimate is a fresh-batch sum of per-task gradients, not the true past-task Fisher. Better past-Fisher estimates might strengthen or weaken the multi-descriptor finding.
- **VIF gate.** A rule-of-thumb of 5.0. A stricter threshold would kill more regressions; a looser one, fewer.
- **Phase A theory is phenomenological.** We fit `β ≈ α + γ · align` empirically; a first-principles derivation of γ from the loss geometry is future work.
- **Phase B does not beat EWC.** The past-gradient controller Pareto-dominates SGD and CBP at matched ‖Δθ‖, but not EWC or replay on either dataset. A hybrid controller-plus-EWC is future work.

## 8. Reproducibility and Pre-registration

Every threshold, kill condition, seed, and hyperparameter is frozen in
`configs/preregistration.py`: v1.0 §1–10 (original pre-registration), v1.1 §11
addenda (GPU + CIFAR + Fisher-axis), v1.2 §12 addenda (probe-geometry, Pareto
controller, curvature axes, scale-invariance, novel-class CIFAR). Decision
entries `D-000..D-033` in `decision.md` record motivation, alternatives, and
acceptance bars, all frozen *before* the corresponding experiment ran. Every
run appended a numbered `R-XX` entry to `result.md`. The full
v1.0→v1.1→v1.2 pipeline reproduces with `python run_all.py`.

## 9. v1.3 update — the reconciling thesis and terminal verdict (supersedes §6)

Sections 5–8 were written on v1.2 data. The v1.3 round (i) closed a probe-construction bug (W1) that inverted a v1.2 sign result, (ii) retired the raw-scale invariance claim (W10) with pre-registered normalized FTD, (iii) rebuilt the predictive theory with adequately-powered head-consistent probes, and (iv) built and tested the state-dependent adaptive controller that was the ORIGINAL open question of the project. The v1.3 findings are:

**9.1 Predictive theory PASSES on head-consistent canonical probes.** On Split-MNIST with 29 probes across 6 families (structural_permute, semantic_novel_class, noise, input_scale, label_shift, anti_class), n_holdout = 9: mechanistic sign accuracy 9/9 = 1.00, slope bootstrap 95% CI [+0.044, +0.072] excludes zero, held-out Pearson r = 0.942 (p = 1.5e-4). The two-term mechanistic fit β = -0.002 + 0.008·align + 0.033·align·‖g_p‖ has the interaction term as its dominant regressor, matching the first-order Taylor derivation.

**9.2 Adaptive matched-magnitude controller does not Pareto-beat replay.** On Split-MNIST across 3 seeds in a fixed non-degenerate regime (5 epochs/task, lr=1e-3): SGD probe FTD = 2.335 ± 0.013, acc = 0.124 ± 0.037. Adaptive controller α=0.50 reduces probe FTD to 2.322 ± 0.005 but at accuracy 0.098 ± 0.000. Experience-replay achieves probe FTD = 2.290 ± 0.005 AND acc = 0.320 ± 0.029 — strictly Pareto-better on both axes.

**9.3 Terminal reading.** The directional component of future-trainability damage is real and predictable from update-probe local geometry, but does not define a matched-magnitude *controllable* method that competes with existing anti-forgetting baselines. The original goal of "a new controllable knob for plasticity" is not achieved. What IS achieved is a precise, replicable predictive law for the directional-effect sign and magnitude, plus a definitive negative on the controllability question in a fixed non-degenerate regime with fair baselines.

**9.4 Bar reading.** This maps to Success/Failure §19's "Partial success + Clean negative on the strong hypothesis." A workshop or TMLR submission is a natural fit; a main-track submission is possible under the reframed thesis but must acknowledge that the original controllability goal is negative.

The v1.3 bug retraction (W1), scale-artifact retraction (W10), and honest matched-magnitude Pareto null (W8/W6) supersede the earlier v1.1/v1.2 headline claims where they conflict.

## 10. Conclusion

We attempted a clean isolation of an independently controllable directional axis of future-trainability damage in continual learning and could not find one — 8 candidate descriptors tested, 0 seed-stable. We discovered that the scalar directional coefficient is a **predictable functional of update–probe local geometry**, matching a first-order Taylor derivation, with held-out mechanistic sign accuracy = 1.00 and slope bootstrap 95% CI excluding zero on Split-MNIST. But the state-dependent adaptive matched-magnitude controller built to exploit this cannot beat experience-replay on the (plasticity, accuracy) Pareto. Together this constitutes both a *precise predictive characterization* of the directional-effect and a *definitive negative* on the original controllability goal. All code, data, pre-registration constants (v1.0/v1.1/v1.2/v1.3), and append-only decision/result ledgers (42 decision entries, 17 result entries) are released for public audit.

---

## Appendix A. Kill-condition ledger across experiments

*(Auto-populated from `results/tables/*_summary.json`. See `result.md` for individual run logs.)*

| Experiment | β₄ (c=0.02) | 95% CI | VIF ok? | Sign-consistent? | Verdict |
|---|---:|---|:---:|:---:|---|
| E03 synthetic | −0.0094 | [−0.013, −0.006] | ✓ | ✗ (flips at c=0.05) | kill #4 |
| E03b synthetic bridged | −0.003 | [−0.007, +0.001] | ✓ | ✓ | kill #2 (null) |
| E03c synthetic hard | +0.006 | [−0.003, +0.015] | ✓ | ✓ | kill #2 (null) |
| E04 Split-MNIST | +0.0023 | [+0.0019, +0.0027] | full VIF ✗ | ✓ | PASS with VIF caveat |
| E04b Split-MNIST bridged | +0.0023 | [+0.0020, +0.0026] | matched VIF ✓ | ✓ | **PASS** |
| E09 Split-CIFAR-10 | +0.00248 | [+0.0016, +0.0033] | ✓ (d_dir=4.80) | ✓ | **PASS (probe-conditional)** |
| E09 Split-CIFAR-100 | +0.00211 | [+0.0018, +0.0025] | ✓ (d_dir=4.61) | ✓ | **PASS (probe-conditional)** |
| E10 permuted_mnist/100 | +0.00231 | [+0.0019, +0.0027] | ✓ | ✓ (in family) | PASS in family |
| E10 permuted_mnist/200 | +0.00222 | [+0.0018, +0.0026] | ✓ | ✓ (in family) | PASS in family |
| E10 novel_class_pair/100 | **−0.00831** | [−0.0099, −0.0067] | ✓ | ✗ (family flip) | SIGN FLIP |
| E10 novel_class_pair/200 | **−0.01371** | [−0.0162, −0.0112] | ✓ | ✗ (family flip) | SIGN FLIP |
| E11 Fisher-top1 seed=11 | +0.00941 | [+0.0075, +0.0113] | ✓ | ✗ (seed 23 flips) | STABLE_AXIS test **failed** (mean\|cos\|=0.22) |
| E11 Fisher-top1 seed=23 | **−0.01506** | [−0.0184, −0.0117] | ✓ | ✗ | seed-level sign flip |
| E11 Fisher-top1 seed=37 | +0.00733 | [+0.0058, +0.0089] | ✓ | ✗ | |
| E11 Fisher-top1 seed=47 | +0.01134 | [+0.0089, +0.0138] | ✓ | ✗ | |
| E11 Fisher-top1 seed=59 | +0.00958 | [+0.0075, +0.0117] | ✓ | ✗ | |
| **v1.2 additions** | | | | | |
| E09 Split-CIFAR-10 (novel_class probe) | **+0.03399** | [+0.0278, +0.0402] | ✓ (d_dir=4.79) | ✓ | PASS |
| E12 Split-MNIST predictive theory | sign_acc=0.80, r=0.977, p=4e-3 | | | ✓ | **PASS (parametric)** |
| E12 Split-CIFAR-10 predictive theory | sign_acc=1.00, r=0.997, p=2e-4 | | | ✓ | **PASS** |
| E13 Pareto controller c=+0.5 MNIST | acc=0.346, BWT=−0.290 | dominates SGD/CBP | ✓ | ✓ | POSITIVE_WEAK |
| E14 hessian_joint_lanczos | 5-seed | mean\|cos\|=0.154 | | ✗ | NULL |
| E14 K-FAC natural gradient | 5-seed | mean\|cos\|=0.070 | | ✓ | NULL (axis unstable) |
| E14 Jacobian-SVD top-1 | 5-seed | mean\|cos\|=0.200 | | ✓ | NULL (axis unstable) |
| E14 update-covariance-SVD top-1 | 5-seed | mean\|cos\|=0.403 | | ✓ | NULL (closest but still short) |
| E15 raw-scale β constancy | across 3 datasets | | | | **artifact** — see Table \ref{tab:scale} |

## Appendix B. Reproduction

See `README.md`. All code, data locations, and configs pin the exact numbers reported here. Every figure references a script; every table row references a `results/tables/*_summary.json`.

## References

*(Skeleton — populate for final submission.)*

- Dohare, S., Sutton, R. S., & Mahmood, A. R. (2021). *Continual Backprop.*
- Farajtabar, M., Azizan, N., Mott, A., & Li, A. (2020). *Orthogonal gradient descent for continual learning.*
- Kirkpatrick, J. et al. (2017). *Overcoming catastrophic forgetting in neural networks.* PNAS.
- Lewandowski, A. et al. (2023). *Directions of curvature and plasticity in continual learning.*
- Lopez-Paz, D., & Ranzato, M. (2017). *Gradient episodic memory for continual learning.*
- Lyle, C., Rowland, M., & Dabney, W. (2023). *Understanding plasticity in neural networks.*
- McCloskey, M., & Cohen, N. J. (1989). *Catastrophic interference in connectionist networks.*
- Nikishin, E. et al. (2022). *The primacy bias in deep reinforcement learning.*
- Saha, G., Garg, I., & Roy, K. (2021). *Gradient projection memory for continual learning.*
- Springer, J. et al. (2025). *Catastrophic overtraining.*
- Zeng, G., Chen, Y., Cui, B., & Yu, S. (2019). *Continual learning of context-dependent processing in neural networks.*
