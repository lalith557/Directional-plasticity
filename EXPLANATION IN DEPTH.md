# EXPLANATION IN DEPTH

## Weight-Norm Growth and Update Geometry in Loss of Plasticity — a Matched-Step-Size Study

> Every number in this document is copied from the released result files in
> `results/tables/*.json` (or computed directly from them). Architecture details are
> taken from the source code (`src/plasticity_stress.py`, `experiments/*.py`, `web/`).
> Nothing is estimated or invented. Where a number is a caveat or is weaker than it
> looks, that is said explicitly.

---

## Table of contents

1. [The project in one page](#1-the-project-in-one-page)
2. [Architecture in depth](#2-architecture-in-depth)
3. [Novelty of the project](#3-novelty-of-the-project)
4. [Datasets used and why](#4-datasets-used-and-why)
5. [How efficiently deep learning is used](#5-how-efficiently-deep-learning-is-used)
6. [Major deep-learning components and algorithms](#6-major-deep-learning-components-and-algorithms)
7. [All the results](#7-all-the-results)
8. [Why each metric is used](#8-why-each-metric-is-used)
9. [Negative results — why they happen and whether they are beneficial](#9-negative-results--why-they-happen-and-whether-they-are-beneficial)
10. [Improvements that can be made](#10-improvements-that-can-be-made)
11. [Scope and limitations](#11-scope-and-limitations)
12. [Backbone files and what each one is for](#12-backbone-files-and-what-each-one-is-for)
13. [How to reproduce](#13-how-to-reproduce)

---

## 1. The project in one page

### 1.1 The problem

A neural network trained on a **stream of tasks** (one after another, without going back)
fails in two different ways:

| Failure | What goes wrong | Direction in time |
|---|---|---|
| **Catastrophic forgetting** | Performance on *old* tasks collapses after learning new ones | Looks **backward** (the past) |
| **Loss of plasticity** | The network gradually becomes *unable to learn new tasks* as well as it used to — even tasks unrelated to earlier ones | Looks **forward** (the future) |

This project studies the **second** problem: loss of plasticity.

### 1.2 The research question

Prior work explains loss of plasticity mostly through **magnitude**: weights grow too
large, move too far from initialization, or units go "dormant" (stop activating). A
separate line of work (OGD, GPM, OWM) constrains the **direction** of updates — but only
to protect *past* tasks.

A normal optimizer changes **direction and magnitude at the same time**, so their effects
are tangled. The project asks:

> **If the size of every update is held exactly fixed, can changing only its direction
> preserve plasticity?**

To answer that, every "directional" update is **renormalized to exactly the length of the
plain-SGD step** at the same parameters (the *matched-step-size control*). Only the
direction can then differ.

### 1.3 The two phases of the project (honest history)

| Phase | Experiments | What was tested | Outcome |
|---|---|---|---|
| **Phase 1 — "Directional Plasticity"** (v1.0 → v1.4) | `e01`–`e19` | Does a *controllable directional axis* of future-trainability damage exist? Tested with a matched-magnitude update generator, a β₄ regression controlling for magnitude / current-task benefit / past-task damage, Split-MNIST, Split-CIFAR-10/100, 8 geometric descriptors, and two directional controllers. | The scalar directional correlation is real and reproducible, but **no stable, controllable axis exists**, and no directional controller beat standard baselines. The test-beds were also either collapsed (class-incremental, SGD ≈ 12 % accuracy) or too easy (task-incremental, SGD ≈ 91 %) — **there was no real plasticity loss to cure**. |
| **Pivot (decision D-044)** | — | "You cannot test a cure without first verifying the patient is sick." | A regime that genuinely induces loss of plasticity was locked: long permuted MNIST, a tiny width-16 MLP, 100 tasks. |
| **Phase 2 — the paper** (v1.5) | `e20`–`e27` + Cor. 1 audit | Weight-norm growth vs plasticity decay, matched-step directional ablations, a fixed-step norm identity, a radial sweep, magnitude baselines, a per-unit method (ANR), and five regimes to find boundary conditions. | **Magnitude control preserves plasticity better than matched-step directional control**, and the norm–plasticity link is **regime-dependent** (strong under capacity stress, absent where plasticity loss is mild). |

The final deliverables are:

- **Research code** — `src/` and `experiments/` (PyTorch).
- **The paper** — `paper/main(iconcise).tex` (IEEE ICA 2026 format, double-blind, compiled on Overleaf).
- **Plasticity Lab** — `web/`, an interactive React dashboard over the real result files.

### 1.4 The answer, in four sentences

1. In the stress regime, final weight norm and plasticity decay are strongly associated
   across seven update rules (Pearson **r = +0.97**), and turning an L2-init knob up
   moves both together.
2. A simple identity shows that a fixed-length step **orthogonal** to the weights *must*
   increase the squared norm by exactly η², and *reducing* the norm requires
   ⟨θ, d⟩ ≤ −η/2 — so direction can only lower the norm by spending part of the step on
   pointing inward. This was verified on the real runs to a relative error of **3×10⁻⁸**.
3. A radial sweep confirms that inward steering lowers the norm monotonically, but its best
   setting (accuracy **0.763**) still trails magnitude methods (L2-init **0.819**,
   Shrink & Perturb **0.799**), and strong inward steering collapses learning.
4. The norm–decay association weakens at width 64 (**+0.45**) and turns negative where
   plasticity loss is mild (width 128 **−0.52**, rotated MNIST **−0.94**, CNN **−0.82**) —
   so weight-norm growth is a **regime-specific correlate**, not a universal cause.

### 1.5 Where this matters (use cases)

- **Lifelong / continual learning systems** — robots, edge devices and on-device models
  that keep learning after deployment must stay trainable. The results say: in a
  capacity-stressed model, *control the weight magnitude* (e.g. L2-init) rather than trying
  to steer update direction at fixed step size.
- **Long-running fine-tuning pipelines** — models updated repeatedly over months
  (recommendation, personalization) can drift into the same rigidity.
- **Diagnostics** — weight norm, dormant-unit fraction and effective rank are cheap
  signals that a model is losing plasticity; the regime study shows *when* those signals
  can be trusted.
- **Research hygiene** — the matched-step control is a reusable experimental tool for
  separating "the update was a different size" from "the update pointed somewhere else".

---

## 2. Architecture in depth

### 2.1 System architecture (end-to-end)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ DATA LAYER                                                                    │
│  Deep Learning Dataset1/ (raw MNIST IDX)  ──►  src/datasets.py  load_mnist()  │
│                                                   │                           │
│              ┌────────────────────────────────────┴───────────────┐           │
│              ▼                                                    ▼           │
│   PermutedMNIST (100 tasks, fixed pixel permutation)   RotatedMNIST (100 tasks, │
│   on-the-fly, 6 000-image train subset per task)       angle ~ U[0°,180°))     │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ MODEL LAYER   PlasticMLP 784-h-h-10 (h = 16 / 64 / 128)   |   PlasticCNN       │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ ONE SHARED TRAINING LOOP  run_arm() / run_arm_cnn()  (src/plasticity_stress.py)│
│                                                                              │
│  for each task t = 1..100:                                                   │
│     for 60 SGD steps (batch 128):                                            │
│        θ_before = flatten(all params)                                        │
│        g = ∇ loss    →   SGD step  Δθ = −η₀·g,   η = ‖Δθ‖                    │
│        UPDATE RULE   →   Δθ'  (orthogonal / inward / random / angle / = Δθ)  │
│        matched step  →   ‖Δθ'‖ = η   (directional rules only)                │
│        θ ← θ_before + Δθ'          (+ norm projection / CBP reset if armed)   │
│     task-boundary ops: Shrink&Perturb  or  ANR (cap + revive)                 │
│     evaluate a_t on the full test set under task t's permutation              │
│     diagnostics: ‖θ‖, dormant fraction, effective rank                        │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ EXPERIMENT SCRIPTS  experiments/e20 … e27   (arms × seeds, fixed seeds)        │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ RESULTS   results/tables/*.json  (per-seed raw + mean/std)                     │
│           results/figures/*.png  (publication figures, dpi 300)                │
└──────────────┬───────────────────────────────────────────────┬───────────────┘
               ▼                                               ▼
┌────────────────────────────────┐          ┌──────────────────────────────────┐
│ PAPER  paper/main(iconcise).tex│          │ PLASTICITY LAB  web/               │
│  + references.bib + figs/      │          │  build_data.py → plasticity.json   │
│  compiled on Overleaf (IEEEtran)│         │  → React pages (Vite, Recharts)    │
└────────────────────────────────┘          └──────────────────────────────────┘
```

**Design principle:** *every* update rule runs through the **same** training loop. The only
thing that changes between arms is one line — how `Δθ'` is computed from the SGD step.
That makes comparisons fair: same data order, same seeds, same step budget, same
evaluation.

### 2.2 Neural-network architectures

#### (a) PlasticMLP — the primary model (`src/plasticity_stress.py`)

A plain ReLU multilayer perceptron with **one shared 10-way head**:

| Layer | Shape | Parameters (width 16) |
|---|---|---:|
| Input | 784 (28×28 image, flattened, scaled to [0,1]) | — |
| Hidden 1 | Linear 784 → 16, ReLU | 12 560 |
| Hidden 2 | Linear 16 → 16, ReLU | 272 |
| Head | Linear 16 → 10 (logits) | 170 |
| **Total** | | **13 002** (backbone = hidden layers = 12 832) |

Wider variants used for the regime study (same depth, same protocol):

| Width h | Total parameters | Used in |
|---:|---:|---|
| 16 | 13 002 | e20, e22 (λ, ANR), e23, e26, e27, Cor. 1 audit, rotated-MNIST MLP (e24) |
| 64 | 55 050 | e22 (width study) |
| 128 | 118 282 | e24 (width study) |

**Why so small?** Width 16 is *deliberately* capacity-constrained. Small networks show
loss of plasticity quickly and clearly (a 13.3-point drop over 100 tasks), which makes
the phenomenon measurable on one laptop GPU in minutes. Wider networks are then used to
show *where the effect stops*.

**Why a single shared head?** Permuted MNIST keeps the same labels (0–9) for every task,
so one head can serve all tasks. This avoids the "head race" confound that broke the
Phase-1 class-incremental test-bed (decision D-043): a decline in accuracy here reflects
the backbone's ability to adapt, not a head being overwritten.

#### (b) PlasticCNN — the second architecture (rotated MNIST only)

| Layer | Shape | Parameters |
|---|---|---:|
| Input | 1×28×28 | — |
| Conv 1 | 3×3, 1 → 16 channels, padding 1, ReLU, MaxPool 2 → 16×14×14 | 160 |
| Conv 2 | 3×3, 16 → 32 channels, padding 1, ReLU, MaxPool 2 → 32×7×7 | 4 640 |
| FC | 1 568 → 64, ReLU (penultimate features) | 100 416 |
| Head | 64 → 10 | 650 |
| **Total** | | **105 866** |

The CNN is used only on rotated MNIST, because permuting pixels destroys the spatial
structure that convolutions rely on.

### 2.3 The training loop and the matched-step control

For every optimizer step (pseudo-code of `run_arm`):

```
θ_before = concat(all parameters)               # the GLOBAL parameter vector
g        = ∇θ CrossEntropy(model(x), y)          # (+ L2-init penalty if that arm)
Δθ       = −η₀ · g                                # what plain SGD would do
η        = ‖Δθ‖                                   # SGD step length at this θ

Δθ'      = rule(Δθ, θ_before)                    # the only arm-specific line
           # directional rules end with:  Δθ' ← Δθ' · (η / ‖Δθ'‖)   (matched step)

θ        = θ_before + Δθ'
record   ‖Δθ'‖, η, cos(Δθ', Δθ), cos(Δθ', θ)      # audit + alignment diagnostics
```

- `η₀ = 0.2` (MLP) / `0.1` (CNN), batch 128, **60 steps per task**, **100 tasks**
  → **6 000 SGD steps per run**.
- Weights are **warm-started**: one random initialization before task 1, then the same
  weights are carried across every task boundary (no reinitialization). This is what lets
  rigidity accumulate.
- After each task, `a_t` = accuracy on the **full 10 000-image MNIST test set** under that
  task's permutation (rotated MNIST: a fixed 2 000-image test subset under that task's
  rotation).
- **Matched-step audit:** the code records the maximum per-step relative deviation
  `|‖Δθ'‖ − η| / η`. The tolerance is 10⁻⁴; the observed maximum is **2.2×10⁻⁷** for the
  two Table-I directional arms (2.3×10⁻⁷ over the radial sweep) — float precision.

### 2.4 The update rules ("arms")

| Arm | Family | What it does (from code) | Matched step? |
|---|---|---|:---:|
| **SGD** | control | `Δθ' = Δθ` | reference |
| **L2-init** (Kumar et al.) | magnitude | adds `λ‖θ − θ₀‖²` (λ = 10⁻²) to the loss on the *live* parameters, pulling weights back toward initialization | no |
| **Shrink & Perturb** (Ash & Adams) | magnitude | at every task boundary: `θ ← 0.7·θ + 0.01·noise` | no |
| **Continual Backprop (CBP)** (Dohare et al.) | unit reset | per-step probability ρ = 10⁻² of re-initializing the lowest-utility hidden unit (utility = EMA 0.99 of mean abs activation); simplified, no maturity threshold | no |
| **Orthogonal steering** (`dir_control`, c = 1) | directional | removes the radial component: `Δθ' ∝ Δθ − (Δθ·θ̂)θ̂`, then renormalizes to η | **yes** |
| **Inward steering** (`dir_inward`, γ) | directional | adds an inward bias: `Δθ' ∝ Δθ − γ·η·θ̂`, then renormalizes to η (γ = 0.6 in the main table; swept 0 → 1) | **yes** |
| **ANR — Adaptive-Norm-Revive** (this project) | per-unit magnitude | at every task boundary: (i) rescale only hidden units whose incoming-weight norm exceeds τ = 3.0; (ii) re-initialize (Kaiming) only dormant units and zero their outgoing weights | no |
| **Norm projection** (e23 MLP, e25 CNN) | magnitude | plain SGD step, then if the global ‖θ‖ > 6.0, rescale θ back to 6.0 | no |
| **Random direction** (e23, e25) | directional control | a fresh uniformly random unit direction, scaled to η | **yes** |
| **Angle-controlled** (e23, e25) | directional control | a direction with a prescribed cosine to the SGD step (1.0 → 0.0), scaled to η | **yes** |

### 2.5 Diagnostics recorded after every task

| Diagnostic | Definition in code | Why it is logged |
|---|---|---|
| Task accuracy `a_t` | test accuracy on task t right after training on it | the operational measure of "can it still learn" |
| Global weight norm ‖θ‖ | L2 norm of the backbone (hidden-layer) parameters for the MLP; all parameters for the CNN | the mechanism variable under test |
| Dormant fraction | fraction of hidden ReLU units whose mean absolute activation on a 1 000-image probe batch is < 10⁻³ | "dead neurons" are a known plasticity failure |
| Effective (stable) rank | ‖F‖²_F / ‖F‖²₂ of the centred penultimate features on the probe batch | detects representation collapse |
| Gradient / radial alignment | cos(Δθ', Δθ) and cos(Δθ', θ) per step | verifies what each rule actually did |
| Matched-step deviation | max_t abs(‖Δθ'‖ − η) / η | proves the directional comparison is honest |
| Cor. 1 audit fields | ‖θ₀‖², ‖θ_T‖², Σ‖Δθ'‖² on the global vector | numerical check of the identity (added for the camera-ready) |

### 2.6 The theory component — a fixed-step norm identity

For a step of fixed length η along a unit direction d (‖d‖ = 1):

```
‖θ + η d‖²  =  ‖θ‖²  +  2η ⟨θ, d⟩  +  η²                 (Proposition 1 — an identity)
```

- **Corollary 1 (orthogonal steps):** if ⟨θ, d⟩ = 0, the squared norm rises by **exactly
  η²** — for every orthogonal direction. A fixed-length step orthogonal to θ *cannot*
  reduce the norm.
- **Corollary 2 (reducing the norm costs budget):** ‖θ'‖ ≤ ‖θ‖ **iff** ⟨θ, d⟩ ≤ −η/2,
  i.e. the inward radial component must be at least **η / (2‖θ‖)** of the unit direction,
  leaving at most √(1 − (η/(2‖θ‖))²) for task descent.
- **Radial–tangential reading:** split d into a radial part (along θ) and a tangential
  part (perpendicular). Only the radial part changes the norm to first order; whatever is
  spent there is unavailable for descent.

The paper calls this an **identity**, not a deep theorem — it is elementary algebra.
Its value is that it turns "direction didn't help" into "here is exactly *why* a
fixed-length directional step is constrained", and it predicts both directional ablation
outcomes before they are run.

### 2.7 Plasticity Lab — the web application architecture (`web/`)

```
results/tables/*.json ──► web/scripts/build_data.py ──► web/public/data/plasticity.json
                          (copies values, NaN → null,        (one normalized file)
                           flags Match-dev N/A, no
                           fabricated curves)                          │
                                                                       ▼
          main.tsx (HashRouter) ─► App.tsx (17 routes) ─► AppShell (top bar, nav, theme)
                                                                       │
                     DataContext.tsx  fetch(plasticity.json) ──────────┤
                                                                       ▼
   pages/  Overview · R20 · λ-Sweep · Radial-Sweep · Width · Rotated MNIST · CNN ·
           Norm↔Plasticity · Update Geometry · Ablations · Comparative · Run Details ·
           Configuration · Datasets · Checkpoints · Raw Results · Paper
                                                                       │
   components/  ResultTable · charts (Recharts) · GeometryDiagram (SVG) · SweepSlider ·
                RegimeView · MetricCard · DataGate (loading / error states)
```

- **Stack:** Vite 5 + React 18 + TypeScript + react-router-dom (HashRouter, so a page
  refresh works without a server rewrite) + Recharts.
- **Data rule:** the UI never computes or invents experimental values — it only reads
  `plasticity.json`. Missing values show as "Not available", never 0. Match-dev is shown
  only for matched-step arms. Correlations are labelled "descriptive, not causal".
- **Run:** `npm --prefix web run dev`.
- **Experimental Studio:** the same app also hosts the Studio pages, backed by a local FastAPI
  engine (`studio/`) that launches real `run_arm` jobs on the GPU. See `docs/STUDIO_GUIDE.md`.

### 2.8 Paper build pipeline

`paper/main(iconcise).tex` (IEEEtran conference class, two-column) + `paper/references.bib`
(37 entries) + `paper/figs/` (2 PNGs; Fig. 1 is drawn in TikZ inside the `.tex`). It is
compiled on **Overleaf** (pdfLaTeX + BibTeX): zip those three items, upload the zip as a new
project and recompile. The compiled paper is `Research Paper.pdf` in the repository root.

### 2.9 Phase-1 architecture (for completeness)

Phase 1 used a different measurement machine:

```
checkpoint after task k ──► update_generator.py: many updates Δθ with the SAME ‖Δθ‖
                            (SGD-seeded, orthogonal-perturbed, random-sphere, convex-mix)
                        ──► apply each update, measure on held-out probe tasks:
                            FTD (future-trainability damage), CTB (current-task benefit),
                            PTD (past-task damage), D_dir (directional descriptor)
                        ──► stats.py:  FTD = β0 + β1‖Δθ‖ + β2·CTB + β3·PTD + β4·D_dir
                            OLS with HC3 robust errors, VIF gate (< 5), PTD caliper
                            matching, bootstrap CIs
```

The question was whether **β₄ ≠ 0** after controlling for magnitude, current benefit and
past damage at the same time. Constants, seeds and kill conditions were frozen in advance
in `configs/preregistration.py`.

---

## 3. Novelty of the project

What is genuinely new — and what is not — stated plainly:

| # | Contribution | Why it is novel |
|---|---|---|
| 1 | **Matched-step-size control for future trainability** | Directional / subspace methods (OGD, GPM, OWM) protect the *past*. To our knowledge, no prior study isolated **direction at exactly constant magnitude** and asked whether it preserves *future* trainability. Every directional arm here is renormalized to the plain-SGD step at the same parameters, verified to ≤ 2.3×10⁻⁷. |
| 2 | **A fixed-step, global-norm identity with an empirical audit** | The algebra is elementary (the paper says so). What is new is using it as a *prediction* for continual-learning update rules (orthogonal ⇒ norm must grow; inward ⇒ budget is consumed) and then **confirming it on the real runs**: ‖θ_T‖² − ‖θ₀‖² = Ση_t² to a relative error of 3×10⁻⁸. |
| 3 | **A radial sweep turning two endpoints into a trade-off curve** | Instead of one or two directional settings, γ ∈ {0, 0.2, …, 1.0} maps the full geometry → norm → accuracy trade-off, showing direction *can* control the norm, but only at an accuracy cost. |
| 4 | **The norm–plasticity link is regime-dependent** | The same seven-arm protocol across five regimes (width 16/64/128, rotated MNIST MLP and CNN) shows the association is strong under capacity stress (+0.97) and vanishes where plasticity loss is mild. This is a boundary condition, not a universal law. |
| 5 | **An honest negative for a targeted per-unit method (ANR)** | ANR nearly eliminates dormant units but does not beat global shrinking. Its component ablation localizes the failure: dormant-unit revival helps, the per-unit norm cap hurts. |
| 6 | **Pre-registered, append-only methodology** | Decision rules were fixed before the confirmatory runs (e.g. D-047: "if ANR doesn't Pareto-beat the best magnitude baseline on ≥ 4/5 seeds, publish the mechanism instead"). Decision entries D-000 … D-047 and result entries R01 … R20 are kept in `decision.md` / `result.md`. |
| 7 | **An interactive, integrity-preserving results lab** | Plasticity Lab exposes every table, sweep and regime from the actual JSON files, with N/A and "descriptive, not causal" built into the UI. |

**Not novel (prior work, used as baselines):** L2-init, Shrink & Perturb, Continual
Backprop, permuted MNIST, dormant-neuron and effective-rank diagnostics.

---

## 4. Datasets used and why

### 4.1 MNIST — the base dataset (`Deep Learning Dataset1/`)

- 70 000 handwritten digits (60 000 train / 10 000 test), 28×28 grayscale, 10 classes.
- Loaded **directly from the raw IDX files** by `src/datasets.py` (no download, no
  `torchvision`) → fully reproducible offline.

### 4.2 Permuted MNIST — the primary benchmark

- **Construction:** each of the 100 tasks applies one fixed random permutation of the 784
  pixels to *every* image; labels are unchanged. Each task trains on its own fixed
  6 000-image subset and is tested on the full 10 000-image test set under its permutation.
  Permutations are applied **on the fly**, so the 100 tasks never need 100 copies of the data.
- **Why it was chosen:**
  1. It is **the canonical loss-of-plasticity benchmark** (used by Dohare et al. 2024 and
     Kumar et al.), so results are comparable to the literature.
  2. Tasks are **unrelated to each other** (a new permutation is a brand-new input
     mapping), so a falling `a_t` measures *loss of ability to learn* rather than
     interference between similar tasks.
  3. The **same label space** lets one shared head serve all tasks → no head-race confound
     (the bug that invalidated Phase 1's class-incremental test-bed).
  4. It is **small and fast**: 100 tasks × 5 seeds × 7+ arms is feasible on one laptop GPU.
  5. With a width-16 MLP it **reliably induces the disease**: SGD drops 13.3 points while
     its norm grows from ≈ 3.3 to 20.4 and dormancy reaches 21 %.

### 4.3 Rotated MNIST — the second task geometry

- Each of 100 tasks rotates all images by a fixed angle drawn uniformly from
  [0°, 180°) (affine-grid sampling, no extra libraries); 6 000-image training subset,
  2 000-image test subset.
- **Why:** it changes the input **smoothly** instead of destroying spatial structure.
  This (a) tests whether conclusions hold under a *different kind* of task shift and
  (b) makes a **CNN** meaningful, giving a second architecture. It turned out to be a
  *mild* stream where SGD barely loses plasticity — exactly what exposed the boundary
  condition.

### 4.4 Phase-1 datasets (not used in the final paper)

| Dataset | Role in Phase 1 | Why |
|---|---|---|
| Synthetic teacher–student | primary discovery (e03) | full control, unlimited matched samples, known ground truth |
| Split-MNIST (5 two-class tasks) | real-data replication (e04, e04b) | guards against a synthetic-only artifact |
| Split-CIFAR-10 / CIFAR-100 (`cifar-10/`, `cifar100/`) | scale stress test (e09, e16) | harder natural images; checks that the scalar effect scales |

**Why the final paper is MNIST-family only:** the final study needed a regime where
plasticity loss is *strong and reproducible* across many arms, seeds and sweeps on one GPU.
That is stated as a limitation (no CIFAR/ImageNet scale) — see §11.

---

## 5. How efficiently deep learning is used

### 5.1 What is efficient

| Aspect | How | Measured effect |
|---|---|---|
| **Right-sized models** | 13 002-parameter MLP for the main regime | the disease appears within 100 tasks; one 100-task run (6 000 SGD steps + 100 evaluations) takes **≈ 27–32 s** on an RTX 4070 laptop GPU (measured in the Cor. 1 audit) |
| **GPU auto-detection** | `src/device.py` picks CUDA if available, else CPU, and stamps the hardware profile into every JSON | the same code runs on CPU or GPU; results are traceable to hardware |
| **One training loop for every rule** | update rules act on a flat parameter vector (`_all_flat` / `_set_flat`) | no duplicated training code; arms differ by one line → fewer bugs, fair comparison |
| **On-the-fly task construction** | permutations / rotations applied when a task is requested | memory stays at one copy of MNIST instead of 100 |
| **Bounded budget** | fixed 6 000-image subset, 60 steps per task | every arm gets exactly the same compute; nothing is tuned per arm |
| **Cheap diagnostics** | dormancy / rank on a 1 000-image probe batch; test evaluation in 2 000-image chunks | diagnostics add little time and bounded memory |
| **Deterministic seeding** | fixed seeds (20260824–20260828), seeded generators for data order, CBP and random directions | re-running e20 gives bit-identical summaries (checked); the Cor. 1 audit re-ran the orthogonal arm and reproduced its final norm (25.1) |
| **Exact matched-step renormalization** | one norm and one scale per step | costs O(d) per step but makes the core comparison exact (≤ 2.3×10⁻⁷) |
| **No heavy frameworks** | plain PyTorch; no Avalanche / continuum / W&B | auditable controls, no hidden preprocessing |

### 5.2 Where efficiency could be better (honest)

- **Flatten / unflatten every step:** the loop concatenates all parameters into one vector
  and copies them back each step. Correct and simple, but it costs an extra full copy of
  the weights per step. In-place `torch._foreach_*` updates would avoid it.
- **Host-to-device copies per batch:** mini-batches are moved CPU → GPU every step. Keeping
  each task's 6 000 images on the GPU would remove that transfer.
- **No mixed precision / no batching of seeds:** seeds and arms run sequentially; `torch.func.vmap`
  or running several seeds in one batch would use the GPU far better (it is mostly idle
  with a 13k-parameter model).
- **Per-task curves are not persisted:** only aggregates are saved, so regenerating curve
  figures requires a full GPU re-run (≈ 30 s × arms × seeds).

---

## 6. Major deep-learning components and algorithms

| Component / algorithm | Where it is used | Role in the project |
|---|---|---|
| Multilayer perceptron (fully connected, ReLU) | `PlasticMLP` | primary model (widths 16 / 64 / 128) |
| Convolutional network (3×3 conv, ReLU, 2×2 max-pool, FC) | `PlasticCNN` | second architecture on rotated MNIST |
| Cross-entropy loss + softmax head | all runs | classification objective |
| Backpropagation / autograd | all runs | computes the gradient g that every rule starts from |
| Stochastic gradient descent | all runs | base optimizer (η₀ = 0.2 MLP / 0.1 CNN, batch 128) |
| Kaiming (He) initialization | PyTorch default for Linear / Conv; ANR revival; CBP resets | weight initialization and re-initialization |
| Warm-starting across tasks | the continual-learning protocol | the condition that produces loss of plasticity |
| L2-init regularization | `l2_init` arm, λ-sweep | magnitude control toward initialization |
| Shrink & Perturb | `shrink_pert` arm | shrink weights + add noise at task boundaries |
| Continual Backprop | `cbp` arm (`CBPReset`) | utility-based re-initialization of low-utility units |
| Adaptive-Norm-Revive (new) | `adaptive_norm_revive` arm + ablation | per-unit norm cap + dormant-unit revival |
| Gradient projection / steering | orthogonal + inward rules | change direction only (radial component removed or biased inward) |
| Matched-step renormalization | every directional rule | isolates direction from magnitude |
| Norm projection onto a ball | `norm_project` arm | direct hard magnitude constraint (‖θ‖ ≤ 6) |
| Random / angle-controlled directions | e23, e25 | controls for "any off-gradient direction" |
| Dormant-neuron detection | diagnostics + ANR | measures and repairs dead ReLU units |
| Stable rank via SVD | diagnostics | representation-collapse measure |
| Permuted / rotated task streams (`affine_grid`, `grid_sample`) | `PermutedMNIST`, `RotatedMNIST` | continual-learning benchmarks |
| *(Phase 1)* EWC (Fisher-weighted penalty), experience replay | `src/baselines.py`, e08/e13/e17/e19 | forgetting baselines |
| *(Phase 1)* Fisher / Hessian-Lanczos / K-FAC / Jacobian-SVD directions | e11, e14 | candidate "directional axes" (8 descriptors) |
| *(Phase 1)* Multi-head task-incremental training | `src/cl_harness.py`, e19 | fixed the degenerate class-incremental regime |
| Statistics: Pearson / Spearman, leave-one-out, paired-seed differences, OLS + HC3, VIF, bootstrap | analysis | quantifies associations without overclaiming |

---

## 7. All the results

All Phase-2 runs: permuted MNIST unless stated, **5 seeds**, mean ± std (std over seeds).

### 7.1 Main result — E20, width-16 permuted MNIST, 100 tasks (Table I of the paper)

| Method | Family | Mean acc ↑ | Decay ↓ | Final ‖θ‖ | Dormant | Eff. rank | Match dev. |
|---|---|---:|---:|---:|---:|---:|---:|
| **L2-init** (λ = 10⁻²) | magnitude | **0.819 ± 0.003** | 0.047 ± 0.012 | 6.6 | 11.3 % | 3.52 | — |
| Shrink & Perturb | magnitude | 0.799 ± 0.010 | **0.046 ± 0.016** | 4.2 | 0.6 % | 3.14 | — |
| ANR (ours) | per-unit magnitude | 0.735 ± 0.010 | 0.090 ± 0.027 | 7.9 | 0.6 % | 2.68 | — |
| SGD | control | 0.744 ± 0.015 | 0.133 ± 0.036 | 20.4 | 21.2 % | 2.93 | 0 (ref.) |
| Continual Backprop | resets | 0.728 ± 0.010 | 0.147 ± 0.019 | 19.1 | 22.5 % | 2.96 | — |
| Orthogonal steering | matched-step directional | 0.703 ± 0.007 | 0.183 ± 0.040 | 25.1 | 28.7 % | 2.30 | 2.2×10⁻⁷ |
| Inward steering (γ = 0.6) | matched-step directional | 0.429 ± 0.005 | 0.051 ± 0.006 | 2.3 | 7.5 % | 1.86 | 2.0×10⁻⁷ |

**Loss of plasticity is present:** plain SGD decays **0.133 ± 0.036** (13.3 points) from its
peak while its norm grows from ≈ 3.3 at initialization to **20.4** (about 6×) and 21.2 % of
hidden units go dormant.

### 7.2 Association between norm growth and decay (across the 7 arms)

| Statistic | Value |
|---|---|
| Pearson r (final ‖θ‖ vs decay, n = 7 arm means) | **+0.970** |
| Spearman ρ | +0.857 |
| Per-seed arm-level r (5 seeds) | 0.942, 0.918, 0.841, 0.906, 0.886 (mean **0.899**) |
| Leave-one-arm-out r | 0.950 – 0.979 (dropping inward → 0.970; dropping orthogonal → 0.950) |
| Effective rank vs mean accuracy | r = +0.887 |
| Effective rank vs decay | r = −0.214 (weak) |

The association is robust to seed choice and is not driven by any single arm. It is an
**association across arms**, not a causal estimate.

### 7.3 Intervention — L2-init strength sweep (E22-λ, 5 seeds)

| λ | 0 (SGD) | 10⁻³ | 3×10⁻³ | **10⁻²** | 3×10⁻² | 10⁻¹ |
|---|---:|---:|---:|---:|---:|---:|
| Final ‖θ‖ | 20.4 | 12.1 | 9.0 | 6.6 | 5.3 | 4.3 |
| Mean acc | 0.744 | 0.776 | 0.778 | **0.819** | 0.809 | 0.693 |
| Decay | 0.133 | 0.078 | 0.076 | 0.047 | **0.043** | 0.080 |

Turning up norm control drives the norm down **monotonically**; decay falls to a floor at
λ ∈ [10⁻², 3×10⁻²]; too much regularization (λ = 10⁻¹) hurts both → a **U-shape**.
This is within-family interventional evidence consistent with the association (not
definitive causal proof — regularization has effects other than norm reduction).

### 7.4 Directional ablations and the Cor. 1 audit

- **Orthogonal steering** has the *largest* norm (25.1) and the *worst* decay (0.183) — it
  cannot reduce the norm, as Corollary 1 predicts.
- **Inward steering** drives the norm to 2.3 but collapses accuracy to 0.429 — the budget
  cost predicted by Corollary 2.
- **Cor. 1 audit (orthogonal arm, 5 seeds, `results/tables/cor1_orthogonal_audit.json`):**

| Seed | ‖θ₀‖ | ‖θ_T‖ (global) | ‖θ_T‖² − ‖θ₀‖² | Σ η_t² | Relative error | mean cos(update, θ) |
|---|---:|---:|---:|---:|---:|---:|
| 20260824 | 3.82 | 25.88 | 655.45 | 655.45 | 4.7×10⁻⁸ | −1.0×10⁻¹⁰ |
| 20260825 | 3.84 | 25.70 | 645.52 | 645.52 | 6.6×10⁻⁸ | −1.9×10⁻¹⁰ |
| 20260826 | 3.79 | 25.75 | 648.83 | 648.83 | 1.7×10⁻⁹ | −5.0×10⁻¹¹ |
| 20260827 | 3.80 | 25.77 | 649.77 | 649.77 | 3.0×10⁻⁸ | −3.3×10⁻¹¹ |
| 20260828 | 3.73 | 26.06 | 665.27 | 665.27 | 8.4×10⁻⁹ | +5.6×10⁻¹¹ |
| **Mean** | | | **652.97** | **652.97** | **3.1×10⁻⁸** (max 6.6×10⁻⁸) | **−6.5×10⁻¹¹** |

The orthogonal arm's norm growth is *exactly* the accumulated η² that Corollary 1
predicts. (The audit is on the *global* vector including the head, which the matched step
acts on; the backbone-only final norms average 25.1, matching Table I.)

### 7.5 Radial-coefficient sweep (E27, 5 seeds; γ = 0 is SGD, γ = 0.6 is the inward arm)

| γ | 0 | **0.2** | 0.4 | 0.6 | 0.8 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|
| Mean acc | 0.744 | **0.763** | 0.655 | 0.429 | 0.157 | 0.105 |
| Decay | 0.133 | 0.053 | 0.056 | 0.051 | 0.221 | 0.215 |
| Final ‖θ‖ | 20.4 | 4.1 | 3.1 | 2.3 | 0.25 | < 0.05 (exactly 0) |
| Dormant | 21.2 % | 6.2 % | 6.9 % | 7.5 % | 80.6 % | 100 % |

- The norm falls **monotonically** — direction *can* control the norm.
- Accuracy is **non-monotonic**: a *mild* inward bias (γ = 0.2) modestly beats SGD
  (0.763 vs 0.744), larger allocations degrade it, and γ ≥ 0.8 collapses learning
  (dormancy → 100 %).
- **No setting reaches magnitude control**: the best (0.763) trails Shrink & Perturb
  (0.799) and L2-init (0.819); at a matched low norm (≈ 4), Shrink & Perturb still wins.
- γ = 0 and γ = 0.6 reproduce the SGD and inward rows of Table I exactly (a validity check).

### 7.6 Magnitude methods — paired-seed comparisons (E20)

| Comparison | Mean paired difference in accuracy | Seeds won |
|---|---:|---:|
| L2-init − SGD | **+0.074** | 5 / 5 |
| Shrink & Perturb − SGD | **+0.055** | 5 / 5 |
| L2-init − Shrink & Perturb | +0.020 | 5 / 5 |
| ANR − Shrink & Perturb | −0.064 | 0 / 5 |
| ANR − SGD | −0.010 | 2 / 5 |

L2-init has the best accuracy; Shrink & Perturb has marginally lower decay (0.046 vs 0.047).
Neither dominates every metric.

### 7.7 ANR and its component ablation (E22-ANR, 5 seeds)

| Variant | Mean acc | Decay | Final ‖θ‖ | Dormant |
|---|---:|---:|---:|---:|
| SGD (reference) | 0.744 | 0.133 | 20.4 | 21.2 % |
| Shrink & Perturb (reference) | 0.799 | 0.046 | 4.2 | 0.6 % |
| ANR shrink-only (per-unit cap) | 0.690 | 0.150 | 14.6 | 36.3 % |
| **ANR revive-only** | **0.761** | **0.072** | 14.4 | 0.0 % |
| ANR full (cap + revive) | 0.735 | 0.090 | 7.9 | 0.6 % |

Revival is the useful element; the per-unit cap is harmful (below SGD, with *more*
dormancy). Every variant stays below the magnitude methods. Pre-committed verdict:
**`METHOD_NULL_PUBLISH_A`** (ANR is reported as an honest negative).

### 7.8 Regime dependence — five regimes, same seven-arm protocol (Table II)

| Regime | Seeds | Arms | SGD decay | Pearson r (‖θ‖, decay) |
|---|---:|---:|---:|---:|
| Permuted MNIST, MLP h = 16 | 5 | 7 | 0.133 | **+0.97** |
| Permuted MNIST, MLP h = 64 | 5 | 7 | 0.052 | +0.45 |
| Permuted MNIST, MLP h = 128 | 5 | 7 | 0.032 | −0.52 |
| Rotated MNIST, MLP h = 16 | 5 | 7 | 0.021 | −0.94 |
| Rotated MNIST, small CNN | 3 | 6 | 0.004 | −0.82 |

Per-regime detail (mean acc / decay / final ‖θ‖):

| Arm | h = 64 | h = 128 | Rotated MLP | Rotated CNN |
|---|---|---|---|---|
| SGD | 0.839 / 0.052 / 27.4 | 0.867 / 0.032 / 30.7 | 0.894 / 0.021 / 14.1 | 0.965 / 0.004 / 14.2 |
| L2-init | **0.856** / 0.023 / 8.9 | 0.865 / 0.024 / 11.0 | 0.857 / 0.036 / 6.3 | 0.941 / 0.016 / 8.8 |
| Shrink & Perturb | 0.825 / 0.041 / 5.1 | 0.827 / 0.036 / 6.2 | 0.803 / 0.058 / 3.9 | 0.886 / 0.033 / 5.7 |
| CBP | 0.836 / 0.058 / 27.3 | 0.866 / 0.031 / 30.6 | 0.890 / 0.024 / 13.8 | — |
| Norm projection (‖θ‖ ≤ 6) | — | — | — | 0.932 / 0.022 / 6.0 |
| Orthogonal | 0.828 / 0.067 / 28.5 | 0.863 / 0.036 / 29.8 | 0.878 / 0.022 / 17.2 | 0.961 / 0.005 / 12.1 |
| Inward (γ = 0.6) | 0.619 / 0.058 / 2.7 | 0.619 / 0.067 / 2.7 | **0.441** / 0.054 / 2.3 | **0.466** / 0.127 / 1.8 |
| ANR | 0.829 / 0.059 / 10.0 | 0.855 / 0.041 / 11.4 | 0.891 / 0.020 / 13.6 | — |
| Random direction | — | — | — | 0.113 / 0.079 / diverged (NaN) |

Two readings:
- The positive norm–decay coupling belongs to the **capacity-constrained stress regime**,
  not to continual learning in general.
- In **every** regime the inward arm still collapses accuracy (0.619 at width 64, 0.441 on
  rotated MNIST, 0.466 for the CNN) — the *geometric* result is regime-independent even
  though the *empirical correlation* is not.

### 7.9 Update geometry controls (E23 MLP, E25 CNN)

Matched-magnitude updates whose cosine to the SGD step is forced below 1:

| cos(update, SGD) | 1.0 | 0.9 | 0.75 | 0.5 | 0.25 | 0.1 | 0.0 (random) |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP mean acc | 0.738 | 0.533 | 0.434 | 0.377 | 0.286 | 0.206 | 0.100 |
| MLP final ‖θ‖ | 20.5 | 81.7 | 125.5 | 176.9 | 253.3 | 724.3 | diverged |
| CNN mean acc | 0.965 | — | 0.952 | 0.927 | 0.754 | 0.426 | 0.113 |
| CNN final ‖θ‖ | 14.2 | — | 68.3 | 122.3 | 263.9 | 534.9 | diverged |

Off-gradient fixed-length steps make the norm **explode** and accuracy collapse — what the
identity predicts when steps have no inward component. The mean radial alignment of plain
SGD steps is only **+0.0097**, so SGD's own steps are almost orthogonal to θ. That is why
SGD's norm grows steadily: it accumulates η² terms (Cor. 1) step after step.
A direct norm projection on the MLP (‖θ‖ ≤ 6) gives 0.774 accuracy, decay 0.086 — better
than SGD but below L2-init and Shrink & Perturb.

### 7.10 Layer-wise norms and alternative metrics (E26, 3 seeds)

- **Which layer grows?** SGD hidden-1 (784→16): 2.30 → **18.60**; hidden-2: 2.31 → 7.36;
  head: 1.82 → 4.91. The first layer carries most of the growth.
- **Do conclusions survive other plasticity metrics?** Yes — the ranking is the same:

| Arm | Peak-to-tail decay | Last-task acc | Late-10 % acc | Slope per task |
|---|---:|---:|---:|---:|
| SGD | 0.151 | 0.689 | 0.688 | −0.00090 |
| L2-init | 0.040 | 0.840 | 0.827 | +0.00022 |
| Shrink & Perturb | 0.046 | 0.833 | 0.804 | +0.00018 |
| CBP | 0.133 | 0.751 | 0.708 | −0.00069 |
| Orthogonal | 0.158 | 0.670 | 0.657 | −0.00094 |
| Inward | 0.050 | 0.428 | 0.436 | +0.00005 |
| ANR | 0.097 | 0.760 | 0.747 | −0.00017 |

### 7.11 Hypotheses — final verdicts (tested regimes only)

| Hypothesis | Verdict |
|---|---|
| **H1** — weight-norm growth predicts plasticity decay | **Supported** in the stress regime (r = +0.97, λ-sweep) |
| **H2** — fixed-magnitude directional steering preserves plasticity | **Not supported** as a competitive mechanism (mild bias +0.019 over SGD; no setting reaches magnitude control; strong steering collapses) |
| **H3** — direct magnitude control preserves plasticity | **Supported** (L2-init +0.074, S&P +0.055 over SGD, 5/5 seeds) |
| **H4** — the norm–plasticity relationship generalizes across width and task | **Not supported** as a universal claim (weakens with width; absent where plasticity loss is mild) |

### 7.12 Phase-1 results (e01–e19), summarized

| Experiment | Question | Result | Verdict |
|---|---|---|---|
| E02 | Are the regressors identifiable? | max VIF 4.81 < 5.0 | pass |
| E03b / E03c | β₄ on synthetic data (mild / hard) | −0.003, CI [−0.007, +0.001] / +0.006, CI [−0.003, +0.015] | CI includes 0 → clean negative |
| E04 / E04b | β₄ on Split-MNIST | +0.0023, CI [+0.0019, +0.0027], p < 10⁻²⁶; bridging lowered d_dir VIF 6.5 → 3.2 | scalar positive |
| E06 | Is the direction stable across seeds? | β significant on 5/5 seeds, but the "top harmful direction" has pairwise cosine ≈ 0 across seeds; only past-gradient alignment survives matching | axis negative |
| E05, E06-N3, E08 | Can projecting out the bad direction help? | no gain over SGD at matched ‖Δθ‖; the past-gradient axis controls forgetting (BWT), not plasticity | method negative |
| E09 | Does it scale to CIFAR? | CIFAR-10 β ≈ +0.0025 with d_dir VIF 4.80; CIFAR-100 also positive | scalar replicates |
| E11, E14 | Is any of 8 geometric axes (Fisher, Hessian-Lanczos, K-FAC, Jacobian-SVD, …) seed-stable? | none (e.g. Hessian-Lanczos mean abs pairwise cosine 0.15; β signs flip across seeds) | negative |
| E15 | Is the "β ≈ 0.002 everywhere" invariance real? | no — an FTD-scale artifact; normalized MNIST β is 4–6× CIFAR β | corrects an earlier claim |
| E12 | Predictive theory: does cos(g_train, g_probe) predict damage? | held-out r = 0.977 (MNIST, p = 4.3×10⁻³), 0.997 (CIFAR-10, p = 2.2×10⁻⁴) — but only 5 held-out probes (MNIST bootstrap CI for r spans [−1, 1]) | pass, small n |
| E16 | Power-adequate re-run (29 probes, 9 held out) | MNIST r = 0.942 (p = 1.5×10⁻⁴), sign 9/9, slope CI [+0.044, +0.072]; CIFAR-10 r = 0.666 (p = 0.050), sign 9/9, slope CI [+0.078, +0.537] | pass (the authoritative version) |
| E17 | Adaptive directional controller (the original motive) | lowers probe FTD slightly (2.322 vs 2.335) but replay dominates (2.290, acc 0.320); SGD acc only 0.124 (degenerate class-incremental regime) | null |
| E19 | Fixed task-incremental regime + two-sided controller | SGD 0.910 (MNIST) / 0.572 (CIFAR-10); best controller changes FTD by ≤ 0.0002 | null → regime too easy → pivot to Phase 2 |

---

## 8. Why each metric is used

| Metric | Why this metric | What a simpler metric would miss |
|---|---|---|
| **Mean accuracy over 100 tasks** | overall learning ability across the whole stream; one number that rewards being good *throughout* | final accuracy alone ignores how the run got there |
| **Plasticity decay** = max_t a_t − mean(a_t over the last 10 % of tasks) | directly measures *erosion* of the ability to fit new tasks: best-ever learning vs late-stream learning; averaging the tail makes it less noisy than the last task alone | mean accuracy can hide a steady decline; the last task alone is noisy. E26 confirms last-task, late-10 % and slope metrics give the same ranking |
| **Final global weight norm ‖θ‖** | the mechanism variable; the identity is stated for the global Euclidean norm | per-layer norms are a different object (reported separately in E26) |
| **Dormant-unit fraction** | dead ReLU units cannot learn, a known plasticity failure (Sokar et al.) | accuracy alone doesn't say *why* learning slowed |
| **Effective (stable) rank** ‖F‖²_F / ‖F‖²₂ | detects representation collapse; stable rank is smooth and cheap (one SVD) | exact rank is too sensitive to tiny singular values |
| **Matched-step deviation** | proves the directional arms really kept the SGD step length; shown only for arms bound by the constraint (N/A, not 0, for the rest) | without it, a "directional" effect could secretly be a magnitude effect |
| **Pearson r across arms** (+ Spearman, per-seed, leave-one-out) | quantifies the cross-arm association; the robustness checks show it isn't one outlier or one seed | a single r with n = 7 could be fragile, hence the extra checks |
| **Mean ± std over 5 seeds** | shows run-to-run variability | a single seed can mislead |
| **Paired-seed differences + win counts** | each seed shares data order across arms, so pairing removes shared noise; with n = 5, win counts are more honest than p-values | unpaired means waste the shared-seed design; significance tests at n = 5 are unreliable |
| **Relative error of the Cor. 1 identity** | scale-free check that the theory matches the code (3×10⁻⁸) | an absolute difference depends on the norm's scale |
| *(Phase 1)* FTD, CTB, PTD, β₄ via OLS-HC3, VIF, caliper matching, bootstrap | separate future damage from magnitude, current benefit and past damage simultaneously; HC3 handles unequal variance; VIF < 5 ensures the coefficients are identifiable | a raw correlation would confound direction with size and with forgetting |

---

## 9. Negative results — why they happen and whether they are beneficial

| Negative result | Why it happens | Beneficial? |
|---|---|---|
| **Negative correlations r = −0.52 (width 128), −0.94 (rotated MLP), −0.82 (CNN)** | In these regimes plain SGD barely loses plasticity (SGD decay ≤ 0.032, below the width-16 seed std of 0.036). With no real decay to explain, the arms that *reduce* the norm pay their own costs. Shrink & Perturb shrinks every task (decay 0.058 on rotated MNIST vs SGD 0.021), and inward steering collapses. Meanwhile the high-norm arms (SGD, CBP, orthogonal) are the best performers. So across arms, *low norm happens to go with higher decay*. This is the association **vanishing**, not "norm growth is good". | **Yes, scientifically.** It establishes the boundary condition: norm control helps *only* when there is norm-driven plasticity loss. Practically it warns against shrinking weights blindly (Shrink & Perturb drops from 0.894 to 0.803 on rotated MNIST). |
| **H2 not supported — directional control loses to magnitude control** | The identity: a fixed-length step can only lower the norm by spending part of the step pointing inward, which is taken from task descent (Cor. 2). Orthogonal steps must raise the squared norm by η² (Cor. 1). | **Yes.** It closes a plausible but dead-end method direction, with a reason, not just an observation. |
| **ANR (our method) did not beat global magnitude control** (−0.064 vs Shrink & Perturb, 0/5 seeds) | The ablation shows the **per-unit norm cap is harmful** (shrink-only 0.690, *more* dormancy, 36 %); the dormant-unit revival is helpful (revive-only 0.761, decay 0.072 even though its norm stays at 14.4). Combining them lets the harmful part drag the result down. | **Partly.** No new SOTA method, but it shows *which* component matters, and that dormancy is a second lever separate from norm (revive-only lowers decay without lowering the norm). |
| **Continual Backprop scored below SGD** (0.728 vs 0.744) | This is a **simplified reimplementation**: no maturity threshold and an untuned reset rate ρ = 10⁻². The paper states it should not be read as representative of tuned CBP. | **No, as a CBP result** — it is a caveat. It does not affect the main conclusions, which rest on L2-init and Shrink & Perturb. |
| **Effective rank vs decay r = −0.21** | Rank tracks accuracy (r = +0.89) but barely tracks decay, so representation collapse is not what drives decay here. | **Yes.** It prevents a wrong attribution (decay ≠ rank collapse in this regime). |
| **Random-direction arm diverged (NaN norm, acc 0.113 / 0.100)** | A fixed-length step with no relation to the gradient adds η² to the squared norm on average *and* provides no descent. With η₀ = 0.1–0.2 over 6 000 steps the weights blow up. | **Yes, as a control.** It proves that "any direction" is not enough; it is excluded from the correlation (6 CNN arms) rather than silently set to 0. |
| **Strong inward steering collapses learning** (γ ≥ 0.8: accuracy 0.16 / 0.11, dormancy → 100 %) | Almost the whole step is spent shrinking the weights to zero, so all units die and nothing is learned. | **Yes.** It is the empirical end-point of Cor. 2's trade-off. |
| **Negative slope per task for SGD (−0.0009/task)** | That *is* the disease: accuracy on each new task declines over the stream. | Expected; confirms the regime is stressed. |
| **Phase 1: no controllable directional axis; directional controllers null (E17, E19)** | The directional effect is real as a scalar but its geometric direction changes from seed to seed. The test-beds were either collapsed (class-incremental) or too easy (task-incremental). | **Yes.** Falsifying a pre-registered hypothesis cleanly is a contribution. It also triggered the pivot that produced the paper. |
| **Phase 1: E10 sign flip** (novel-class probe gave negative β) | A probe-construction bug (an inconsistent head label mapping) — retracted in decision D-034. | **Yes, procedurally** — caught and logged openly instead of being published. |

**Overall:** the negatives are *beneficial for knowledge* (they define where magnitude
control matters, rule out a method direction, and localize ANR's failure). They are
*not beneficial as a product* — the project does not deliver a new method that beats
L2-init.

---

## 10. Improvements that can be made

**Scale and generality**
1. Test larger and deeper networks (ResNet-18, small ViT) and harder streams
   (Split/continual CIFAR-100, Tiny-ImageNet, continual RL).
2. Add **normalization layers** (BatchNorm / LayerNorm) — they rescale activations
   independently of ‖θ‖ and change the geometry.
3. Add **adaptive optimizers** (Adam / AdamW) and extend the identity to per-coordinate
   steps (a preconditioned norm instead of the Euclidean one).
4. Extend the identity to **per-layer / per-unit** norms — E26 shows the first layer
   carries most of the growth.

**Stronger baselines and methods**
5. A **tuned Continual Backprop** (maturity threshold, ρ sweep) for a fair comparison.
6. **Hybrid methods:** combine the mild inward bias (γ = 0.2) or dormant-unit revival with
   L2-init, since revival lowered decay without lowering the norm.
7. Tune ANR's target τ and drop the per-unit cap (the harmful part), keeping revival.

**Statistics and evidence**
8. More seeds for the CNN (3 → 5) and more arms per regime (the correlations use n = 6–7
   arm means).
9. Persist **per-task trajectories** (`task_test_acc`, per-task ‖θ‖ and dormancy) in the
   JSON so curve figures and the web app can show real curves without a GPU re-run.
10. Add plasticity measures beyond per-task accuracy (e.g. trainability of a fresh probe
    task from the current weights, loss sharpness / Hessian spectrum).

**Engineering and reproducibility**
11. Add the Cor. 1 audit to `experiments/` as a numbered experiment (`run_all.py` already
    covers e01–e27; the audit artifact exists but its one-off script is not included).
12. Make the norm definition consistent everywhere (MLP diagnostics use backbone-only,
    the CNN diagnostics and the matched step use all parameters).
13. Speed: keep task data on the GPU, update parameters in place instead of flatten /
    unflatten, and batch seeds with `torch.func.vmap`.
14. Benchmark compute overhead per arm (currently not measured).

---

## 11. Scope and limitations

**Scope (what is covered)**
- Loss of plasticity (future trainability), not catastrophic forgetting.
- 100-task permuted MNIST (primary) and rotated MNIST; ReLU MLPs of width 16 / 64 / 128 and
  one small CNN; plain SGD; warm-started weights; a shared 10-way head.
- The **global** Euclidean parameter norm; directional interventions at **exactly matched**
  step size.
- Claims are framed as "supported / not supported in the tested regime", never
  "proved / disproved".

**Limitations**
- **Dataset scale:** MNIST family only — no CIFAR / ImageNet scale in the final study.
- **Models:** small two-hidden-layer MLPs plus one small CNN; deeper models are untested.
- **No normalization layers and no adaptive optimizers** — both break the fixed-global-step
  premise of the identity.
- **Statistics:** the norm–decay and rank–accuracy relationships are **arm-level
  associations** (n = 7), not seed-level causal estimates; the λ-sweep adds only one
  within-family intervention.
- **Directional coverage:** a radial sweep plus an orthogonal endpoint, random and
  angle controls — not an exhaustive search (no learned or subspace directions).
- **Theory scope:** the identity concerns the global norm only; per-layer / per-unit
  geometry and non-norm pathologies are out of scope.
- **Baselines:** CBP is a simplified reimplementation (no maturity threshold, untuned ρ);
  ANR was not tuned beyond its a-priori target.
- **Measurement:** plasticity is measured only by per-task accuracy; per-task curves are not
  persisted; compute overhead was not benchmarked.
- **CNN:** 3 seeds; the random-direction arm diverged and is excluded from its correlation.
- **Phase 1:** the E12 predictive-theory correlations (0.977 / 0.997) used only 5 held-out
  probes; the power-adequate E16 values (0.942 MNIST, 0.666 CIFAR-10) are the reliable ones.

---

## 12. Backbone files and what each one is for

### Core library

**`src/plasticity_stress.py`** — the heart of Phase 2.
Defines `PlasticMLP` and `PlasticCNN`, the `PermutedMNIST` and `RotatedMNIST` task streams,
every update rule (orthogonal, inward, random, angle, norm projection, CBP, ANR,
Shrink & Perturb, L2-init) and the single training loop (`run_arm`, `run_arm_cnn`). It also
computes all diagnostics (norm, dormancy, effective rank, alignment, matched-step deviation)
and now returns the Cor. 1 audit fields. Every paper number flows through this file.

**`src/datasets.py`**
Reads MNIST straight from the raw IDX files in `Deep Learning Dataset1/` (`load_mnist`) and
loads CIFAR-10/100 from PNGs with an NPZ cache (Phase 1). No network download is needed,
so the data path is fully reproducible offline.

**`src/device.py`**
Auto-detects CUDA and falls back to CPU (`pick_device`), and produces the hardware summary
(`env_summary`: torch version, GPU name, free memory) that is written into every result JSON.
This is how each number is traced back to the machine that produced it.

**`configs/preregistration.py`**
The frozen constants of the project — seeds, thresholds, kill conditions — locked before the
Phase-1 experiments ran, with labelled addenda for later rounds. It enforces the
"decide the rule before seeing the data" discipline.

**`src/update_generator.py`, `src/metrics.py`, `src/stats.py`** (Phase 1)
Generate many updates with the same ‖Δθ‖ from different sources, measure FTD / CTB / PTD /
D_dir on probe tasks, and run the β₄ regression with HC3 errors, VIF gates, caliper matching
and bootstrap. Together they implement the Phase-1 "is there a directional residual?" test.

**`src/cl_harness.py`** (Phase 1, v1.4)
The task-incremental multi-head harness (`MultiHeadMLP` / `MultiHeadCNN`) that fixed the
collapsed class-incremental regime. It showed the old regime was too easy, which led to the
Phase-2 pivot.

### Experiments (the paper)

**`experiments/e20_plasticity_stress.py`** — the main experiment (Table I, Fig. 2).
Runs the seven arms × five seeds on width-16 permuted MNIST, writes
`results/tables/e20_summary.json`, the per-task accuracy figure and the mechanism scatter, and
applies the pre-committed ANR decision rule (`METHOD_NULL_PUBLISH_A`).

**`experiments/e22_reviewer_experiments.py`**
Three reviewer-driven studies: the width-64 seven-arm replication, the five-seed L2-init λ
sweep (the interventional evidence), and the ANR component ablation (shrink-only vs
revive-only vs full).

**`experiments/e23_update_geometry.py`**
Answers "only two directions were tested": adds a matched-step random-direction control, an
angle sweep (cosine to SGD from 1.0 to 0.0), a direct norm-projection arm, and per-step
gradient / radial alignment logging on the width-16 MLP.

**`experiments/e24_generalization.py`**
The boundary-condition experiments: the seven arms on rotated MNIST (a different task
geometry) and on a width-128 MLP. These produced the negative correlations (−0.94, −0.52).

**`experiments/e25_cnn.py`**
The second architecture: a small CNN on rotated MNIST (3 seeds) with norm projection replacing
the MLP-specific CBP and ANR, plus a random-direction arm and a CNN angle sweep.

**`experiments/e26_diagnostics.py`**
Layer-wise norm breakdown (which layer drives growth) and alternative plasticity metrics
(last-task accuracy, late-10 % accuracy, slope) to check the conclusions don't depend on
the decay definition.

**`experiments/e27_radial_sweep.py`** — Fig. 3.
Sweeps the inward coefficient γ ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0} at matched step under the
identical e20 protocol, reproducing SGD (γ = 0) and the inward arm (γ = 0.6) exactly, and writes
the radial-sweep figure used in the paper.

**`experiments/e16_predictive_theory.py`** (Phase 1 headline)
The power-adequate test of the Phase-1 predictive law (29 head-consistent probes, 9 held out):
damage sign and size follow cos(g_train, g_probe). It is the authoritative Phase-1 positive result.

**`run_all.py`**
Single-entry reproduction of the pipeline in dependency order, one subprocess per stage, with
`--only` / `--skip` flags. It covers e01–e27 and finally rebuilds the web dataset.

### Results

**`results/tables/e20_summary.json`** (and its siblings `e22_*`, `e24_*`, `e25_*`, `e27_*`)
The ground truth for every table, figure and web-app value: per-seed raw values, mean/std
aggregates and the hardware profile. Nothing downstream is computed from anything else.

**`results/tables/cor1_orthogonal_audit.json`**
The numerical check of Corollary 1 on the orthogonal arm (5 seeds): endpoint change in the
squared norm, Ση_t², relative error (3×10⁻⁸ mean) and mean radial cosine (−6.5×10⁻¹¹). It
backs the confirming sentence added to the paper.

### Paper

**`paper/main(iconcise).tex`** — the paper source.
The anonymized IEEE ICA 2026 camera-ready with every reviewer fix applied. Fig. 1 (the
geometry diagram) is drawn in TikZ inside this file. The compiled paper with the author block
is `Research Paper.pdf` (repository root); the web app serves the same PDF.

**`paper/references.bib`**
The 37 bibliography entries for that paper (fixed years and venues, brace-protected accents,
Lyle et al. 2024 added).

**`paper/figs/plasticity_curves.png`, `paper/figs/radial_sweep.png`**
Fig. 2 (per-task accuracy over the 100 tasks, from e20) and Fig. 3 (the radial sweep, from
e27), at 300 dpi with publication labels.

**`paper/phase1_manuscript.md`**
The complete Phase-1 manuscript (the directional future-trainability study, v1.3), kept for
the record; it is not the published paper.

### Plasticity Lab (web)

**`web/scripts/build_data.py`**
The data layer: reads the real `results/tables/*.json`, converts NaN to `null` (never 0), marks
Match-dev as applicable only for matched-step arms, canonicalizes arm names, and writes one
normalized file. It enforces the "never invent a value" rule at the source.

**`web/public/data/plasticity.json`**
The single dataset the whole UI reads. Rebuild it with `npm --prefix web run data` whenever the
result tables change.

**`web/src/data/DataContext.tsx`**
Fetches `plasticity.json` once and provides it to every page through React context, with
explicit loading and error states (the source path and failure reason are shown, so no page
silently renders empty data).

**`web/src/App.tsx`** (+ `main.tsx`, `lib/nav.ts`)
Defines the 17 routes (Overview, R20, sweeps, regimes, analysis, reproducibility, paper) under
a HashRouter so deep links and refreshes work, and the navigation structure used by the top bar.

**`web/src/components/ResultTable.tsx`, `charts.tsx`, `GeometryDiagram.tsx`, `SweepSlider.tsx`, `RegimeView.tsx`**
The reusable building blocks: the sortable comparison table with row selection and N/A
handling, the Recharts bar / line / scatter wrappers, the SVG drawing of the identity's
geometry, the slider that walks the λ and γ sweeps, and the regime-segmented view.

**`web/src/pages/*.tsx`**
One file per analysis view. For example `R20.tsx` (main table with Setup / Results / Runs /
Config / Notes tabs), `RadialSweep.tsx`, `NormPlasticity.tsx` (the association with its
"descriptive, not causal" framing), and `WidthStudy.tsx` / `RotatedMNIST.tsx` / `CNNProbe.tsx`
(the regimes).

### Logs

**`decision.md`**
The append-only decision log (D-000 … D-047). Every design choice, pivot (D-044), bug fix
(D-045 L2-init penalty) and pre-committed decision rule (D-047) is recorded with its reason
and rejected alternatives.

**`result.md`**
The append-only experiment log (R01 … R20). Each run is recorded with its command,
environment, numbers, kill-condition status and interpretation, including the negative results.

---

## 13. How to reproduce

```bash
# 1. Environment
pip install -r requirements.txt
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128   # GPU (optional)

# 2. Paper experiments (each writes results/tables/*.json and results/figures/*.png)
python experiments/e20_plasticity_stress.py      # Table I, Fig. 2
python experiments/e22_reviewer_experiments.py   # width 64, λ-sweep, ANR ablation
python experiments/e23_update_geometry.py        # random / angle / norm-projection controls
python experiments/e24_generalization.py         # rotated MNIST, width 128
python experiments/e25_cnn.py                    # small CNN on rotated MNIST
python experiments/e26_diagnostics.py            # layer-wise norms, alternative metrics
python experiments/e27_radial_sweep.py           # Fig. 3

# 3. Web dashboard
npm --prefix web run data    # rebuild web/public/data/plasticity.json from results/tables
npm --prefix web run dev     # open the printed localhost URL

# 4. Paper
#    Zip paper/main(iconcise).tex, paper/references.bib and paper/figs/,
#    upload the zip to Overleaf → pdfLaTeX → Recompile
```
