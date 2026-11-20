# Continual Learning Experimental Studio — Guide

The Studio is a registry, execution engine, API and web interface around the **existing**
research code. It has no model or training loop of its own: every run calls
`run_arm` / `run_arm_cnn` in `src/plasticity_stress.py`. This guide covers setup, what a run
does, the exact experimental protocol it reproduces, the API, and what the Studio cannot do.

---

## 1. Start it

Install the Python dependencies once (`pip install -r requirements.txt`; for GPU, install the
CUDA build of PyTorch first) and the web dependencies (`npm --prefix web install`).

**Backend** (from the project root; binds to localhost only):

```bash
python -m uvicorn studio.api.server:app --host 127.0.0.1 --port 8765
```

**Frontend** (Vite dev server; proxies `/api/*` and WebSockets to port 8765):

```bash
npm --prefix web run dev
```

Open the printed URL (e.g. `http://localhost:5173`). The artifact-based pages work without the
backend; configuring, running, monitoring and comparing experiments need it.

**Tests** (about 40 s on a GPU):

```bash
python -m pytest studio/tests -q
```

## 2. Run a small experiment

UI: **Studio → Experiment Studio** → choose a preset (e.g. *Smoke test*) or configure
dataset / architecture / update rule → check the preview → **Run**. You are taken to the
**Live Monitor**; results then appear in the **Results Explorer**.

API (the same thing from a shell):

```bash
curl -s -X POST http://127.0.0.1:8765/experiments/run -H "Content-Type: application/json" -d "{\"config\": {\"dataset\": {\"name\": \"permuted_mnist\", \"n_tasks\": 5}}, \"metadata\": {\"name\": \"smoke\"}}"
```

## 3. What happens when you press Run

1. The configuration is normalized against the schema (`studio/engine/config_loader.py`);
   unsupported parameters are rejected, defaults are the paper values.
2. It is validated (`experiment_validator.py`): dataset/architecture/method compatibility
   taken from the code (CNN only on rotated MNIST; CBP/ANR MLP-only), ranges, and warnings
   for every departure from the paper protocol.
3. A compute estimate is computed from *measured* timings only (pooled over completed Studio
   runs on the same device and architecture, or the Cor. 1 audit reference measurement).
   Submissions above 600 s (or above 3 000 steps without a measurement) need explicit
   confirmation.
4. One run per seed is created under `studio/experiments/<run_id>/`, each with a new ID and a
   **read-only** `config.json` whose SHA-256 is recorded in `meta.json`.
5. The queue runs one experiment at a time on the GPU, each in its own process
   (`python -m studio.engine.experiment_runner <run_id>`). The runner re-verifies the hash,
   records the environment (torch/CUDA/cuDNN versions, GPU, SHA-256 of the research code and of
   the MNIST files `load_mnist` actually reads), then calls `run_arm` / `run_arm_cnn`.
6. A read-only `progress_cb` hook streams per-step position and per-task metrics to
   `progress.json` and `events.jsonl`; the UI receives them over a WebSocket.
7. On completion, `result.json` stores the summary (paper metric definitions) and the per-task
   arrays. If the configuration equals a per-seed legacy record (e20 / e27), the result is
   compared with it and labelled `reproduced_exact`, `reproduced_within_tolerance` or
   `mismatch`; otherwise `no_legacy_counterpart`.

## 4. Reproducibility guarantees (verified)

| Property | How it is guaranteed | Evidence |
|---|---|---|
| The hook does not change training | read-only callback; `progress_cb=None` is the original code path | `test_hook_does_not_change_mlp_results` (7 methods, bit-identical), `test_hook_does_not_change_cnn_training` |
| Same configuration ⇒ same result | seeds control init, task streams, batch order, noise and random directions | `test_seed_reproducibility_across_runs` (identical per-task arrays) |
| The Studio reproduces the paper pipeline | run with a legacy per-seed configuration and compare | **`legacy.e20.sgd` seed 20260824 and `legacy.e20.dir_control_c1.0` seed 20260824 reproduced with every metric difference exactly 0** (runs `run-20261120-014830-8d9a02`, `run-20261120-014923-154ba3`) |
| Snapshots cannot drift silently | read-only file + SHA-256 check before execution | `test_tampered_snapshot_is_detected` |
| Existing artifacts are never modified | the registry only reads `results/tables/` | `test_registry_never_writes_legacy_tables` |
| Paper values match the artifacts | 68 values recomputed and compared at printed precision; each paper value carries its exact LaTeX source | `test_all_paper_values_are_consistent_with_artifacts`, `test_every_paper_reference_value_is_quoted_from_the_tex` |
| Cor. 1 holds in Studio runs | every MLP run stores ‖θ₀‖², ‖θ_T‖², Σ‖Δθ'‖² | orthogonal reproduction: relative error 4.70×10⁻⁸ — identical to the Cor. 1 audit artifact for that seed |

**CUDA caveat:** MLP runs are bit-reproducible. CNN training (weights, accuracy, dormancy) is
reproducible, but the CNN effective-rank diagnostic jitters by ~10⁻⁵ between runs because the
cuDNN forward pass on the probe batch is not bitwise deterministic.

---

## 5. Protocol reference (exact, from the code)

### 5.1 Datasets and task streams

**MNIST** — `src/datasets.load_mnist("Deep Learning Dataset1")` reads the four IDX files
(60 000 train / 10 000 test, uint8 28×28). Both streams scale pixels to `[0, 1]`.

**`PermutedMNIST(…, n_tasks, seed, train_subset)`** — `rng = np.random.default_rng(seed)`;
`perms[t] = rng.permutation(784)` for every task (drawn first, in task order), then
`train_idx[t] = rng.choice(60000, train_subset, replace=False)`. Task t trains on
`train_idx[t]` under `perms[t]` and is tested on the **full 10 000-image test set** under the
same permutation. Permutations are applied on request.

**`RotatedMNIST(…, n_tasks, seed, train_subset=6000, test_subset=2000)`** — angles
`rng.uniform(0, 180, n_tasks)` degrees; one shared 2 000-image test subset; rotation via
`F.affine_grid` + `F.grid_sample` (zero padding).

Task order is fixed by the seed; there is no shuffling of tasks.

### 5.2 Seeds

| RNG use | Seed |
|---|---|
| Model initialization | `torch.manual_seed(seed)` at the start of `run_arm` / `run_arm_cnn` |
| Task permutations / angles / subsets | `np.random.default_rng(seed)` inside the stream |
| Mini-batch order for task t | `torch.Generator().manual_seed(seed + 17·t)` |
| Shrink & Perturb noise at boundary t | `torch.Generator().manual_seed(seed + 71·t)` |
| Random-direction / angle rule at within-task step s | `torch.Generator('cpu').manual_seed(seed·100003 + s)` (s resets every task) |
| CBP reset decision | `Generator().manual_seed((seed + 131·s) + cbp.step)` |
| ANR revival / CBP reset weights | `nn.init.kaiming_uniform_` on the global torch RNG |

Paper seeds: **20260824 – 20260828** (5 seeds); e25 (CNN) and e26 use the first 3. Re-running
e20 reproduces `e20_summary.json` bit-identically on an RTX 4070.

### 5.3 Architectures and training

| Model | Definition | Parameters |
|---|---|---|
| `PlasticMLP(784, hidden, depth=2, 10)` | `depth` × (Linear → ReLU), shared Linear head | h=16: **13 002** (backbone 12 832); h=64: 55 050; h=128: 118 282 |
| `PlasticCNN(base=16, fc=64)` | Conv3×3(1→16)+ReLU+MaxPool2 → Conv3×3(16→32)+ReLU+MaxPool2 → FC(1568→64)+ReLU → Linear(64→10) | **105 866** |

Initialization: PyTorch defaults. The update is applied manually, `θ ← θ_before + Δθ'`, on the
flattened vector of **all** parameters (head included); no momentum, no weight decay. One
initialization before task 1; weights carry across every task boundary.

| Setting | e20 (main) | e22 width-64 | e24 width-128 | e24 rotated MLP | e25 CNN | e27 radial |
|---|---|---|---|---|---|---|
| Dataset | permuted | permuted | permuted | rotated | rotated | permuted |
| Width / arch | MLP 16 | MLP 64 | MLP 128 | MLP 16 | CNN 16/32/fc64 | MLP 16 |
| η₀ (lr) | 0.2 | 0.2 | 0.2 | 0.2 | **0.1** | 0.2 |
| Batch / steps per task / tasks | 128 / 60 / 100 | same | same | same | same | same |
| Train subset per task | 6 000 | 6 000 | 6 000 | 6 000 | 6 000 | 6 000 |
| Test set | full 10 000 | full | full | 2 000 subset | 2 000 subset | full |
| Seeds | 5 | 5 | 5 | 5 | **3** | 5 |

Diagnostics use the first 1 000 test images of each task (task-transformed).
**`run_arm`'s signature defaults are not the paper values** (`hidden=128`, `shrink=0.6`,
`cbp_reset_rate=1e-4`, `anr_target_norm=2.0`); the experiment scripts and the Studio always
pass every parameter explicitly.

### 5.4 Update rules

`Δθ = −η₀ ∇θ L` (flattened), `η = ‖Δθ‖`, `θ̂ = θ/‖θ‖` (global vector).

| Method key | Implementation | Matched step |
|---|---|:---:|
| `sgd` | `Δθ' = Δθ` | reference |
| `cbp` (MLP only) | utility `u ← 0.99u + 0.01·mean abs(activation)` per hidden unit; after every step, with probability ρ re-initialize the lowest-utility unit of each hidden layer (no maturity threshold). Paper ρ = 10⁻². | no |
| `l2_init` | loss += `λ Σ‖p − p₀‖²` on the live parameters. Paper λ = 10⁻². | no |
| `shrink_pert` | at each task boundary `θ ← s·θ + σ·ε`. Paper s = 0.7, σ = 0.01. | no |
| `dir_control` (orthogonal) | `d = Δθ − c(Δθ·θ̂)θ̂`; `Δθ' = d·η/‖d‖`. Paper c = 1.0. | **yes** |
| `dir_inward` (inward) | `d = Δθ − γηθ̂`; `Δθ' = d·η/‖d‖`. Paper γ = 0.6; e27 sweeps γ ∈ {0, 0.2, …, 1.0}. | **yes** |
| `adaptive_norm_revive` (MLP only) | at each task boundary, per hidden layer: cap each unit's incoming norm at τ; re-initialize dormant units (mean abs activation < 10⁻³) and zero their outgoing weights. Paper τ = 3.0. | no |
| `norm_project` | after the SGD step, if `‖θ‖ > 6.0`, rescale θ to 6.0 (e23, e25) | no |
| `random_dir` | `Δθ' = r·η/‖r‖`, `r ~ N(0, I)` (seeded) | **yes** |
| `angle` | `Δθ' = (a ĝ + √(1−a²) u)·η`, `u ⊥ ĝ` random unit, `a = cos_target` | **yes** |

The CNN runner supports every rule except `cbp` and `adaptive_norm_revive` (per-unit MLP rules).

### 5.5 Measurements

| Quantity | Definition in code |
|---|---|
| Per-task accuracy `a_t` | test accuracy on task t immediately after training on it |
| **Plasticity decay (paper)** | `max_t a_t − mean(a_t over the last ⌈T/10⌉ tasks)`, computed in the experiment scripts |
| `plasticity_decay` returned by `run_arm` | `mean(first T/5) − mean(last T/5)` — a different metric, stored by the Studio as `decay_first_last_20pct` and never used for paper numbers |
| Weight norm (MLP tables) | ‖backbone‖ = hidden-layer weights and biases, head excluded |
| Weight norm (CNN) | ‖all parameters‖ |
| Dormant fraction | hidden units with mean abs activation < 10⁻³ on the probe batch (MLP: all hidden layers; CNN: FC layer) |
| Effective rank | stable rank ‖F‖²_F / ‖F‖²₂ of mean-centred penultimate features |
| Matched-step deviation | `max_t abs(‖Δθ'_t‖ − η_t) / η_t` |
| Update–gradient / radial alignment | per-step `cos(Δθ', Δθ)` and `cos(Δθ', θ_before)`, returned as means |
| Radial / tangential components | derived per step from the logged cosine and step norm (not extra measurements) |
| Cor. 1 audit fields | `theta0_sq`, `thetaT_sq`, `sum_applied_sq` on the global vector |

### 5.6 Paper artifacts in `results/tables/`

| ID | Script | Artifact | Paper element |
|---|---|---|---|
| e20 | `e20_plasticity_stress.py` | `e20_summary.json` (per-seed raw) | Table I, Fig. 2, r = 0.97 |
| e21 | `e21_norm_strength_sweep.py` | `e21_summary.json` | superseded 3-seed λ pilot (not in the paper) |
| e22 | `e22_reviewer_experiments.py` | `e22_arch64_summary.json`, `e22_lambda_summary.json`, `e22_anr_ablation_summary.json` | Table II (width 64), λ sweep, ANR ablation |
| e23 | `e23_update_geometry.py` | `e23_summary.json` | supplementary geometry controls |
| e24 | `e24_generalization.py` | `e24_rotmnist_summary.json`, `e24_arch128_summary.json` | Table II |
| e25 | `e25_cnn.py` | `e25_cnn_summary.json` | Table II (CNN) |
| e26 | `e26_diagnostics.py` | `e26_summary.json` | supplementary diagnostics |
| e27 | `e27_radial_sweep.py` | `e27_radial_sweep_summary.json` (per-seed raw) | Fig. 3 |
| cor1 | one-off audit (script not included) | `cor1_orthogonal_audit.json` | Cor. 1 sentence |

Per-seed raw values exist only for e20, e27 and cor1; the others store mean/std. No artifact
stores per-task trajectories. Phase-1 artifacts (e01–e19) use a different protocol (β₄
regression) and are listed by the Studio for completeness only.

### 5.7 Compute

Everything in the paper fits one **RTX 4070 Laptop GPU (8 GB)**: a width-16, 100-task, 60-step
run takes 27–32 s and well under 1 GB of GPU memory. No experiment in the repository requires
a larger GPU.

---

## 6. API reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, CUDA, queue |
| GET | `/schema`, `/presets` | configurable parameters; existing configurations to reproduce/adapt |
| GET | `/experiments` | registry (filters: `source`, `category`, `status`, `dataset`, `method`, `q`) |
| GET | `/experiments/{id}` | full record (`legacy.*`, `run-*`, or `group:<run-id>` for a seed group) |
| POST | `/experiments/configure` | validate + preview (`save: true` stores proposed runs) |
| POST | `/experiments/run` | queue `config`+`seeds`, proposed `experiment_ids`, or `reproduce_of` |
| POST | `/experiments/{id}/cancel` | cancel queued or running run |
| GET | `/experiments/{id}/status`, `/metrics`, `/logs`, `/results`, `/config?format=json\|yaml` | run data and downloads |
| POST | `/experiments/compare` | side-by-side table, validated pairing, curves (`format: markdown` for a report) |
| POST | `/analysis/correlation`, `/analysis/curves`; GET `/analysis/regimes` | method-level statistics, per-task curves |
| GET | `/resources/gpu`, `/queue`, `/registry/summary`, `/paper-results` | resources, queue, overview, paper mapping |
| WS | `/ws/experiments/{id}` | live `events` / `progress` / `status` / `heartbeat` / `end` messages |

## 7. Code layout

- `studio/engine/` — `paths`, `config_loader`, `experiment_validator`, `experiment_registry`,
  `legacy_artifacts`, `training_service`, `metrics_collector`, `evaluation_service`,
  `checkpoint_manager`, `resource_monitor`, `reproducibility`, `experiment_runner`,
  `execution_queue`, `paper_results`, `fsutil`.
- `studio/api/server.py` — the FastAPI app. `studio/tests/` — 53 tests.
- `studio/experiments/` — one folder per Studio run (config, environment, events, result).
- `web/src/lib/api.ts`, `web/src/components/studio/*`, `web/src/pages/studio/*` — the Studio pages
  (Experiment Studio, Live Monitor, Results Explorer, Experiment Detail, Method Comparison,
  Scientific Analysis, Architecture Explorer, Paper Results, Documentation, System Architecture).

The only change to the research code is an optional, read-only `progress_cb` argument in
`run_arm` / `run_arm_cnn` (`src/plasticity_stress.py`); with the default `None` the computation
is unchanged, and a test enforces bit-identity.

---

## 8. Known limitations

**Execution**

1. **No resume.** `run_arm` keeps CBP utilities, the global torch RNG state and per-step lists
   inside the function, so restoring a mid-run checkpoint would not reproduce an uninterrupted
   run. Cancelled, failed or interrupted runs are re-run from the seed under a new ID.
2. **No pause.** Only cancellation (checked every 5 steps, forced after a 20 s grace period).
3. **One GPU job at a time.** The queue is deliberately serial.
4. **Checkpoints are opt-in and final-only** (`checkpoint_policy: "final"`), matching the
   original pipeline, which saved none.
5. **GPU utilization is device-wide** (NVML) and includes other processes; the per-run figure is
   torch's peak allocated memory for the runner process.
6. **Compute estimates** exist only where a matching run has been measured; otherwise they are
   reported as "not measurable yet".

**Evidence**

7. **Existing artifacts have no per-task trajectories**; curves exist only for Studio runs.
8. **Legacy configurations are reconstructed** from script constants; only e20 and e27 stored
   per-seed values, so other artifacts can be compared only at the level of means.
9. **Three Table I records reproduced exactly so far** (SGD, Shrink & Perturb and Orthogonal,
   seed 20260824, across five Studio runs). Other paper values are labelled "consistent with the
   artifact", which is not an independent reproduction.
10. **CNN effective rank** jitters by ~10⁻⁵ between CUDA runs; CNN training itself is reproducible.
11. **Phase-1 artifacts (e01–e19)** are not comparable with Phase-2 records.
12. **e18** (Phase-1 scale/mechanism check) has a script but no artifact; the Studio does not
    configure or launch it.

**Statistics**

13. Regime correlations use **6–7 method-level means**; they are descriptive, carry no confidence
    interval (shown only for n ≥ 10) and do not establish causation.
14. With **5 seeds** the exact two-sided sign test cannot go below p = 0.0625; paired t-tests are
    reported with their normality caveat. No method is declared superior automatically.

**Interface**

15. The **identity explorer** is a mathematical illustration, not a training run.
16. Designed for desktop, laptop and tablet screens; not optimized for phones.
