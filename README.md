# Weight-Norm Growth and Update Geometry in Loss of Plasticity

### A matched-step-size study of continual learning — research code, paper and interactive lab

Neural networks trained on a long stream of tasks gradually lose **plasticity**: their
ability to learn *new* tasks declines. Prior work ties this to update **magnitude**
(weight-norm growth, drift from initialization, dormant units), while subspace methods
constrain update **direction**. A normal optimizer changes both at once, so this project
separates them:

> **If the length of every update is held exactly fixed, can changing only its direction
> preserve plasticity?**

Every directional update rule is renormalized to the exact length of the plain-SGD step at the
same parameters (the *matched-step-size control*, verified to ≤ 2.3×10⁻⁷ relative deviation),
so only direction differs between methods.

**Paper:** [`Research Paper.pdf`](Research%20Paper.pdf) — *Weight-Norm Growth and Update Geometry
in Loss of Plasticity: A Matched-Step-Size Study* (Gona Lalith, Ankita Maji; IEEE ICA 2026).
LaTeX source in [`paper/`](paper).

---

## Key findings

On a 100-task permuted-MNIST stream with a 784-16-16-10 MLP (five seeds):

| Method | Family | Mean accuracy ↑ | Plasticity decay ↓ | Final ‖θ‖ | Dormant units |
|---|---|---:|---:|---:|---:|
| **L2-init** (λ = 10⁻²) | magnitude | **0.819 ± 0.003** | 0.047 ± 0.012 | 6.6 | 11.3 % |
| Shrink & Perturb | magnitude | 0.799 ± 0.010 | **0.046 ± 0.016** | 4.2 | 0.6 % |
| SGD | control | 0.744 ± 0.015 | 0.133 ± 0.036 | 20.4 | 21.2 % |
| ANR (per-unit, this project) | per-unit magnitude | 0.735 ± 0.010 | 0.090 ± 0.027 | 7.9 | 0.6 % |
| Continual Backprop | unit resets | 0.728 ± 0.010 | 0.147 ± 0.019 | 19.1 | 22.5 % |
| Orthogonal steering | matched-step direction | 0.703 ± 0.007 | 0.183 ± 0.040 | 25.1 | 28.7 % |
| Inward steering (γ = 0.6) | matched-step direction | 0.429 ± 0.005 | 0.051 ± 0.006 | 2.3 | 7.5 % |

1. **Norm growth tracks plasticity loss in the stress regime.** Across the seven rules, final
   weight norm and decay correlate at Pearson r = +0.97 (arm-level association), and
   increasing the L2-init strength lowers both.
2. **A fixed-step identity explains why direction alone struggles.** For a step of length η
   along a unit direction d, ‖θ + ηd‖² = ‖θ‖² + 2η⟨θ, d⟩ + η². Orthogonal steps must raise the
   squared norm by exactly η² (confirmed on the real runs to a relative error of 3×10⁻⁸), and any
   norm reduction requires ⟨θ, d⟩ ≤ −η/2.
3. **A radial sweep turns this into a trade-off.** Inward steering lowers the norm
   monotonically, but its best setting (γ = 0.2, accuracy 0.763) still trails the magnitude
   methods, and strong inward steering collapses learning.
4. **The link is regime-dependent.** The norm–decay association weakens at width 64 (+0.45) and
   turns negative where plasticity loss is mild (width 128: −0.52, rotated MNIST: −0.94, small
   CNN: −0.82). Weight-norm growth is a regime-specific correlate, not a universal cause.

Every number above is read from `results/tables/*.json`; see
[`EXPLANATION IN DEPTH.md`](EXPLANATION%20IN%20DEPTH.md) for the complete results, methods and
caveats.

---

## Repository layout

```
├── src/                        Research library (PyTorch)
│   ├── plasticity_stress.py    Paper core: PlasticMLP / PlasticCNN, permuted & rotated MNIST
│   │                           streams, every update rule, the shared training loop (run_arm)
│   ├── datasets.py · device.py MNIST IDX / CIFAR loaders · CUDA detection and hardware summary
│   └── …                       Phase-1 modules (update generator, β₄ statistics, probes, CL harness)
├── experiments/                e01–e27, one script per numbered experiment (paper: e20–e27)
├── configs/preregistration.py  Frozen constants, thresholds and kill conditions (Phase 1)
├── results/
│   ├── tables/                 JSON / CSV summaries — the source of truth for every number
│   └── figures/                PNG figures written by the experiments
├── paper/                      Paper source: main(iconcise).tex, references.bib, figs/
│                               + phase1_manuscript.md (the earlier Phase-1 write-up)
├── studio/                     Experimental Studio: FastAPI engine that runs real experiments
│   ├── engine/ · api/ · tests/
│   └── experiments/            One folder per Studio run (config, environment, metrics, result)
├── web/                        Plasticity Lab: React + Vite app (analysis pages + Studio UI)
├── docs/STUDIO_GUIDE.md        Studio setup, exact protocol reference, API, limitations
├── EXPLANATION IN DEPTH.md     Full explanation: architecture, results, metrics, limitations
├── decision.md · result.md     Append-only decision log (D-000 …) and experiment log (R01 …)
├── directional_plasticity_project.md   Original Phase-1 research specification
├── Research Paper.pdf          The compiled paper
├── run_all.py                  Runs the whole pipeline (e01–e27, then rebuilds the web data)
└── requirements.txt
```

---

## Setup

**Python 3.11+** (developed on 3.13). For GPU runs install the CUDA build of PyTorch first,
then the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU (optional)
pip install -r requirements.txt
```

**Node.js 18+** for the web app:

```bash
npm --prefix web install
```

**Datasets** (not included in the repository):

| Folder | Contents | Used by |
|---|---|---|
| `Deep Learning Dataset1/` | the four standard MNIST IDX files (train / t10k images and labels; both `train-images-idx3-ubyte` and `train-images.idx3-ubyte` naming work) | all paper experiments and the Studio |
| `cifar-10/` | Kaggle CIFAR-10 (`train.7z`, `trainLabels.csv`; extracted on first use via `py7zr`) | Phase-1 only |
| `cifar100/` | CIFAR-100 PNGs organized as `train/<class>/*.png` | Phase-1 only |

The loaders read these local files directly and cache CIFAR as NPZ; nothing is downloaded at
run time.

---

## Running

**Paper experiments** (each writes `results/tables/*.json` and `results/figures/*.png`; a
width-16 run takes about 30 s on an RTX 4070):

```bash
python experiments/e20_plasticity_stress.py      # Table I, Fig. 2
python experiments/e22_reviewer_experiments.py   # width 64, λ sweep, ANR ablation
python experiments/e23_update_geometry.py        # random-direction / angle / norm-projection controls
python experiments/e24_generalization.py         # rotated MNIST, width 128
python experiments/e25_cnn.py                    # small CNN on rotated MNIST
python experiments/e26_diagnostics.py            # layer-wise norms, alternative metrics
python experiments/e27_radial_sweep.py           # Fig. 3
```

**Everything** (Phase 1 and Phase 2 in dependency order, then the web dataset):

```bash
python run_all.py                    # all stages
python run_all.py --only e20 e27     # selected stages
```

**Plasticity Lab** (interactive analysis of every result file):

```bash
npm --prefix web run data    # rebuild web/public/data/plasticity.json from results/tables
npm --prefix web run dev     # open the printed localhost URL
```

**Experimental Studio** (configure, launch and monitor real runs from the browser): start the
backend, then the web app as above.

```bash
python -m uvicorn studio.api.server:app --host 127.0.0.1 --port 8765
```

**Tests:**

```bash
python -m pytest studio/tests -q
```

**Paper:** zip `paper/main(iconcise).tex`, `paper/references.bib` and `paper/figs/`, upload the
zip to Overleaf as a new project, and compile with pdfLaTeX.

---

## The web app

`web/` is a Vite + React 18 + TypeScript app (HashRouter, Recharts, three.js) with five
sections:

- **Studio** — overview, Experiment Studio (configure and queue runs), Live Monitor.
- **Results** — results explorer, method comparison, and one page per experiment (R20, λ sweep,
  radial sweep, width study, rotated MNIST, CNN probe).
- **Analysis** — scientific analysis, update geometry with an interactive identity explorer,
  architecture explorer, norm ↔ plasticity, ablations, comparative view.
- **Docs** — documentation, an interactive 3D / 2D system architecture with a guided tour,
  configuration, datasets, checkpoints, raw results.
- **Paper Results** — every number in the paper mapped to the artifact it comes from.
- **Paper** — the paper PDF (`web/public/paper/plasticity-paper.pdf`, a copy of `Research Paper.pdf`).

Data flows one way: `results/tables/*.json` → `web/scripts/build_data.py` →
`web/public/data/plasticity.json` → the pages. The analysis pages work without the backend;
the Studio pages talk to it through the Vite proxy (`/api`, including WebSockets).

---

## Scientific integrity

- **No fabricated values.** Every displayed number is copied from a result file and names its
  source; missing values are shown as "Not available", never 0.
- **Artifacts are read-only.** The web app and the Studio read `results/tables/` and never
  write to it. Each Studio run gets a new ID and a read-only configuration snapshot verified by
  SHA-256 before it executes.
- **Reproduction is checked, not assumed.** A Studio run is labelled "reproduced" only when its
  configuration equals a per-seed legacy record and its metrics match it; the SGD and orthogonal
  arms of Table I (seed 20260824) reproduce with every metric difference exactly 0.
- **Association, not causation.** Correlations across methods are labelled as method-level
  associations with their n, shown alongside the regimes where the sign reverses.
- **Pre-registration and append-only logs.** Phase-1 thresholds and kill conditions are frozen
  in `configs/preregistration.py`; `decision.md` and `result.md` record every decision and run,
  including the negative results.

---

## Project history

The project ran in two phases. **Phase 1** (e01–e19) asked whether update direction carries an
independently controllable effect on future trainability, using matched-magnitude update
generators and a β₄ regression on Split-MNIST and Split-CIFAR. The scalar effect was real and
predictable from update–probe gradient geometry (held-out r = 0.977 on Split-MNIST), but no
seed-stable, controllable directional axis existed, and the test-beds showed little genuine
plasticity loss. **Phase 2** (e20–e27, the paper) moved to a regime that clearly induces loss of
plasticity and tested direction against magnitude under the matched-step control. The Phase-1
write-up is kept in `paper/phase1_manuscript.md`, with the full record in `decision.md` and
`result.md`.

---

## Tech stack

| Layer | Technology |
|---|---|
| Deep learning | PyTorch 2.x (CUDA 12.8 build on an RTX 4070 Laptop GPU, 8 GB) |
| Scientific Python | NumPy, SciPy, pandas, statsmodels (HC3-robust OLS, VIF), scikit-learn, matplotlib |
| Studio backend | FastAPI, uvicorn, WebSockets, NVML (GPU monitoring) |
| Web | React 18, TypeScript, Vite 5, React Router, Recharts, three.js |

---

*Gona Lalith · Ankita Maji · 2026*
