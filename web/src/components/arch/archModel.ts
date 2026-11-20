/* Single source of truth for the system-architecture views (2D diagram and 3D workflow).
   Every description is taken from the code it names; numbers are the implemented values. */
import type { IconName } from "./icons";

export type LayerId = "researcher" | "frontend" | "backend" | "datasets" | "engine" | "metrics" | "storage";

export type Detail = {
  title: string;
  sub?: string;
  body: string;
  code?: string[];
  api?: string[];
  facts?: string[];
};

export type Layer = { id: LayerId; label: string; tag?: string; color: string; icon: IconName; detail: Detail };
export type Comp = { id: string; layer: LayerId; label: string; sub?: string; icon: IconName; detail: Detail; accent?: boolean };

export type FlowKind = "control" | "request" | "live" | "data" | "update" | "metrics" | "storage";
export type Flow = {
  id: string; from: string; to: string; kind: FlowKind; label: string; detail: string;
  main?: boolean; bend?: [number, number, number];
};
export type TourStep = { title: string; text: string; layer: LayerId; flows: string[] };

export const FLOW_KINDS: Record<FlowKind, { label: string; color: string; hint: string }> = {
  control: { label: "Control", color: "#8b5cf6", hint: "user actions, process launch, next task" },
  request: { label: "HTTP request / response", color: "#3b82f6", hint: "REST calls between UI and API" },
  live: { label: "Live stream", color: "#06b6d4", hint: "progress callback, WebSocket, NVML" },
  data: { label: "Training data", color: "#10b981", hint: "task streams, minibatches, GPU work" },
  update: { label: "Gradient / update", color: "#f59e0b", hint: "∇L, Δθ, update rule" },
  metrics: { label: "Diagnostics", color: "#ec4899", hint: "per-task and per-step measurements" },
  storage: { label: "File I/O", color: "#94a3b8", hint: "snapshots, results, logs, checkpoints" },
};

export const LAYERS: Layer[] = [
  {
    id: "researcher", label: "Researcher", tag: "User", color: "#3b6fd6", icon: "user",
    detail: {
      title: "Researcher (User)",
      body: "Designs, launches and inspects experiments in the web UI, or calls the API directly. Nothing runs until Run is pressed; submissions estimated above 600 s need explicit confirmation.",
      facts: ["Each seed becomes its own run with a new ID", "Reproductions never overwrite: they get new IDs"],
    },
  },
  {
    id: "frontend", label: "Frontend", tag: "React + TypeScript", color: "#4a6fa5", icon: "home",
    detail: {
      title: "Frontend (React + TypeScript)",
      body: "Single-page app built with Vite 5, React 18, TypeScript and Recharts. It reaches the backend only through /api, which Vite proxies, WebSockets included, to 127.0.0.1:8765.",
      code: ["web/src/pages/studio/*", "web/src/lib/api.ts", "web/vite.config.ts"],
      facts: ["Pages built on the frozen artifacts keep working without the backend"],
    },
  },
  {
    id: "backend", label: "Backend", tag: "FastAPI", color: "#7a5bb5", icon: "api",
    detail: {
      title: "Backend (FastAPI)",
      body: "FastAPI app served by uvicorn on 127.0.0.1:8765: 21 HTTP endpoints and one WebSocket. It never trains in-process; every run is a separate Python subprocess.",
      code: ["studio/api/server.py", "studio/engine/*"],
      facts: ["Lifespan hook starts and stops the execution queue", "CORS limited to localhost"],
    },
  },
  {
    id: "datasets", label: "Datasets & Task Generator", color: "#4f8f5b", icon: "image",
    detail: {
      title: "Datasets & Task Generator",
      body: "MNIST IDX files are loaded once by load_mnist and turned into task streams. The seed fixes every permutation, rotation, training subset and batch order.",
      code: ["src/datasets.py", "src/plasticity_stress.py (PermutedMNIST, RotatedMNIST)"],
    },
  },
  {
    id: "engine", label: "Experiment Engine", tag: "PyTorch", color: "#2f6fb0", icon: "flame",
    detail: {
      title: "Experiment Engine (PyTorch)",
      body: "The original research loop, run_arm (MLP) and run_arm_cnn (CNN), unchanged apart from an optional read-only progress hook. Runs on CUDA.",
      code: ["src/plasticity_stress.py"],
      facts: ["100 tasks × 60 SGD steps, batch 128", "η₀ = 0.2 (MLP) · 0.1 (CNN)", "Paper seeds 20260824 – 20260828"],
    },
  },
  {
    id: "metrics", label: "Metrics & Analysis", color: "#c8862a", icon: "bars",
    detail: {
      title: "Metrics & Analysis",
      body: "The quantities the paper reports, computed exactly as the code defines them (see Documentation → metric definitions).",
      code: ["src/plasticity_stress.py", "studio/engine/metrics_collector.py", "studio/engine/evaluation_service.py"],
    },
  },
  {
    id: "storage", label: "Storage & Compute", tag: "Layer", color: "#46679a", icon: "database",
    detail: {
      title: "Storage & Compute Layer",
      body: "Everything a run leaves behind lives in one directory per run. The frozen paper artifacts are only ever read.",
      code: ["studio/experiments/<run_id>/", "results/tables/*.json"],
    },
  },
];

export const LAYER: Record<LayerId, Layer> = Object.fromEntries(LAYERS.map((l) => [l.id, l])) as Record<LayerId, Layer>;

export const COMPS: Comp[] = [
  // ── Researcher ──────────────────────────────────────────────────────────────
  { id: "researcher", layer: "researcher", label: "Researcher", sub: "User", icon: "user", detail: LAYERS[0].detail },

  // ── Frontend ────────────────────────────────────────────────────────────────
  { id: "fe_overview", layer: "frontend", label: "Overview", icon: "home", detail: {
    title: "Overview", sub: "route /",
    body: "Landing page: headline findings from the frozen result tables plus a live Studio status strip (run counts, GPU, recent runs).",
    api: ["GET /registry/summary", "GET /resources/gpu"], code: ["web/src/pages/Overview.tsx", "web/src/components/studio/StudioStatus.tsx"] } },
  { id: "fe_studio", layer: "frontend", label: "Experiment Studio", icon: "flask", detail: {
    title: "Experiment Studio", sub: "route /studio",
    body: "A form generated from the schema, exposing only the parameters run_arm / run_arm_cnn implement. Presets reproduce the paper's arms. Every edit is validated by the backend in a live preview: warnings for departures from the paper protocol, a compute estimate and the config hash.",
    api: ["GET /schema", "GET /presets", "POST /experiments/configure", "POST /experiments/run"], code: ["web/src/pages/studio/ExperimentStudio.tsx"] } },
  { id: "fe_monitor", layer: "frontend", label: "Live Monitor", icon: "play", detail: {
    title: "Live Monitor", sub: "route /monitor/:id",
    body: "Follows a running experiment over a WebSocket: task and step progress, latest accuracy, ‖θ‖, dormancy and effective rank, per-task charts, NVML GPU readings, the runner log, and cancel.",
    api: ["WS /ws/experiments/{id}", "GET /experiments/{id}/logs", "POST /experiments/{id}/cancel"], code: ["web/src/pages/studio/LiveMonitor.tsx"] } },
  { id: "fe_analysis", layer: "frontend", label: "Scientific Analysis", icon: "bars", detail: {
    title: "Scientific Analysis", sub: "route /analysis",
    body: "Method-level norm–decay correlations (Pearson and Spearman, n always shown, confidence interval only when n ≥ 10), leave-one-out, regime comparison, per-task curves and paired seed-level tests.",
    api: ["POST /analysis/correlation", "GET /analysis/regimes", "POST /analysis/curves"], code: ["web/src/pages/studio/ScientificAnalysis.tsx"] } },
  { id: "fe_compare", layer: "frontend", label: "Method Comparison", icon: "scale", detail: {
    title: "Method Comparison", sub: "route /compare",
    body: "Side-by-side metrics for the selected records. Paired statistics only when protocols are identical and seeds are shared (exact sign test plus paired t with its caveat). Exports a Markdown report.",
    api: ["POST /experiments/compare"], code: ["web/src/pages/studio/MethodComparison.tsx"] } },
  { id: "fe_geometry", layer: "frontend", label: "Update Geometry", icon: "cube", detail: {
    title: "Update Geometry", sub: "route /analysis/geometry",
    body: "The fixed-step identity ‖θ + ηd‖² = ‖θ‖² + 2η⟨θ, d⟩ + η² with an interactive explorer (a mathematical illustration, not a training run) and the radial-sweep results.",
    code: ["web/src/pages/UpdateGeometry.tsx", "web/src/components/studio/IdentityExplorer.tsx"] } },
  { id: "fe_explorer", layer: "frontend", label: "Results Explorer", icon: "layers", detail: {
    title: "Results Explorer", sub: "route /explorer",
    body: "Every record in one table: the 92 frozen artifact records and all Studio runs, with status (completed, diverged, superseded, incomplete), filters and a detail page that shows provenance.",
    api: ["GET /experiments", "GET /experiments/{id}"], code: ["web/src/pages/studio/ResultsExplorer.tsx", "web/src/pages/studio/ExperimentDetail.tsx"] } },
  { id: "fe_paper", layer: "frontend", label: "Paper Results", icon: "file", detail: {
    title: "Paper Results", sub: "route /paper-results",
    body: "Maps each number printed in the paper to the artifact it came from. 68 values are recomputed and compared at the paper's printed precision; each quotes its LaTeX source.",
    api: ["GET /paper-results"], code: ["web/src/pages/studio/PaperResults.tsx", "studio/engine/paper_results.py"] } },
  { id: "fe_docs", layer: "frontend", label: "Documentation", icon: "book", detail: {
    title: "Documentation", sub: "route /docs",
    body: "Metric definitions exactly as implemented, the integrity rules, known limitations and this architecture.",
    code: ["web/src/pages/studio/Documentation.tsx", "docs/STUDIO_GUIDE.md"] } },

  // ── Backend ─────────────────────────────────────────────────────────────────
  { id: "svc_experiment", layer: "backend", label: "Experiment Service", icon: "gear", detail: {
    title: "Experiment Service", sub: "queue + runner",
    body: "Single-GPU first-in-first-out queue that runs one experiment at a time. Each run is launched as `python -m studio.engine.experiment_runner <run_id>`. Cancelling writes a CANCEL file that the run checks every 5 steps; it is force-stopped after a 20 s grace period.",
    code: ["studio/engine/execution_queue.py", "studio/engine/experiment_runner.py"],
    api: ["POST /experiments/run", "POST /experiments/{id}/cancel", "GET /queue"],
    facts: ["A CUDA request without CUDA fails instead of silently falling back to CPU", "Runs interrupted by a restart are marked failed, never resumed"] } },
  { id: "svc_config", layer: "backend", label: "Config Service", icon: "layers", detail: {
    title: "Config Service",
    body: "Schema of everything the training code supports. It normalises each request (unknown keys rejected, defaults are the paper values), hashes the scientific part with SHA-256, and exports JSON or YAML.",
    code: ["studio/engine/config_loader.py"], api: ["GET /schema", "POST /experiments/configure", "GET /experiments/{id}/config"] } },
  { id: "svc_registry", layer: "backend", label: "Registry Service", icon: "database", detail: {
    title: "Registry Service",
    body: "Creates run IDs (run-YYYYMMDD-HHMMSS-xxxxxx) and run directories with a read-only config.json. Indexes the 92 frozen artifact records, read-only, so new runs and paper results sit side by side.",
    code: ["studio/engine/experiment_registry.py", "studio/engine/legacy_artifacts.py"],
    api: ["GET /experiments", "GET /experiments/{id}", "GET /registry/summary"] } },
  { id: "svc_metrics", layer: "backend", label: "Metrics Service", icon: "trend", detail: {
    title: "Metrics Service",
    body: "Turns the per-task stream from the training loop into records (step norm, update–gradient alignment, radial and tangential components) and paper-definition summaries. Computes method-level statistics.",
    code: ["studio/engine/metrics_collector.py", "studio/engine/evaluation_service.py"],
    api: ["GET /experiments/{id}/metrics", "POST /analysis/correlation", "WS /ws/experiments/{id}"] } },
  { id: "svc_resource", layer: "backend", label: "Resource (GPU) Service", icon: "chip", detail: {
    title: "Resource (GPU) Service",
    body: "Reads the device through torch and NVML: utilisation, memory, temperature and power. Compute estimates use measured Studio timings only; with none available the estimate reads “not measurable yet”.",
    code: ["studio/engine/resource_monitor.py"], api: ["GET /resources/gpu"],
    facts: ["Utilisation is device-wide, so it includes other processes"] } },
  { id: "svc_log", layer: "backend", label: "Log Service", icon: "file", detail: {
    title: "Log Service",
    body: "Serves each run's runner output (log.txt), its event stream (events.jsonl) and its lifecycle status.",
    code: ["studio/engine/experiment_registry.py (read_log, read_events)"], api: ["GET /experiments/{id}/logs", "GET /experiments/{id}/status"] } },
  { id: "svc_result", layer: "backend", label: "Result Service", icon: "cube", detail: {
    title: "Result Service",
    body: "Serves run summaries (result.json: paper-definition metrics plus per-task arrays), seed-group aggregates and comparisons, and optional final checkpoints with their SHA-256.",
    code: ["studio/engine/experiment_registry.py", "studio/engine/checkpoint_manager.py"],
    api: ["GET /experiments/{id}/results", "POST /experiments/compare"] } },
  { id: "svc_validation", layer: "backend", label: "Validation Service", icon: "shield", detail: {
    title: "Validation Service",
    body: "Checks compatibility and ranges and warns on every departure from the paper protocol. Re-verifies the config hash before execution, compares a finished run with its per-seed paper record (exact ≤ 10⁻⁹, within tolerance ≤ 10⁻⁴), and checks paper values against the artifacts.",
    code: ["studio/engine/experiment_validator.py", "studio/engine/paper_results.py"], api: ["GET /paper-results"],
    facts: ["Table I SGD and Orthogonal rows (seed 20260824) reproduced with every metric difference exactly 0"] } },

  // ── Datasets ────────────────────────────────────────────────────────────────
  { id: "ds_permuted", layer: "datasets", label: "Permuted MNIST", sub: "100 tasks", icon: "image", detail: {
    title: "Permuted MNIST", sub: "100 tasks",
    body: "Each task applies its own fixed random pixel permutation to MNIST. Labels are unchanged, so the network keeps one shared 10-way head. 6 000 training images per task, tested on the full permuted 10 000-image test set.",
    facts: ["The standard loss-of-plasticity benchmark", "MLP only: permuting pixels destroys the spatial structure convolutions rely on"] } },
  { id: "ds_rotated", layer: "datasets", label: "Rotated MNIST", sub: "MLP / CNN", icon: "refresh", detail: {
    title: "Rotated MNIST", sub: "MLP / CNN",
    body: "Each task rotates the images by its own angle, drawn uniformly from [0°, 180°) and applied with an affine grid. 6 000 training images and a fixed 2 000-image test subset per task. The only stream used for the CNN." } },

  // ── Engine ──────────────────────────────────────────────────────────────────
  { id: "eng_input", layer: "engine", label: "Input", sub: "(x, y)", icon: "input", detail: {
    title: "Input (x, y)",
    body: "Minibatches of 128 (image, label) pairs from the current task, in seeded order. Each task starts from the weights the previous one left: no reset, no replay." } },
  { id: "eng_mlp", layer: "engine", label: "MLP", sub: "784 – h – h – 10 · ReLU", icon: "network", detail: {
    title: "MLP (784 – h – h – 10)", sub: "ReLU",
    body: "PlasticMLP: two ReLU hidden layers of width h and a shared 10-way head. h = 16 in the main stress regime (13 002 parameters); widths 64 and 128 in the width study. On Rotated MNIST a small CNN (conv16 → conv32 → FC 64 → head, 105 866 parameters) is also used.",
    code: ["src/plasticity_stress.py (PlasticMLP, PlasticCNN)"] } },
  { id: "eng_opt", layer: "engine", label: "Optimization", sub: "update rule", icon: "gear", detail: {
    title: "Optimization", sub: "update rule",
    body: "Forms the plain SGD step Δθ = −η∇L; the selected rule then turns it into the applied step Δθ'. Directional rules keep ‖Δθ'‖ = ‖Δθ‖ exactly (matched step size).",
    facts: ["The matched-step deviation is measured at every step"] } },
  { id: "eng_train", layer: "engine", label: "Training", sub: "sequential tasks", icon: "refresh", detail: {
    title: "Training", sub: "sequential tasks",
    body: "Applies θ ← θ + Δθ' for 60 steps per task, then moves to the next task. Step norms, update–gradient alignment and the radial cosine are recorded at every step." } },
  { id: "eng_eval", layer: "engine", label: "Evaluation", sub: "plasticity & metrics", icon: "bars", detail: {
    title: "Evaluation", sub: "plasticity & metrics",
    body: "After each task: test accuracy a_t, global ‖θ‖, dormant fraction and effective rank on a 1 000-image probe. They are streamed to the Studio through the read-only progress_cb hook." } },
  { id: "eng_methods", layer: "engine", label: "Update Methods", sub: "shared step size", icon: "layers", detail: {
    title: "Update Methods", sub: "shared step size",
    body: "Seven update rules share one training loop. Directional rules change only the direction of the step, at matched size. Magnitude and reset methods change θ by other means, so their matched-step deviation is reported as N/A." } },
  { id: "m_sgd", layer: "engine", label: "SGD", icon: "trend", detail: {
    title: "SGD", sub: "reference",
    body: "Plain stochastic gradient descent: the reference step every other rule is compared with." } },
  { id: "m_cbp", layer: "engine", label: "CBP", icon: "refresh", detail: {
    title: "Continual Backprop (CBP)", sub: "reset method · MLP only",
    body: "Resets low-utility hidden units during training (reset probability ρ = 10⁻² per step in the paper arm)." } },
  { id: "m_l2", layer: "engine", label: "L2-init", icon: "target", detail: {
    title: "L2-init", sub: "magnitude method",
    body: "Adds λ‖θ − θ₀‖² to the loss (λ = 10⁻²), pulling the weights back toward their initialisation." } },
  { id: "m_sp", layer: "engine", label: "ShrinkPerturb", icon: "sigma", detail: {
    title: "Shrink & Perturb", sub: "magnitude method",
    body: "At each task boundary θ ← s·θ + σ·ε with s = 0.7 and σ = 0.01 (Gaussian noise ε)." } },
  { id: "m_orth", layer: "engine", label: "Orthogonal Steering", icon: "geometry", detail: {
    title: "Orthogonal Steering", sub: "directional · matched step",
    body: "Removes the radial part of the SGD step, Δθ' = Δθ − c(Δθ·θ̂)θ̂ with c = 1, then rescales to ‖Δθ‖. The step is tangent to the ‖θ‖-sphere, yet ‖θ‖² still grows by η² per step (Cor. 1, verified per step to a relative error of 3 × 10⁻⁸)." } },
  { id: "m_inward", layer: "engine", label: "Inward Steering", icon: "input", detail: {
    title: "Inward Steering", sub: "directional · matched step",
    body: "Δθ' = normalize(Δθ − γ‖Δθ‖θ̂)·‖Δθ‖ with γ = 0.6: tilts the fixed-length step toward the origin to counter norm growth. γ ≥ 0.8 collapsed learning in the radial sweep." } },
  { id: "m_anr", layer: "engine", label: "Adaptive-Norm-Revive (ANR)", icon: "shield", accent: true, detail: {
    title: "Adaptive-Norm-Revive (ANR)", sub: "magnitude + revive · MLP only",
    body: "Between tasks it caps each hidden unit's incoming-weight norm at τ = 3 and re-initialises dormant units (mean |activation| < 10⁻³), zeroing their outgoing weights." } },

  // ── Metrics ─────────────────────────────────────────────────────────────────
  { id: "met_norm", layer: "metrics", label: "Parameter Norm", sub: "‖θ‖₂", icon: "sigma", detail: {
    title: "Parameter Norm ‖θ‖₂",
    body: "Euclidean norm of the weights. MLP tables use the backbone (hidden layers, head excluded); the CNN uses all parameters. The matched step and Cor. 1 act on the full parameter vector." } },
  { id: "met_decay", layer: "metrics", label: "Plasticity Decay", icon: "trend", detail: {
    title: "Plasticity Decay",
    body: "max_t a_t − mean of a_t over the last ⌈10 %⌉ tasks: how far the ability to learn a new task has fallen from its best.",
    facts: ["run_arm's own 'plasticity_decay' (first 20 % − last 20 %) is a different quantity, stored separately"] } },
  { id: "met_dormant", layer: "metrics", label: "Dormant Neurons", icon: "network", detail: {
    title: "Dormant Neurons",
    body: "Share of hidden ReLU units whose mean |activation| on the 1 000-image probe batch is below 10⁻³ (MLP: pooled over hidden layers; CNN: FC layer)." } },
  { id: "met_rank", layer: "metrics", label: "Effective Rank", icon: "layers", detail: {
    title: "Effective Rank",
    body: "Stable rank ‖F‖²_F / ‖F‖²₂ of the mean-centred penultimate features on the probe batch." } },
  { id: "met_geometry", layer: "metrics", label: "Update Geometry", sub: "radial / tangential", icon: "geometry", detail: {
    title: "Update Geometry", sub: "radial / tangential",
    body: "Per step: radial = ‖Δθ'‖·cos(Δθ', θ), tangential = ‖Δθ'‖·√(1 − cos²). Derived from what the loop already logs, not an extra measurement." } },

  // ── Storage & compute ──────────────────────────────────────────────────────
  { id: "st_registry", layer: "storage", label: "Experiment Registry", sub: "configs · metadata · status", icon: "gear", detail: {
    title: "Experiment Registry", sub: "configs · metadata · status",
    body: "studio/experiments/<run_id>/ holds meta.json (status, timestamps, hardware, validation), the read-only config.json with its SHA-256, and environment.json (torch and CUDA versions, code and dataset fingerprints)." } },
  { id: "st_checkpoints", layer: "storage", label: "Checkpoints", sub: "final.pt · final.json", icon: "folder", detail: {
    title: "Checkpoints", sub: "final.pt · final.json",
    body: "Optional final model state (checkpoint_policy = final) with a SHA-256 in final.json. The original pipeline saved none, so there are no intermediate checkpoints and no resume." } },
  { id: "st_results", layer: "storage", label: "Results & Logs", sub: "metrics · per-task · seeds", icon: "database", detail: {
    title: "Results & Logs", sub: "metrics · per-task · seeds",
    body: "Each run writes result.json, events.jsonl, progress.json and log.txt. The frozen paper artifacts in results/tables/*.json are read-only; only e20 and e27 store per-seed values." } },
  { id: "st_compute", layer: "storage", label: "Compute", sub: "CPU / CUDA GPU", icon: "chip", detail: {
    title: "Compute", sub: "CPU / CUDA GPU",
    body: "Runs on an NVIDIA RTX 4070 Laptop GPU (8 GB, CUDA 12.8): about 30 s and 78 MB per width-16 run. device = auto uses CUDA when it is present.",
    facts: ["No documented experiment requires an A100"] } },
];

export const COMP: Record<string, Comp> = Object.fromEntries(COMPS.map((c) => [c.id, c]));
export const METHOD_IDS = ["m_sgd", "m_cbp", "m_l2", "m_sp", "m_orth", "m_inward", "m_anr"];

export const FLOWS: Flow[] = [
  { id: "f_u_studio", from: "researcher", to: "fe_studio", kind: "control", main: true, label: "configure · run · inspect",
    detail: "The researcher picks a preset or edits the configuration; the preview is validated live before anything runs." },
  { id: "f_u_api", from: "researcher", to: "svc_experiment", kind: "request", main: true, label: "direct API (curl)",
    detail: "The same endpoints work without the UI, e.g. POST /experiments/run with a JSON configuration." },
  { id: "f_cfg", from: "fe_studio", to: "svc_config", kind: "request", label: "POST /experiments/configure",
    detail: "Every edit is sent (debounced) for normalisation, validation, a compute estimate and the config hash." },
  { id: "f_run", from: "fe_studio", to: "svc_experiment", kind: "request", main: true, label: "POST /experiments/run",
    detail: "Creates one run per seed and queues them; returns the run IDs the Live Monitor follows." },
  { id: "f_reg", from: "svc_registry", to: "st_registry", kind: "storage", main: true, label: "meta.json · read-only config.json",
    detail: "Run directories are created with a read-only snapshot; status and timestamps are updated atomically." },
  { id: "f_spawn", from: "svc_experiment", to: "eng_input", kind: "control", main: true, label: "subprocess per run",
    detail: "The queue launches python -m studio.engine.experiment_runner <run_id>, which verifies the config hash, records the environment and calls run_arm / run_arm_cnn." },
  { id: "f_snapshot", from: "st_registry", to: "eng_input", kind: "storage", label: "config.json (SHA-256 checked)",
    detail: "The runner reads the read-only snapshot and refuses to start if its hash differs from the one in meta.json." },
  { id: "f_perm", from: "ds_permuted", to: "eng_input", kind: "data", main: true, label: "task t · permuted images",
    detail: "Task t's fixed pixel permutation applied to its 6 000-image training subset." },
  { id: "f_rot", from: "ds_rotated", to: "eng_input", kind: "data", main: true, label: "task t · rotated images",
    detail: "Task t's rotation angle applied to its training subset." },
  { id: "f_p1", from: "eng_input", to: "eng_mlp", kind: "data", main: true, label: "minibatch x (128 × 784)",
    detail: "Forward pass: ReLU hidden layers to 10 logits, then the cross-entropy loss." },
  { id: "f_p2", from: "eng_mlp", to: "eng_opt", kind: "update", main: true, label: "∇L → Δθ = −η∇L",
    detail: "The backward pass gives the gradient; the plain SGD step is formed before any rule acts." },
  { id: "f_methods", from: "eng_methods", to: "eng_opt", kind: "update", main: true, label: "rule transforms the step",
    detail: "The selected method turns Δθ into Δθ', or adds a penalty or a reset." },
  { id: "f_p3", from: "eng_opt", to: "eng_train", kind: "update", main: true, label: "θ ← θ + Δθ'",
    detail: "The applied step is written into the parameters; step norms, alignment and the radial cosine are logged." },
  { id: "f_p4", from: "eng_train", to: "eng_eval", kind: "data", main: true, label: "end of task t (60 steps)",
    detail: "After 60 steps the model is evaluated on task t's test set and on the probe batch." },
  { id: "f_next", from: "eng_eval", to: "eng_input", kind: "control", main: true, label: "next task t + 1", bend: [0, 0.6, 4.6],
    detail: "Training continues on the next task from the current weights. This is where plasticity can be lost." },
  { id: "f_gpu", from: "st_compute", to: "eng_train", kind: "data", main: true, label: "CUDA forward / backward",
    detail: "Forward and backward passes run on the GPU. When CUDA is requested but missing, the runner fails rather than silently using the CPU." },
  { id: "f_m_norm", from: "eng_eval", to: "met_norm", kind: "metrics", main: true, label: "‖θ‖ per task", detail: "Global weight norm after each task." },
  { id: "f_m_decay", from: "eng_eval", to: "met_decay", kind: "metrics", main: true, label: "a_t per task", detail: "Task accuracies; decay is computed from the whole sequence." },
  { id: "f_m_dorm", from: "eng_eval", to: "met_dormant", kind: "metrics", label: "probe activations", detail: "Mean |activation| per hidden unit on the probe batch." },
  { id: "f_m_rank", from: "eng_eval", to: "met_rank", kind: "metrics", label: "penultimate features", detail: "Probe-batch features for the stable rank." },
  { id: "f_m_geo", from: "eng_opt", to: "met_geometry", kind: "metrics", label: "Δθ' vs θ every step", detail: "The applied step and the current parameters give the radial and tangential components." },
  { id: "f_progress", from: "eng_eval", to: "svc_metrics", kind: "live", main: true, label: "progress_cb → events.jsonl",
    detail: "A read-only callback streams the step position and per-task metrics. Results are bit-identical with and without it (tested)." },
  { id: "f_ws", from: "svc_metrics", to: "fe_monitor", kind: "live", main: true, label: "WebSocket /ws/experiments/{id}",
    detail: "Events, progress, status and heartbeats are pushed to the browser while the run is active." },
  { id: "f_write", from: "eng_eval", to: "st_results", kind: "storage", main: true, label: "result.json · events.jsonl",
    detail: "On completion the summary (paper metric definitions) and per-task arrays are written; then the run is validated." },
  { id: "f_ckpt", from: "eng_mlp", to: "st_checkpoints", kind: "storage", label: "state_dict → final.pt (opt-in)",
    detail: "Only when checkpoint_policy = final; the file's SHA-256 is recorded." },
  { id: "f_nvml", from: "st_compute", to: "svc_resource", kind: "live", label: "NVML readings", detail: "Utilisation, memory, temperature and power of the device." },
  { id: "f_logs", from: "st_results", to: "svc_log", kind: "storage", label: "log.txt · events.jsonl", detail: "Runner output and the per-task event log." },
  { id: "f_legacy", from: "st_results", to: "svc_validation", kind: "storage", label: "results/tables/*.json (read-only)",
    detail: "Frozen artifacts used for reproduction checks and for the paper mapping." },
  { id: "f_explorer", from: "svc_registry", to: "fe_explorer", kind: "request", label: "GET /experiments", detail: "All records with status, source and validation." },
  { id: "f_compare", from: "svc_result", to: "fe_compare", kind: "request", label: "POST /experiments/compare", detail: "Side-by-side metrics and validated pairing." },
  { id: "f_paper", from: "svc_validation", to: "fe_paper", kind: "request", label: "GET /paper-results", detail: "Paper values with their artifact sources and consistency checks." },
  { id: "f_analysis", from: "svc_metrics", to: "fe_analysis", kind: "request", label: "POST /analysis/correlation", detail: "Method-level statistics, labelled with their n." },
  { id: "f_logs_ui", from: "svc_log", to: "fe_monitor", kind: "request", label: "GET /experiments/{id}/logs", detail: "The runner log, shown in the monitor." },
  { id: "f_gpu_ui", from: "svc_resource", to: "fe_overview", kind: "request", label: "GET /resources/gpu", detail: "Live device readings for the status strip and the monitor." },
];

export const TOUR: TourStep[] = [
  { title: "Configure", layer: "frontend", flows: ["f_u_studio", "f_cfg"],
    text: "The researcher opens the Experiment Studio and picks a preset (for example Table I · Orthogonal) or edits the parameters. The backend validates each edit: unsupported keys are rejected, departures from the paper protocol are flagged, and the runtime is estimated from measured runs." },
  { title: "Submit", layer: "backend", flows: ["f_run", "f_reg"],
    text: "Run creates one run per seed. The registry gives each a new ID and writes a read-only config.json, with its SHA-256 stored in meta.json." },
  { title: "Launch", layer: "backend", flows: ["f_spawn", "f_snapshot"],
    text: "The single-GPU queue starts a separate Python process for the run. The runner re-checks the hash and records the software, hardware and dataset fingerprints before training." },
  { title: "Stream tasks", layer: "datasets", flows: ["f_perm", "f_rot"],
    text: "The task generator produces task t, a fixed pixel permutation or a rotation, from seeded MNIST subsets." },
  { title: "Forward & backward", layer: "engine", flows: ["f_p1", "f_p2", "f_gpu"],
    text: "Minibatches of 128 pass through the MLP on the GPU; the gradient gives the plain SGD step Δθ = −η∇L." },
  { title: "Apply the update rule", layer: "engine", flows: ["f_methods", "f_p3"],
    text: "The selected method transforms the step. Directional rules keep its length equal to the SGD step, so between those arms only the direction differs." },
  { title: "Next task", layer: "engine", flows: ["f_p4", "f_next"],
    text: "After 60 steps the task is evaluated, and training moves on to task t + 1 from the current weights: 100 tasks in sequence." },
  { title: "Measure plasticity", layer: "metrics", flows: ["f_m_norm", "f_m_decay", "f_m_dorm", "f_m_rank", "f_m_geo"],
    text: "Accuracy, ‖θ‖, dormant units and effective rank are recorded after every task; the radial/tangential split of the update is recorded at every step." },
  { title: "Watch live", layer: "frontend", flows: ["f_progress", "f_ws", "f_logs_ui"],
    text: "A read-only callback streams each finished task to the backend, which pushes it over a WebSocket to the Live Monitor." },
  { title: "Store & validate", layer: "storage", flows: ["f_write", "f_legacy", "f_explorer", "f_paper"],
    text: "result.json and the event log are written. If the configuration matches a per-seed paper record, the run is compared with it (exact, within tolerance, or mismatch) and appears in the Results Explorer." },
];

export const flowsOfLayer = (layer: LayerId) =>
  FLOWS.filter((f) => COMP[f.from]?.layer === layer || COMP[f.to]?.layer === layer);
