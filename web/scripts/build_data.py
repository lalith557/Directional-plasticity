#!/usr/bin/env python
"""
build_data.py  --  Plasticity Lab data layer.

Reads the REAL released experiment summaries in results/tables/*.json and emits a
single normalized dataset at web/public/data/plasticity.json that drives the UI.

Scientific-integrity rules encoded here:
  * Every number is copied from a released summary; nothing is invented.
  * NaN (present in e25_cnn random_dir) is converted to null, never to 0.
  * Match-step deviation is only marked "applicable" for the fixed-step directional
    interventions; magnitude/reset methods are flagged so the UI can show N/A.
  * Per-task histories do NOT exist in these summaries, so no curve is fabricated.
Run:  python web/scripts/build_data.py
"""
import json, math, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
OUT = ROOT / "web" / "public" / "data" / "plasticity.json"


def load(name):
    with open(TABLES / f"{name}.json", "r", encoding="utf-8") as f:
        return json.load(f)  # json.load parses NaN -> float('nan')


def clean(x):
    """Recursively convert non-finite floats to None (JSON-safe, no fabrication)."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x


def pearson(xs, ys):
    pairs = [(a, b) for a, b in zip(xs, ys) if a is not None and b is not None]
    n = len(pairs)
    if n < 2:
        return None
    xm = sum(a for a, _ in pairs) / n
    ym = sum(b for _, b in pairs) / n
    num = sum((a - xm) * (b - ym) for a, b in pairs)
    dx = math.sqrt(sum((a - xm) ** 2 for a, _ in pairs))
    dy = math.sqrt(sum((b - ym) ** 2 for _, b in pairs))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def spearman(xs, ys):
    pairs = [(a, b) for a, b in zip(xs, ys) if a is not None and b is not None]
    if len(pairs) < 2:
        return None

    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    xr = rank([a for a, _ in pairs])
    yr = rank([b for _, b in pairs])
    return pearson(xr, yr)


# ---- method display metadata (keys as they appear in the summaries) ----------
METHODS = {
    "sgd":                  dict(short="SGD", name="SGD", tag="Baseline",
                                 family="control", normControl="none", matchStep="reference"),
    "l2_init":              dict(short="L2", name="L2-init", tag="Small initial norm",
                                 family="magnitude", normControl="regularize to init", matchStep="na"),
    "shrink_pert":          dict(short="S&P", name="Shrink & Perturb", tag="Controlled shrinkage",
                                 family="magnitude", normControl="shrink + noise", matchStep="na"),
    "adaptive_norm_revive": dict(short="ANR", name="ANR", tag="Adaptive-Norm-Revive",
                                 family="magnitude", normControl="per-unit target", matchStep="na"),
    "cbp":                  dict(short="CBP", name="Continual Backprop", tag="Utility-based unit resets",
                                 family="control", normControl="unit resets", matchStep="na"),
    "dir_control_c1.0":     dict(short="Dir", name="Directional", tag="Orthogonal to weight vector",
                                 family="directional", normControl="Cor. 1 (orthogonal)", matchStep="applicable"),
    "dir_inward_g0.6":      dict(short="In", name="Inward", tag="Along negative weight vector",
                                 family="directional", normControl="Cor. 2 (inward)", matchStep="applicable"),
    # extra keys that appear in CNN regime
    "dir_control":          dict(short="Dir", name="Directional", tag="Orthogonal to weight vector",
                                 family="directional", normControl="orthogonal", matchStep="applicable"),
    "dir_inward":           dict(short="In", name="Inward", tag="Along negative weight vector",
                                 family="directional", normControl="inward", matchStep="applicable"),
    "norm_project":         dict(short="NP", name="Norm-Project", tag="Projected norm control",
                                 family="magnitude", normControl="norm projection", matchStep="na"),
    "random_dir":           dict(short="Rnd", name="Random direction", tag="Random unit step",
                                 family="directional", normControl="random", matchStep="applicable"),
}


def method_meta(key):
    m = METHODS.get(key)
    if m:
        return dict(key=key, **m)
    return dict(key=key, short=key, name=key, tag="", family="other",
                normControl="", matchStep="na")


def arm_row(key, agg):
    """Normalize one aggregate entry into a UI arm row (metric + std)."""
    g = lambda k: agg.get(k)
    return dict(
        **method_meta(key),
        acc=g("mean_acc_mean"), accStd=g("mean_acc_std"),
        decay=g("decay_mean"), decayStd=g("decay_std"),
        norm=g("final_wl2_mean"), normStd=g("final_wl2_std"),
        dormant=g("final_dormant_mean"), dormantStd=g("final_dormant_std"),
        effRank=g("final_eff_rank_mean"), effRankStd=g("final_eff_rank_std"),
        matchDev=g("matched_max_dev_mean"),
        peakAcc=g("peak_acc_mean"), endAcc=g("end_acc_mean"),
    )


# canonicalize regime-specific keys so the same method unifies across regimes
CANON = {"dir_control": "dir_control_c1.0", "dir_inward": "dir_inward_g0.6"}


def regime_points(agg, order=None):
    keys = order or list(agg.keys())
    pts = []
    for k in keys:
        if k not in agg:
            continue
        a = agg[k]
        ck = CANON.get(k, k)
        pts.append(dict(
            **method_meta(ck),
            norm=a.get("final_wl2_mean"), decay=a.get("decay_mean"),
            acc=a.get("mean_acc_mean"), dormant=a.get("final_dormant_mean"),
        ))
    return pts


MLP_ORDER = ["sgd", "l2_init", "shrink_pert", "cbp",
             "dir_control_c1.0", "dir_inward_g0.6", "adaptive_norm_revive"]

# ============================ load sources ===================================
e20 = load("e20_summary")
e22_lambda = load("e22_lambda_summary")
e27 = load("e27_radial_sweep_summary")
e22_arch64 = load("e22_arch64_summary")
e24_arch128 = load("e24_arch128_summary")
e24_rot = load("e24_rotmnist_summary")
e25 = load("e25_cnn_summary")
e22_anr = load("e22_anr_ablation_summary")

# ---- R20 (primary) ----------------------------------------------------------
r20_arms = [arm_row(k, e20["aggregate"][k]) for k in MLP_ORDER]
r20_perseed = {}
for k, runs in e20["per_seed_raw"].items():
    r20_perseed[k] = [
        dict(seed=e20["seeds"][i],
             acc=r["mean_acc"], decay=r["decay"], norm=r["final_wl2"],
             dormant=r["final_dormant"], effRank=r["final_eff_rank"],
             peakAcc=r["peak_acc"], endAcc=r["end_acc"], matchDev=r["matched_max_dev"])
        for i, r in enumerate(runs)
    ]

# ---- width-16 correlation (computed here from the 7 arm means) --------------
w16_norm = [e20["aggregate"][k]["final_wl2_mean"] for k in MLP_ORDER]
w16_decay = [e20["aggregate"][k]["decay_mean"] for k in MLP_ORDER]
w16_acc = [e20["aggregate"][k]["mean_acc_mean"] for k in MLP_ORDER]
w16_r = pearson(w16_norm, w16_decay)
w16_rho = spearman(w16_norm, w16_decay)
w16_r_acc = pearson(w16_norm, w16_acc)  # rank-vs-acc secondary
effrank_acc = pearson(
    [e20["aggregate"][k]["final_eff_rank_mean"] for k in MLP_ORDER], w16_acc)
effrank_decay = pearson(
    [e20["aggregate"][k]["final_eff_rank_mean"] for k in MLP_ORDER], w16_decay)

# ---- lambda sweep -----------------------------------------------------------
lam_pts = []
for lam in e22_lambda["lambdas"]:
    key = str(int(lam)) if lam == int(lam) else str(lam)
    a = e22_lambda["aggregate"].get(key) or e22_lambda["aggregate"].get(str(lam))
    lam_pts.append(dict(
        lam=lam, acc=a["mean_acc_mean"], accStd=a["mean_acc_std"],
        decay=a["decay_mean"], decayStd=a["decay_std"],
        norm=a["final_wl2_mean"], normStd=a["final_wl2_std"],
        dormant=a["final_dormant_mean"], dormantStd=a["final_dormant_std"],
    ))

# ---- radial (gamma) sweep ---------------------------------------------------
rad_pts = []
for g in e27["gammas"]:
    key = str(int(g)) if g == int(g) else str(g)
    a = e27["aggregate"].get(key) or e27["aggregate"].get(str(g))
    rad_pts.append(dict(
        gamma=g, acc=a["mean_acc_mean"], accStd=a["mean_acc_std"],
        decay=a["decay_mean"], decayStd=a["decay_std"],
        norm=a["final_wl2_mean"], normStd=a["final_wl2_std"],
        dormant=a["final_dormant_mean"], dormantStd=a["final_dormant_std"],
        effRank=a["final_eff_rank_mean"], radialAlign=a["radial_align_mean_mean"],
        matchDev=a["matched_max_dev_mean"],
    ))
rad_perseed = {}
for g, runs in e27["per_seed_raw"].items():
    rad_perseed[g] = [
        dict(seed=e27["seeds"][i], acc=r["mean_acc"], decay=r["decay"],
             norm=r["final_wl2"], dormant=r["final_dormant"])
        for i, r in enumerate(runs)
    ]

# ---- ANR ablation -----------------------------------------------------------
ABL_ORDER = ["anr_full", "anr_revive_only", "anr_shrink_only"]
ABL_NAMES = {"anr_full": "ANR (full)",
             "anr_revive_only": "ANR (revive-only)",
             "anr_shrink_only": "ANR (shrink-only)"}
ABL_TAG = {"anr_full": "both mechanisms",
           "anr_revive_only": "no per-unit cap",
           "anr_shrink_only": "no revival"}
abl_variants = []
for k in ABL_ORDER + ["sgd", "shrink_pert"]:
    a = e22_anr["aggregate"][k]
    nm = ABL_NAMES.get(k) or method_meta(k)["name"]
    abl_variants.append(dict(
        key=k, name=nm, tag=ABL_TAG.get(k, ""),
        reference=k in ("sgd", "shrink_pert"),
        acc=a["mean_acc_mean"], accStd=a["mean_acc_std"],
        decay=a["decay_mean"], decayStd=a["decay_std"],
        norm=a["final_wl2_mean"], normStd=a["final_wl2_std"],
        dormant=a["final_dormant_mean"], dormantStd=a["final_dormant_std"],
    ))

# ---- regimes (boundary-condition story) -------------------------------------
regimes = [
    dict(id="width16", name="Permuted MNIST · MLP width 16",
         short="Width 16", dataset="Permuted MNIST", arch="MLP (784-16-16-10)",
         seeds=len(e20["seeds"]), arms=len(MLP_ORDER),
         pearson=w16_r, computedHere=True, source="e20_summary.json",
         points=regime_points(e20["aggregate"], MLP_ORDER)),
    dict(id="width64", name="Permuted MNIST · MLP width 64",
         short="Width 64", dataset="Permuted MNIST", arch=f"MLP ({e22_arch64['arch']})",
         seeds=len(e22_arch64["seeds"]), arms=e22_arch64.get("n_arms", 7),
         pearson=e22_arch64["pearson_norm_decay_narms"], source="e22_arch64_summary.json",
         points=regime_points(e22_arch64["aggregate"], MLP_ORDER)),
    dict(id="width128", name="Permuted MNIST · MLP width 128",
         short="Width 128", dataset="Permuted MNIST", arch=f"MLP ({e24_arch128['arch']})",
         seeds=len(e24_arch128["seeds"]), arms=7,
         pearson=e24_arch128["pearson_norm_decay_narms"], source="e24_arch128_summary.json",
         points=regime_points(e24_arch128["aggregate"], MLP_ORDER)),
    dict(id="rot_mlp", name="Rotated MNIST · MLP width 16",
         short="Rotated (MLP)", dataset="Rotated MNIST", arch="MLP (784-16-16-10)",
         seeds=len(e24_rot["seeds"]), arms=7,
         pearson=e24_rot["pearson_norm_decay_narms"], source="e24_rotmnist_summary.json",
         points=regime_points(e24_rot["aggregate"], MLP_ORDER)),
    dict(id="rot_cnn", name="Rotated MNIST · Small CNN",
         short="Rotated (CNN)", dataset="Rotated MNIST", arch=e25["arch"],
         seeds=len(e25["seeds"]), arms=e25.get("n_arms", 6),
         pearson=e25["pearson_norm_decay_narms"], source="e25_cnn_summary.json",
         points=regime_points(e25["arm_aggregate"],
                              ["sgd", "l2_init", "shrink_pert", "norm_project",
                               "dir_control", "dir_inward", "random_dir"])),
]

# ---- CNN angle sweep (update-geometry evidence) -----------------------------
angle_pts = []
for c in e25["angle_cos"]:
    key = str(int(c)) if c == int(c) else str(c)
    a = e25["angle_aggregate"].get(key) or e25["angle_aggregate"].get(str(c))
    angle_pts.append(dict(cos=c, acc=a["mean_acc_mean"], accStd=a["mean_acc_std"],
                          norm=a["final_wl2_mean"], normStd=a["final_wl2_std"],
                          gradAlign=a["grad_align_mean"]))

# ---- architectures / datasets (derived) -------------------------------------
architectures = sorted({r["arch"] for r in regimes})
datasets = sorted({r["dataset"] for r in regimes})

# ---- raw file inventory (for Raw Results page) ------------------------------
USED = ["e20_summary", "e22_lambda_summary", "e27_radial_sweep_summary",
        "e22_arch64_summary", "e24_arch128_summary", "e24_rotmnist_summary",
        "e25_cnn_summary", "e22_anr_ablation_summary"]
raw_files = []
# Also stage a copy of each referenced JSON into web/public/data/raw/ so the
# Checkpoints page can serve them as direct downloads. The source file in
# results/tables remains the single source of truth; this is just a static
# web mirror.
import shutil
RAW_OUT = ROOT / "web" / "public" / "data" / "raw"
RAW_OUT.mkdir(parents=True, exist_ok=True)
for n in USED:
    p = TABLES / f"{n}.json"
    raw_files.append(dict(name=f"{n}.json",
                          path=f"results/tables/{n}.json",
                          bytes=p.stat().st_size))
    shutil.copy2(p, RAW_OUT / f"{n}.json")

# ============================ assemble ========================================
data = dict(
    generatedAt=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    meta=dict(
        arms=len(MLP_ORDER),
        seedsPerArm=len(e20["seeds"]),
        tasks=e20["regime"]["n_tasks"],
        architectures=len(architectures),
        datasets=len(datasets),
        architectureList=architectures,
        datasetList=datasets,
        hardware=e20["hardware"],
        matchedStepTolerance="1e-4",
    ),
    r20=dict(
        id="r20", name="R20 — Plasticity Stress", status="Completed",
        subtitle="Long-horizon continual learning on Permuted-MNIST with a shared classifier.",
        dataset="Permuted-MNIST", architecture="MLP (width 16)", arch="784-16-16-10",
        config=dict(
            dataset="Permuted-MNIST", architecture="MLP (width 16)",
            tasks=e20["regime"]["n_tasks"], stepsPerTask=e20["regime"]["steps_per_task"],
            lr=e20["regime"]["lr"], trainSubset=e20["regime"]["train_subset"],
            batch=e20["regime"]["batch"], epochs=e20["regime"]["epochs"],
            classifier="Shared 10-way head", seeds=len(e20["seeds"]),
            hidden=e20["regime"]["hidden"], depth=e20["regime"]["depth"],
        ),
        seeds=e20["seeds"],
        arms=r20_arms, perSeed=r20_perseed,
        verdict=e20.get("method_verdict"),
        source="results/tables/e20_summary.json",
    ),
    lambda_=dict(
        values=e22_lambda["lambdas"], points=lam_pts, seeds=len(e22_lambda["seeds"]),
        grid=e22_lambda.get("grid"), source="results/tables/e22_lambda_summary.json",
        observation="Moderate controlled norm reduction (lambda ~ 1e-2) performs best; "
                    "excessive regularization (lambda = 1e-1) becomes harmful.",
    ),
    radial=dict(
        gammas=e27["gammas"], points=rad_pts, perSeed=rad_perseed,
        seeds=len(e27["seeds"]), note=e27.get("note"),
        source="results/tables/e27_radial_sweep_summary.json",
        observation="Increasing inward radial allocation reduces the parameter norm "
                    "monotonically, but accuracy does not improve monotonically: a mild "
                    "bias beats SGD while strong inward steering collapses learning. "
                    "Norm reduction alone is not sufficient.",
    ),
    ablation=dict(
        variants=abl_variants, seeds=len(e22_anr["seeds"]),
        source="results/tables/e22_anr_ablation_summary.json",
        observation="In this ablation, removing the per-unit norm cap (revive-only) improves "
                    "accuracy relative to full ANR, whereas removing revival (shrink-only) is "
                    "associated with lower accuracy, greater plasticity decay, and more dormant "
                    "units. These results implicate the cap as a contributor to ANR's "
                    "underperformance in the evaluated regime, but do not establish it as the "
                    "sole cause.",
    ),
    regimes=regimes,
    angle=dict(cos=e25["angle_cos"], points=angle_pts,
               arch=e25["arch"], source="results/tables/e25_cnn_summary.json"),
    normPlasticity=dict(
        primary=dict(pearson=w16_r, spearman=w16_rho, n=len(MLP_ORDER),
                     accPearson=w16_r_acc,
                     points=[dict(**method_meta(k),
                                  norm=e20["aggregate"][k]["final_wl2_mean"],
                                  decay=e20["aggregate"][k]["decay_mean"],
                                  acc=e20["aggregate"][k]["mean_acc_mean"])
                             for k in MLP_ORDER]),
        effRankAcc=effrank_acc, effRankDecay=effrank_decay,
        source="results/tables/e20_summary.json (arm-level, n=7)",
    ),
    rawFiles=raw_files,
)

data = clean(data)
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, allow_nan=False)  # allow_nan=False guarantees valid JSON

print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size} bytes)")
print(f"width-16 Pearson r = {w16_r:.4f}  Spearman = {w16_rho:.4f}  n=7")
print("regime r:", {r['short']: (round(r['pearson'], 3) if r['pearson'] is not None else None) for r in regimes})
print("eff-rank vs acc r =", round(effrank_acc, 3), " eff-rank vs decay r =", round(effrank_decay, 3))
