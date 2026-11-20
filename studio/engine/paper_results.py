"""Map the paper's findings onto the evidence that supports them.

Four kinds of value are kept strictly apart:

* ``paper``     — the number as printed in paper/main(iconcise).tex (reference set below;
                  each carries the exact LaTeX snippet it was read from — tested for drift)
* ``artifact``  — the same quantity recomputed from results/tables/*.json
* ``reproduced``— Studio runs whose configuration equals a per-seed legacy record and whose
                  metrics matched it (validation status reproduced_*)
* ``generated`` — Studio runs with no legacy counterpart (new evidence, not a reproduction)
"""
from __future__ import annotations

import math

from . import evaluation_service as ev
from . import experiment_registry as reg
from . import legacy_artifacts as la

TABLE_I_ORDER = ["l2_init", "shrink_pert", "adaptive_norm_revive", "sgd", "cbp",
                 "dir_control_c1.0", "dir_inward_g0.6"]

# value, decimals (for the rounding check), LaTeX anchor that must appear in the .tex
PAPER = {
    "table1": {
        "l2_init": dict(acc=(0.819, 3), acc_std=(0.003, 3), decay=(0.047, 3), decay_std=(0.012, 3),
                        norm=(6.6, 1), dormant=(0.113, 3), tex=r"\textbf{0.819 $\pm$ 0.003} & 0.047 $\pm$ 0.012 & 6.6  & 0.113"),
        "shrink_pert": dict(acc=(0.799, 3), acc_std=(0.010, 3), decay=(0.046, 3), decay_std=(0.016, 3),
                            norm=(4.2, 1), dormant=(0.006, 3), tex=r"0.799 $\pm$ 0.010          & 0.046 $\pm$ 0.016 & 4.2  & 0.006"),
        "adaptive_norm_revive": dict(acc=(0.735, 3), acc_std=(0.010, 3), decay=(0.090, 3),
                                     decay_std=(0.027, 3), norm=(7.9, 1), dormant=(0.006, 3),
                                     tex=r"0.735 $\pm$ 0.010          & 0.090 $\pm$ 0.027 & 7.9  & 0.006"),
        "sgd": dict(acc=(0.744, 3), acc_std=(0.015, 3), decay=(0.133, 3), decay_std=(0.036, 3),
                    norm=(20.4, 1), dormant=(0.212, 3), tex=r"0.744 $\pm$ 0.015          & 0.133 $\pm$ 0.036 & 20.4 & 0.212"),
        "cbp": dict(acc=(0.728, 3), acc_std=(0.010, 3), decay=(0.147, 3), decay_std=(0.019, 3),
                    norm=(19.1, 1), dormant=(0.225, 3), tex=r"0.728 $\pm$ 0.010          & 0.147 $\pm$ 0.019 & 19.1 & 0.225"),
        "dir_control_c1.0": dict(acc=(0.703, 3), acc_std=(0.007, 3), decay=(0.183, 3), decay_std=(0.040, 3),
                                 norm=(25.1, 1), dormant=(0.287, 3), match_dev=(2.2e-7, None),
                                 tex=r"0.703 $\pm$ 0.007       & 0.183 $\pm$ 0.040 & 25.1 & 0.287"),
        "dir_inward_g0.6": dict(acc=(0.429, 3), acc_std=(0.005, 3), decay=(0.051, 3), decay_std=(0.006, 3),
                                norm=(2.3, 1), dormant=(0.075, 3), match_dev=(2.0e-7, None),
                                tex=r"0.429 $\pm$ 0.005       & 0.051 $\pm$ 0.006 & 2.3  & 0.075"),
    },
    "association": dict(pearson=(0.97, 2), spearman=(0.86, 2), n=7,
                        tex=r"(Pearson $r{=}0.97$, Spearman $\rho{=}0.86$, $n{=}7$ arm-level means)"),
    "regimes": [
        dict(key="e20", label="Permuted MNIST, MLP h=16", seeds=5, arms=7, sgd_decay=(0.133, 3), r=(0.97, 2),
             tex=r"Permuted MNIST, MLP $h{=}16$  & 5 & 7 & {$0.133$} & $+0.97$"),
        dict(key="e22_arch64", label="Permuted MNIST, MLP h=64", seeds=5, arms=7, sgd_decay=(0.052, 3),
             r=(0.45, 2), tex=r"Permuted MNIST, MLP $h{=}64$  & 5 & 7 & {$0.052$} & $+0.45$"),
        dict(key="e24_arch128", label="Permuted MNIST, MLP h=128", seeds=5, arms=7, sgd_decay=(0.032, 3),
             r=(-0.52, 2), tex=r"Permuted MNIST, MLP $h{=}128$ & 5 & 7 & {$0.032$} & $-0.52$"),
        dict(key="e24_rotmnist", label="Rotated MNIST, MLP h=16", seeds=5, arms=7, sgd_decay=(0.021, 3),
             r=(-0.94, 2), tex=r"Rotated MNIST, MLP $h{=}16$   & 5 & 7 & {$0.021$} & $-0.94$"),
        dict(key="e25_cnn", label="Rotated MNIST, small CNN", seeds=3, arms=6, sgd_decay=(0.004, 3),
             r=(-0.82, 2), tex=r"Rotated MNIST, small CNN      & 3 & 6 & {$0.004$} & $-0.82$"),
    ],
    "anr_ablation": dict(revive_acc=(0.761, 3), revive_decay=(0.072, 3), shrink_acc=(0.690, 3),
                         shrink_decay=(0.150, 3), shrink_dormant=(0.362, 3), full_acc=(0.735, 3),
                         tex=r"(revive-only $0.761$, decay $0.072$; the best variant)"),
    "radial": dict(best_acc=(0.763, 3), norms=[(20.4, 1), (4.1, 1), (3.1, 1), (2.3, 1), (0.25, 2), "<0.05"],
                   tex=r"$20.4\!\to\!4.1\!\to\!3.1\!\to\!2.3\!\to\!0.25\!\to\!{<}0.05$"),
    "paired": dict(l2_minus_sgd=(0.074, 3), sp_minus_sgd=(0.055, 3), l2_minus_sp=(0.020, 3),
                   tex=r"L2-init exceeds SGD by $+0.074$, Shrink\&Perturb exceeds SGD by"),
    "cor1": dict(rel_err="3e-8", cos="-6e-11", tex=r"within a relative error of"),
}


def _round_ok(paper_val, decimals, artifact_val) -> bool | None:
    if artifact_val is None:
        return None
    if decimals is None:                     # compare at two significant figures
        return float(f"{artifact_val:.1e}") == float(f"{paper_val:.1e}")
    return round(artifact_val + 0.0, decimals) == round(paper_val, decimals) or \
        abs(artifact_val - paper_val) <= 0.5 * 10 ** (-decimals) + 1e-12


def _cmp(paper_val, decimals, artifact_val) -> dict:
    ok = _round_ok(paper_val, decimals, artifact_val)
    return {"paper": paper_val, "artifact": artifact_val,
            "status": "artifact_consistent" if ok else ("artifact_missing" if ok is None else "mismatch")}


def _m(rec, key, stat="mean"):
    v = (rec or {}).get("metrics", {}).get(key) or {}
    return v.get(stat)


def regime_correlation(group: str) -> dict:
    recs = [r for r in la.index() if r["group"] == group]
    xs = [_m(r, "final_weight_norm") for r in recs]
    ys = [_m(r, "decay") for r in recs]
    return {"group": group, **ev.correlation(xs, ys, labels=[r["arm_label"] for r in recs])}


def _studio_evidence(record_ids: set[str]) -> dict:
    reproduced, generated = [], []
    for rid in reg.list_run_ids():
        m = reg.read_meta(rid)
        if m["status"] != "completed":
            continue
        vs = (m.get("validation") or {})
        if vs.get("status", "").startswith("reproduced") and vs.get("against") in record_ids:
            reproduced.append({"run_id": rid, "against": vs["against"], "seed": vs.get("seed"),
                               "status": vs["status"], "max_abs_diff": vs.get("max_abs_diff")})
        elif vs.get("status") == "mismatch" and vs.get("against") in record_ids:
            reproduced.append({"run_id": rid, "against": vs["against"], "seed": vs.get("seed"),
                               "status": "mismatch", "max_abs_diff": vs.get("max_abs_diff")})
        elif vs.get("status") == "no_legacy_counterpart":
            generated.append(rid)
    return {"reproduced": reproduced, "generated_count": len(generated)}


def findings() -> dict:
    e20 = {r["arm_key"]: r for r in la.index() if r["group"] == "e20"}
    # Finding 1 — association
    names = list(e20)
    xs = [_m(e20[n], "final_weight_norm") for n in names]
    ys = [_m(e20[n], "decay") for n in names]
    corr = ev.correlation(xs, ys, labels=[e20[n]["arm_label"] for n in names])
    per_seed_r = []
    for i in range(5):
        px = [e20[n]["per_seed"][i]["final_weight_norm"] for n in names]
        py = [e20[n]["per_seed"][i]["decay"] for n in names]
        per_seed_r.append(ev.correlation(px, py)["pearson_r"])
    pa = PAPER["association"]
    f1 = {"id": 1, "title": "Weight-norm growth and plasticity decay are associated (width 16)",
          "paper": {"pearson_r": pa["pearson"][0], "spearman_rho": pa["spearman"][0], "n": pa["n"],
                    "tex": pa["tex"]},
          "checks": {"pearson_r": _cmp(pa["pearson"][0], 2, corr["pearson_r"]),
                     "spearman_rho": _cmp(pa["spearman"][0], 2, corr["spearman_rho"]),
                     "n": {"paper": pa["n"], "artifact": corr["n"],
                           "status": "artifact_consistent" if corr["n"] == pa["n"] else "mismatch"}},
          "correlation": corr, "per_seed_r": per_seed_r,
          "leave_one_out": ev.leave_one_out(xs, ys, [e20[n]["arm_label"] for n in names]),
          "points": [{"id": e20[n]["id"], "label": e20[n]["arm_label"], "family": e20[n]["family"],
                      "x": _m(e20[n], "final_weight_norm"), "y": _m(e20[n], "decay")} for n in names],
          "records": [e20[n]["id"] for n in names], "artifact": "results/tables/e20_summary.json"}
    # Finding 2 — identity
    cor1 = la.get("legacy.cor1.orthogonal")
    f2 = {"id": 2, "title": "Fixed-step global-norm identity",
          "statement": "‖θ + ηd‖² = ‖θ‖² + 2η⟨θ,d⟩ + η²  (‖d‖ = 1)",
          "corollaries": ["Cor. 1 — if ⟨θ,d⟩ = 0 the squared norm rises by exactly η².",
                          "Cor. 2 — ‖θ'‖ ≤ ‖θ‖ iff ⟨θ,d⟩ ≤ −η/2, i.e. an inward radial component "
                          "≥ η/(2‖θ‖) of the unit direction."],
          "kind": "mathematical identity (proved); the audit below is a measured check",
          "paper": {"relative_error": PAPER["cor1"]["rel_err"], "cosine": PAPER["cor1"]["cos"]},
          "artifact": ({"relative_error_mean": cor1["metrics"]["cor1_relative_error"]["mean"],
                        "relative_error_max": cor1["cor1"]["rel_err_max"],
                        "radial_cos_mean": cor1["metrics"]["radial_align_mean"]["mean"],
                        "endpoint_delta_mean": cor1["cor1"]["endpoint_delta_mean"],
                        "sum_eta_sq_mean": cor1["cor1"]["sum_eta_sq_mean"],
                        "source": cor1["artifact"]} if cor1 else None),
          "records": [cor1["id"]] if cor1 else []}
    # Finding 3 — direction vs magnitude (Table I)
    rows = []
    for key in TABLE_I_ORDER:
        p, r = PAPER["table1"][key], e20.get(key)
        checks = {"mean_acc": _cmp(p["acc"][0], 3, _m(r, "mean_acc")),
                  "mean_acc_std": _cmp(p["acc_std"][0], 3, _m(r, "mean_acc", "std")),
                  "decay": _cmp(p["decay"][0], 3, _m(r, "decay")),
                  "decay_std": _cmp(p["decay_std"][0], 3, _m(r, "decay", "std")),
                  "final_weight_norm": _cmp(p["norm"][0], 1, _m(r, "final_weight_norm")),
                  "final_dormant": _cmp(p["dormant"][0], 3, _m(r, "final_dormant"))}
        if "match_dev" in p:
            checks["matched_max_dev"] = _cmp(p["match_dev"][0], None, _m(r, "matched_max_dev"))
        rows.append({"key": key, "id": r["id"] if r else None, "label": r["arm_label"] if r else key,
                     "family": r["family"] if r else None, "checks": checks, "tex": p["tex"]})
    raw = {k: e20[k]["per_seed"] for k in ("l2_init", "shrink_pert", "sgd", "adaptive_norm_revive")}
    seedmap = lambda k: {row["seed"]: row["mean_acc"] for row in raw[k]}
    pp = PAPER["paired"]
    paired = {
        "l2_init − sgd": {**ev.paired(seedmap("l2_init"), seedmap("sgd"), "mean_acc"),
                          "paper": pp["l2_minus_sgd"][0]},
        "shrink_pert − sgd": {**ev.paired(seedmap("shrink_pert"), seedmap("sgd"), "mean_acc"),
                              "paper": pp["sp_minus_sgd"][0]},
        "l2_init − shrink_pert": {**ev.paired(seedmap("l2_init"), seedmap("shrink_pert"), "mean_acc"),
                                  "paper": pp["l2_minus_sp"][0]},
        "anr − shrink_pert": {**ev.paired(seedmap("adaptive_norm_revive"), seedmap("shrink_pert"),
                                          "mean_acc"), "paper": -0.064},
    }
    radial = [r for r in la.index() if r["group"] == "e27_radial"]
    f3 = {"id": 3, "title": "Magnitude control beats matched-step directional control",
          "table": rows, "paired_mean_acc": paired,
          "radial": [{"gamma": r["sweep"]["value"], "id": r["id"], "mean_acc": _m(r, "mean_acc"),
                      "final_weight_norm": _m(r, "final_weight_norm"), "decay": _m(r, "decay"),
                      "final_dormant": _m(r, "final_dormant")} for r in radial],
          "radial_paper": {"best_acc": PAPER["radial"]["best_acc"][0],
                           "norms": [n if isinstance(n, str) else n[0] for n in PAPER["radial"]["norms"]]},
          "records": [r["id"] for r in rows if r["id"]] + [r["id"] for r in radial]}
    # Finding 4 — ANR
    anr = {r["arm_key"]: r for r in la.index() if r["group"] == "e22_anr"}
    pa4 = PAPER["anr_ablation"]
    f4 = {"id": 4, "title": "Adaptive-Norm-Revive: dormancy reduction without an accuracy gain",
          "variants": [{"key": k, "id": r["id"], "label": r["arm_label"], "mean_acc": _m(r, "mean_acc"),
                        "decay": _m(r, "decay"), "final_weight_norm": _m(r, "final_weight_norm"),
                        "final_dormant": _m(r, "final_dormant")} for k, r in anr.items()],
          "checks": {"revive_only_acc": _cmp(pa4["revive_acc"][0], 3, _m(anr.get("anr_revive_only"), "mean_acc")),
                     "revive_only_decay": _cmp(pa4["revive_decay"][0], 3, _m(anr.get("anr_revive_only"), "decay")),
                     "shrink_only_acc": _cmp(pa4["shrink_acc"][0], 3, _m(anr.get("anr_shrink_only"), "mean_acc")),
                     "shrink_only_decay": _cmp(pa4["shrink_decay"][0], 3, _m(anr.get("anr_shrink_only"), "decay")),
                     "shrink_only_dormant": _cmp(pa4["shrink_dormant"][0], 3,
                                                 _m(anr.get("anr_shrink_only"), "final_dormant")),
                     "full_acc": _cmp(pa4["full_acc"][0], 3, _m(anr.get("anr_full"), "mean_acc"))},
          "records": [r["id"] for r in anr.values()] + ([e20["adaptive_norm_revive"]["id"]] if "adaptive_norm_revive" in e20 else [])}
    # Finding 5 — regimes
    regimes = []
    for rg in PAPER["regimes"]:
        c = regime_correlation(rg["key"])
        recs = [r for r in la.index() if r["group"] == rg["key"]]
        sgd = next((r for r in recs if r["method"] == "sgd"), None)
        regimes.append({"key": rg["key"], "label": rg["label"], "tex": rg["tex"],
                        "paper": {"r": rg["r"][0], "seeds": rg["seeds"], "arms": rg["arms"],
                                  "sgd_decay": rg["sgd_decay"][0]},
                        "artifact": {"r": c["pearson_r"], "n_arms": c["n"], "excluded": c["excluded"],
                                     "seeds": sgd["n_seeds"] if sgd else None,
                                     "sgd_decay": _m(sgd, "decay"),
                                     "architecture": sgd["architecture"] if sgd else None,
                                     "dataset": sgd["dataset"] if sgd else None,
                                     "source": sgd["artifact"] if sgd else None},
                        "checks": {"r": _cmp(rg["r"][0], 2, c["pearson_r"]),
                                   "n_arms": {"paper": rg["arms"], "artifact": c["n"],
                                              "status": "artifact_consistent" if c["n"] == rg["arms"] else "mismatch"},
                                   "sgd_decay": _cmp(rg["sgd_decay"][0], 3, _m(sgd, "decay"))},
                        "scope_note": c["scope_note"], "records": [r["id"] for r in recs]})
    f5 = {"id": 5, "title": "The norm–decay association is regime-dependent", "regimes": regimes}

    all_ids = set()
    for f in (f1, f2, f3, f4):
        all_ids |= set(f.get("records", []))
    for rg in regimes:
        all_ids |= set(rg["records"])
    studio = _studio_evidence(all_ids)
    out = {"findings": [f1, f2, f3, f4, f5], "studio_evidence": studio,
           "labels": {"paper": "Paper-reported value (paper/main(iconcise).tex)",
                      "artifact": "Recomputed from the released artifact in results/tables/",
                      "reproduced": "Studio run matched a per-seed legacy record",
                      "generated": "Studio run with no legacy counterpart (new evidence)"}}
    n_checks = n_ok = 0
    def walk(o):
        nonlocal n_checks, n_ok
        if isinstance(o, dict):
            if "status" in o and "paper" in o and "artifact" in o:
                n_checks += 1
                n_ok += o["status"] == "artifact_consistent"
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(out["findings"])
    out["consistency"] = {"checks": n_checks, "artifact_consistent": n_ok}
    return out
