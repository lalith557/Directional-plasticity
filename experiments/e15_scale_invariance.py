"""E15 - FTD-scale-invariance investigation (Phase D1).

Pre-registered in decision.md D-030 BEFORE running.

Question: beta_d_dir is nearly constant (~+0.002) across MNIST/CIFAR-10/CIFAR-100
even though raw FTD scale changes ~7x. Is this a real property of the effect, or
an artifact of the permuted-probe construction at fixed step budgets?

We re-analyze the EXISTING v1.1 measurement CSVs (e04b, e09_cifar10, e09_cifar100)
under multiple FTD normalizations. If normalization collapses the three settings'
beta to a common value, the invariance IS a scale artifact. Report either way.

Also examines what FTD looks like as a function of probe budget (from CSV headers).
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scistats

from configs.preregistration import STATS, SCALE_INV
from src.stats import fit_beta4_regression, ptd_caliper_match

RESULTS_TABLES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "tables"
)
RESULTS_FIGS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures"
)


def normalize(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Return df with a new 'ftd_norm' column under the requested normalization."""
    df = df.copy()
    if kind == "raw":
        df["ftd_norm"] = df["ftd"]
    elif kind == "by_ref_ftd":
        # divide by max abs FTD as a proxy for reference scale
        s = df["ftd"].abs().max()
        df["ftd_norm"] = df["ftd"] / max(s, 1e-9)
    elif kind == "by_ftd_std":
        s = df["ftd"].std() + 1e-9
        df["ftd_norm"] = df["ftd"] / s
    elif kind == "by_max_source_ftd":
        # divide by the mean-abs FTD of the SGD-source subset (analog of "reference plasticity")
        sgd_scale = df[df["source"] == "sgd_seed"]["ftd"].abs().mean()
        df["ftd_norm"] = df["ftd"] / max(sgd_scale, 1e-9)
    else:
        raise ValueError(kind)
    return df


def refit(df: pd.DataFrame, outcome: str = "ftd_norm") -> dict:
    matched = ptd_caliper_match(df, caliper=STATS.ptd_primary_caliper)
    if len(matched) < 20: return {"beta": None, "ci": (None, None), "p": None, "n": len(matched)}
    r = fit_beta4_regression(matched, covariates=list(STATS.covariates),
                             outcome=outcome, cov_type=STATS.ols_cov_type)
    return {"beta": float(r.coefs["d_dir"]),
            "ci": (float(r.ci_low["d_dir"]), float(r.ci_high["d_dir"])),
            "p": float(r.pvalues["d_dir"]),
            "n": int(len(matched))}


def main() -> None:
    inputs = {
        "MNIST (E04b)":       "e04b_measurements.csv",
        "CIFAR-10 (E09)":     "e09_cifar10_measurements.csv",
        "CIFAR-100 (E09)":    "e09_cifar100_measurements.csv",
    }

    all_results = {}
    for label, csv_name in inputs.items():
        path = os.path.join(RESULTS_TABLES, csv_name)
        if not os.path.isfile(path):
            print(f"[E15] skip {label}: no CSV")
            continue
        df = pd.read_csv(path)
        row = {}
        for norm_kind in SCALE_INV.normalizations:
            dfn = normalize(df, norm_kind)
            row[norm_kind] = refit(dfn, outcome="ftd_norm")
            print(f"[E15] {label} normalization={norm_kind}: "
                  f"beta={row[norm_kind]['beta']} CI={row[norm_kind]['ci']} "
                  f"n={row[norm_kind]['n']}")
        # summary stats of raw FTD
        row["raw_ftd_stats"] = {
            "mean_abs": float(df["ftd"].abs().mean()),
            "std": float(df["ftd"].std()),
            "sgd_source_mean_abs": float(df[df["source"] == "sgd_seed"]["ftd"].abs().mean()) if "sgd_seed" in df["source"].values else None,
        }
        all_results[label] = row

    # ANALYSIS: does normalization collapse the three settings?
    summary = {"normalizations": {}, "raw_stats": {k: v["raw_ftd_stats"]
                                                     for k, v in all_results.items()}}
    for norm_kind in SCALE_INV.normalizations:
        betas = [v[norm_kind]["beta"] for v in all_results.values()
                 if v[norm_kind]["beta"] is not None]
        if not betas: continue
        rng = max(betas) - min(betas)
        rng_ratio = rng / max(abs(np.mean(betas)), 1e-9)
        summary["normalizations"][norm_kind] = {
            "betas_by_dataset": {k: all_results[k][norm_kind]["beta"] for k in all_results},
            "min": min(betas), "max": max(betas), "range": rng,
            "range_over_mean_ratio": rng_ratio,
        }

    # Plot: beta_d_dir vs normalization, per dataset
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(SCALE_INV.normalizations))
    width = 0.25
    ds_list = list(all_results.keys())
    for i, ds in enumerate(ds_list):
        vals = [all_results[ds][n]["beta"] if all_results[ds][n]["beta"] is not None else 0
                for n in SCALE_INV.normalizations]
        ax.bar(x + (i - len(ds_list)/2 + 0.5) * width, vals, width, label=ds)
    ax.set_xticks(x); ax.set_xticklabels(SCALE_INV.normalizations, rotation=15)
    ax.set_ylabel("beta_d_dir (matched c=0.02)")
    ax.set_title("E15: beta_d_dir under different FTD normalizations")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_FIGS, "e15_scale_invariance.png"), dpi=140)
    plt.close(fig)

    # Interpretation
    # If under normalization "by_ftd_std" the three settings collapse to the same beta,
    # the scale-invariance is an artifact of FTD-scale differences.
    raw_max = summary["raw_stats"]["MNIST (E04b)"]["mean_abs"] if "MNIST (E04b)" in summary["raw_stats"] else 0
    interpretation = {
        "raw_ftd_ranges_across_settings": (
            f"MNIST mean|FTD|={summary['raw_stats'].get('MNIST (E04b)', {}).get('mean_abs')} "
            f"CIFAR-10={summary['raw_stats'].get('CIFAR-10 (E09)', {}).get('mean_abs')} "
            f"CIFAR-100={summary['raw_stats'].get('CIFAR-100 (E09)', {}).get('mean_abs')}"
        ),
        "beta_by_setting_across_normalizations": summary["normalizations"],
    }

    with open(os.path.join(RESULTS_TABLES, "e15_summary.json"), "w") as f:
        json.dump({"per_dataset": all_results,
                   "summary": summary,
                   "interpretation": interpretation}, f, indent=2, default=str)
    print("[E15] wrote e15_summary.json")


if __name__ == "__main__":
    main()
