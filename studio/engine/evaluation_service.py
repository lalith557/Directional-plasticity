"""Statistics with explicit scope rules.

* Correlations across methods are labelled as *method-level* (n = number of method
  means), never as seed-level observations.
* Confidence intervals for r are only reported when n ≥ 10 (Fisher-z needs a
  reasonable sample; the paper's regimes have n = 6–7, so none is shown).
* Paired comparisons require identical protocol (everything except the update rule)
  and shared seeds; otherwise they are refused with the reason.
* Descriptive differences, test statistics and interpretation are separate fields.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy import stats

MIN_N_FOR_CI = 10


def _clean_pairs(xs: Iterable, ys: Iterable):
    pts = [(float(x), float(y)) for x, y in zip(xs, ys)
           if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y))]
    return pts


def correlation(xs, ys, labels=None, level: str = "method") -> dict:
    """Pearson + Spearman over finite pairs; excluded points are reported, not dropped silently."""
    labels = list(labels) if labels is not None else [str(i) for i in range(len(list(xs)))]
    xs, ys = list(xs), list(ys)
    kept, excluded = [], []
    for lab, x, y in zip(labels, xs, ys):
        ok = x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y))
        (kept if ok else excluded).append(lab)
    pts = _clean_pairs(xs, ys)
    n = len(pts)
    out = {"n": n, "level": level, "included": kept, "excluded": excluded,
           "pearson_r": None, "pearson_p": None, "spearman_rho": None, "spearman_p": None,
           "fit": None, "ci95": None,
           "ci_note": (f"No confidence interval: n = {n} {level}-level points (< {MIN_N_FOR_CI})."
                       if n < MIN_N_FOR_CI else None),
           "scope_note": (f"Computed across {n} method-level means; these are not {n} independent "
                          "seed-level observations and do not establish causation."
                          if level == "method" else f"Computed across {n} seed-level points.")}
    if n < 3:
        out["ci_note"] = "Fewer than 3 finite points; no correlation computed."
        return out
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    if np.std(x) == 0 or np.std(y) == 0:
        out["ci_note"] = "A variable is constant; correlation undefined."
        return out
    pr = stats.pearsonr(x, y); sr = stats.spearmanr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    out.update(pearson_r=float(pr[0]), pearson_p=float(pr[1]), spearman_rho=float(sr[0]),
               spearman_p=float(sr[1]), fit={"slope": float(slope), "intercept": float(intercept)})
    if n >= MIN_N_FOR_CI and abs(pr[0]) < 1:
        z, se = np.arctanh(pr[0]), 1 / math.sqrt(n - 3)
        out["ci95"] = [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]
        out["ci_note"] = "Fisher-z approximate 95% CI (assumes bivariate normality)."
    return out


def leave_one_out(xs, ys, labels) -> list[dict]:
    out = []
    for i, lab in enumerate(labels):
        xx = [x for j, x in enumerate(xs) if j != i]
        yy = [y for j, y in enumerate(ys) if j != i]
        c = correlation(xx, yy)
        out.append({"dropped": lab, "pearson_r": c["pearson_r"]})
    return out


def aggregate(values: list) -> dict:
    """Mean ± population std (ddof = 0, matching np.std in the experiment scripts)."""
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not v:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}


def aggregate_curves(curves: list[list]) -> dict:
    """Per-task mean/std across seeds (only over runs that reached each task)."""
    if not curves:
        return {"mean": [], "std": [], "n": []}
    T = max(len(c) for c in curves)
    mean, std, n = [], [], []
    for t in range(T):
        vals = [c[t] for c in curves if t < len(c) and c[t] is not None and math.isfinite(c[t])]
        mean.append(float(np.mean(vals)) if vals else None)
        std.append(float(np.std(vals)) if vals else None)
        n.append(len(vals))
    return {"mean": mean, "std": std, "n": n}


def sign_test_p(wins: int, losses: int) -> float | None:
    """Exact two-sided binomial sign test (ties excluded)."""
    m = wins + losses
    if m == 0:
        return None
    return float(stats.binomtest(wins, m, 0.5, alternative="two-sided").pvalue)


def paired(a_rows: dict[int, float], b_rows: dict[int, float], metric: str,
           higher_is_better: bool = True) -> dict:
    """Paired differences a − b over shared seeds (callers must check protocol equality)."""
    seeds = sorted(set(a_rows) & set(b_rows))
    seeds = [s for s in seeds if a_rows[s] is not None and b_rows[s] is not None]
    if not seeds:
        return {"ok": False, "reason": "no shared seeds with finite values"}
    d = [a_rows[s] - b_rows[s] for s in seeds]
    better = [(x > 0) if higher_is_better else (x < 0) for x in d]
    wins = sum(better)
    losses = sum(1 for x, b in zip(d, better) if not b and x != 0)
    out = {"ok": True, "metric": metric, "seeds": seeds, "n": len(seeds), "differences": d,
           "descriptive": {"mean_difference": float(np.mean(d)), "std_difference": float(np.std(d)),
                           "a_better_on": wins, "of": len(seeds)},
           "tests": {"sign_test_p_two_sided": sign_test_p(wins, losses)},
           "interpretation_note": ("Descriptive paired differences. With n = "
                                   f"{len(seeds)} seeds the exact sign test cannot reach p < 0.05 "
                                   "unless n ≥ 6; no significance claim is made."
                                   if len(seeds) < 6 else
                                   "Sign test reported; significance is not asserted automatically.")}
    if len(seeds) >= 3 and np.std(d) > 0:
        t = stats.ttest_rel([a_rows[s] for s in seeds], [b_rows[s] for s in seeds])
        out["tests"]["paired_t_p_two_sided"] = float(t.pvalue)
        out["tests"]["paired_t_note"] = "Assumes normally distributed differences; n is small."
    return out


HIGHER_IS_BETTER = {"mean_acc": True, "peak_acc": True, "end_acc": True, "decay": False,
                    "final_weight_norm": False, "final_dormant": False, "final_eff_rank": True,
                    "matched_max_dev": False}
