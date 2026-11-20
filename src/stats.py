"""β4 regression, VIF gating, bootstrap CI, and Control C caliper matching."""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class RegressionResult:
    coefs: dict
    stderr: dict
    ci_low: dict
    ci_high: dict
    pvalues: dict
    vif: dict
    n_obs: int
    r_squared: float

    def summary_dict(self) -> dict:
        return {
            "n_obs": self.n_obs,
            "r_squared": self.r_squared,
            "coefs": self.coefs,
            "stderr": self.stderr,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "pvalues": self.pvalues,
            "vif": self.vif,
        }


def compute_vif(df: pd.DataFrame, covariates: List[str]) -> dict:
    X = sm.add_constant(df[covariates].values.astype(float))
    return {
        cov: float(variance_inflation_factor(X, i + 1))
        for i, cov in enumerate(covariates)
    }


def fit_beta4_regression(
    df: pd.DataFrame,
    covariates: List[str] = ("mag", "ctb", "ptd", "d_dir"),
    outcome: str = "ftd",
    cov_type: str = "HC3",
    alpha: float = 0.05,
) -> RegressionResult:
    X = sm.add_constant(df[list(covariates)].values.astype(float))
    y = df[outcome].values.astype(float)
    model = sm.OLS(y, X).fit(cov_type=cov_type)
    names = ["const"] + list(covariates)
    ci = model.conf_int(alpha=alpha)
    coefs = {n: float(v) for n, v in zip(names, model.params)}
    se = {n: float(v) for n, v in zip(names, model.bse)}
    ci_low = {n: float(ci[i][0]) for i, n in enumerate(names)}
    ci_high = {n: float(ci[i][1]) for i, n in enumerate(names)}
    pvals = {n: float(v) for n, v in zip(names, model.pvalues)}
    vif = compute_vif(df, list(covariates))
    return RegressionResult(
        coefs=coefs, stderr=se, ci_low=ci_low, ci_high=ci_high,
        pvalues=pvals, vif=vif, n_obs=int(len(y)),
        r_squared=float(model.rsquared),
    )


def bootstrap_beta4(
    df: pd.DataFrame,
    covariates: List[str] = ("mag", "ctb", "ptd", "d_dir"),
    outcome: str = "ftd",
    n_iter: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float, float]:
    """Return (mean, ci_low, ci_high, sign_flip_fraction) for β_{d_dir}."""
    rng = np.random.default_rng(seed)
    n = len(df)
    dir_idx = list(covariates).index("d_dir")
    betas = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        sub = df.iloc[idx]
        try:
            r = fit_beta4_regression(sub, covariates=covariates, outcome=outcome)
            betas[i] = r.coefs["d_dir"]
        except Exception:
            betas[i] = np.nan
    betas = betas[~np.isnan(betas)]
    lo = float(np.quantile(betas, alpha / 2))
    hi = float(np.quantile(betas, 1 - alpha / 2))
    mean = float(betas.mean())
    # sign-flip fraction: fraction of samples opposite sign from full-sample mean
    sf = float(np.mean(np.sign(betas) != np.sign(mean)))
    return mean, lo, hi, sf


def ptd_caliper_match(df: pd.DataFrame, caliper: float, ptd_col: str = "ptd") -> pd.DataFrame:
    """Bin by PTD calipers, then within-bin equalize the count per `source`.

    We drop any bin where any source has zero samples; remaining bins are
    truncated to the min-source count so PTD distribution is balanced across
    the source groups.
    """
    if len(df) == 0:
        return df.copy()
    df = df.copy()
    df["_bin"] = (df[ptd_col] / caliper).round().astype(int)
    kept = []
    for _, grp in df.groupby("_bin"):
        counts = grp.groupby("source").size()
        if len(counts) < grp["source"].nunique():
            continue
        min_n = counts.min()
        if min_n < 1:
            continue
        for _, sg in grp.groupby("source"):
            kept.append(sg.iloc[:min_n])
    if not kept:
        return df.head(0).drop(columns=["_bin"])
    matched = pd.concat(kept, axis=0).drop(columns=["_bin"])
    return matched.reset_index(drop=True)


def within_vs_between_caliper_variance(
    df: pd.DataFrame, caliper: float, col: str = "ptd",
) -> float:
    """Return within-bin variance divided by total variance (Control C diagnostic).

    Low value (near 0) means calipers really did group similar PTDs; high value
    means the calipers are wider than the actual PTD spread, which means matching
    isn't matching much.
    """
    d = df.copy()
    d["_bin"] = (d[col] / caliper).round().astype(int)
    total_var = float(d[col].var())
    if total_var == 0:
        return 0.0
    within = float(
        d.groupby("_bin")[col].var(ddof=0).fillna(0).mul(d.groupby("_bin").size()).sum()
        / len(d)
    )
    return within / total_var
