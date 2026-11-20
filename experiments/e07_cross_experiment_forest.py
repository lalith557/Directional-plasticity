"""E07 - Cross-experiment forest plot summarizing beta_d_dir across all runs.

Reads the summary JSONs from E03, E03b, E03c, E04, and (if present) E06/N2, and
produces a single figure showing point estimates + 95% CIs for beta_d_dir under
the primary caliper (0.02) across regimes.
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "results", "tables")
FIGS = os.path.join(ROOT, "results", "figures")


def pick_caliper(summary: dict, target: float = 0.02):
    for c in summary.get("control_c_sweep", []):
        if c.get("caliper") == target and c.get("beta_d_dir") is not None:
            return c
    return None


def load(name):
    p = os.path.join(TABLES, f"{name}_summary.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def main() -> None:
    entries = []
    for tag, label in [
        ("e03",                       "E03   synthetic (pre-reg 3-source)"),
        ("e03b",                      "E03b  synthetic + convex_mix"),
        ("e03c",                      "E03c  synthetic (hard regime)"),
        ("e04",                       "E04   Split-MNIST replication"),
        ("e04b",                      "E04b  Split-MNIST bridged (VIF fix)"),
        ("e09_cifar10",               "E09   Split-CIFAR-10 (permuted probe)"),
        ("e09_cifar100",              "E09   Split-CIFAR-100 (permuted probe)"),
        ("e09_cifar10_novel_class",   "E09   Split-CIFAR-10 (novel_class probe)"),
    ]:
        s = load(tag)
        if s is None: continue
        cal = pick_caliper(s, 0.02)
        if cal is None: continue
        entries.append({
            "label": label, "beta": cal["beta_d_dir"],
            "lo": cal["ci_low"], "hi": cal["ci_high"],
            "n": cal["n_matched"], "sig": cal["excludes_zero"],
        })

    # Optional: per-seed betas from E06/N2
    n2 = None
    e06 = load("e06")
    if e06 is not None:
        n2 = e06.get("N2_signature_stability", {}).get("per_seed", {})
        for seed, v in (n2 or {}).items():
            entries.append({
                "label": f"E06/N2 seed={seed} (MNIST)",
                "beta": v["beta_d_dir_g"],
                "lo": v["ci"][0], "hi": v["ci"][1],
                "n": v["n_matched"], "sig": v["excludes_zero"],
            })

    # E10 cells (probe-family x budget) - crucial to show the sign flip
    e10 = load("e10")
    if e10 is not None:
        for cell in e10.get("cells", []):
            m = cell.get("matched")
            if m is None: continue
            entries.append({
                "label": f"E10  {cell['family']}/{cell['budget']}",
                "beta": m["beta_d_dir"],
                "lo": m["ci_low"], "hi": m["ci_high"],
                "n": cell["n_matched"], "sig": m["excludes_zero"],
            })

    # E11 per-seed Fisher-axis betas
    e11 = load("e11")
    if e11 is not None:
        for r in e11.get("per_seed", []):
            if r.get("beta_d_dir_fisher_top1") is None: continue
            entries.append({
                "label": f"E11  Fisher-top1 seed={r['seed']}",
                "beta": r["beta_d_dir_fisher_top1"],
                "lo": r["ci_low"], "hi": r["ci_high"],
                "n": r["n_matched"], "sig": r["excludes_zero"],
            })

    if not entries:
        print("[E07] no summaries found - run experiments first")
        return

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(entries) + 2))
    ys = list(range(len(entries)))[::-1]
    for y, e in zip(ys, entries):
        ax.errorbar(e["beta"], y, xerr=[[e["beta"] - e["lo"]], [e["hi"] - e["beta"]]],
                    fmt=("s" if e["sig"] else "o"),
                    color=("firebrick" if e["sig"] else "gray"),
                    capsize=3, markersize=6)
        ax.text(max(e["hi"], 0) + 0.001, y,
                f"  n={e['n']}", va="center", fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([e["label"] for e in entries], fontsize=9)
    ax.set_xlabel("beta_d_dir (95% CI) at PTD caliper = 0.02")
    ax.set_title("E07 - Cross-experiment forest plot of beta_d_dir")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGS, "e07_cross_experiment_forest.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[E07] wrote {out}")


if __name__ == "__main__":
    main()
