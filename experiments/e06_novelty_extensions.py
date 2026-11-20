"""E06 - Novelty extensions beyond the pre-registered spec.

Three sub-experiments, each cheap and modular:

  N1 (multi-descriptor regression): Regress FTD on a *vector* of directional
      descriptors instead of a single scalar. Descriptors:
        d_dir_g     = cos(delta, g_ref)  [the E04 primary]
        d_dir_task  = cos(delta, current-task grad, freshly sampled)
        d_dir_rand  = cos(delta, a fixed random reference)
        d_dir_past  = cos(delta, sum of gradients on past tasks)
      Which descriptor(s) survive the joint regression under caliper matching?

  N2 (directional signature stability): repeat E04-style β4 estimation over
      several seeds and record whether the *sign* and magnitude of β4 are
      stable, and whether the top harmful direction u_bad is stable across
      seeds (measured by pairwise cos similarity).

  N3 (empirical-u_bad method rerun): estimate u_bad from the actual FTD-worst
      updates (top PCA of FTD-weighted deltas) and rerun a small projection
      arm to see if this specific direction, unlike +g_ref, gives Candidate A
      any purchase.

All three consume the same MNIST warmup + update-generation code from E04, so
runtime stays modest on CPU.
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from configs.preregistration import (
    PRIMARY_SEED, SPLIT_MNIST, UPDATE_GEN, PROBE, STATS, KILL, BOOTSTRAP_SEEDS,
)
from src.seeds import seed_everything
from src.models import flat_params, set_flat_params, flat_grad
from src.datasets import load_mnist, split_mnist_tasks
from src.stats import compute_vif, fit_beta4_regression, ptd_caliper_match
from src.constraint import project_out, project_out_multi
# reuse E04 helpers via import
from experiments.e04_split_mnist_replication import (
    MNISTStudent, task_iter, eval_loss, build_probes, probe_ftd,
    gen_sgd_updates, gen_orth_updates, gen_random_sphere_updates,
    gen_convex_mix_updates, _reference_gradient_mnist, cos,
)


RESULTS_TABLES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "tables"
)
RESULTS_FIGS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures"
)


def _stringify(obj):
    if isinstance(obj, dict): return {k: _stringify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_stringify(v) for v in obj]
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.integer): return int(obj)
    return obj


def _load_and_split_mnist():
    tx, ty, vx, vy = load_mnist(SPLIT_MNIST.mnist_dir)
    tasks = split_mnist_tasks(tx, ty, vx, vy, SPLIT_MNIST.task_class_pairs)
    for task in tasks:
        pair = task["pair"]
        mask_tr = np.isin(ty, pair); mask_te = np.isin(vy, pair)
        task["train_y"] = torch.from_numpy(ty[mask_tr].astype(np.int64))
        task["test_y"] = torch.from_numpy(vy[mask_te].astype(np.int64))
    tx_norm = torch.from_numpy(tx.reshape(len(tx), -1).astype(np.float32) / 255.0)
    ty_all = torch.from_numpy(ty.astype(np.int64))
    return tasks, tx_norm, ty_all


def _warmup_student(seed, tasks, n_warmup=4):
    student = MNISTStudent(input_dim=784,
                           hidden=SPLIT_MNIST.student_hidden_width,
                           depth=SPLIT_MNIST.student_hidden_depth,
                           n_out=10)
    for t_idx in range(n_warmup):
        task = tasks[t_idx]
        n = task["train_x"].shape[0]
        steps = SPLIT_MNIST.train_epochs_per_task * max(1, n // SPLIT_MNIST.train_batch)
        g = torch.Generator().manual_seed(seed + 17 * t_idx)
        opt = torch.optim.SGD(student.parameters(), lr=SPLIT_MNIST.train_lr)
        step = 0
        while step < steps:
            for xb, yb in task_iter(task["train_x"], task["train_y"],
                                    SPLIT_MNIST.train_batch, g):
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(student(xb), yb).backward(); opt.step()
                step += 1
                if step >= steps: break
    return student


def _past_gradient(student, past_tasks, seed, batch):
    """Sum of per-task gradients across past tasks (proxy for past-Fisher direction)."""
    m = deepcopy(student)
    accum = torch.zeros(sum(p.numel() for p in m.parameters()))
    for i, task in enumerate(past_tasks):
        g = torch.Generator().manual_seed(seed + 331 * (i + 1))
        xb, yb = next(task_iter(task["train_x"], task["train_y"], batch, g))
        m.zero_grad(set_to_none=True)
        F.cross_entropy(m(xb), yb).backward()
        accum += flat_grad(m)
    return accum


def _make_update_frame_with_multi(seed, per_source=15):
    """Generate updates and compute multi-descriptor cosines. Returns df + delta list."""
    seed_everything(seed)
    tasks, tx_norm, ty_all = _load_and_split_mnist()
    student = _warmup_student(seed, tasks, n_warmup=4)
    baseline = deepcopy(student)
    task4 = tasks[4]
    x4, y4 = task4["train_x"], task4["train_y"]
    norms = UPDATE_GEN.target_norms

    a = gen_sgd_updates(baseline, x4, y4, per_source, norms,
                        UPDATE_GEN.sgd_inner_steps, UPDATE_GEN.sgd_inner_lr,
                        UPDATE_GEN.sgd_inner_batch, seed + 501)
    b = gen_orth_updates(baseline, x4, y4, per_source, norms,
                         UPDATE_GEN.sgd_inner_batch, seed + 502)
    c = gen_random_sphere_updates(baseline, x4, y4, per_source, norms,
                                  UPDATE_GEN.sgd_inner_batch, seed + 503)
    d = gen_convex_mix_updates(baseline, x4, y4, per_source, norms,
                               UPDATE_GEN.sgd_inner_steps, UPDATE_GEN.sgd_inner_lr,
                               UPDATE_GEN.sgd_inner_batch, seed + 504)
    updates = a + b + c + d

    g_ref = _reference_gradient_mnist(baseline, x4, y4,
                                      UPDATE_GEN.sgd_inner_batch, seed + 991)
    # Fresh task-grad from a different batch of task 4 (structurally same direction
    # but a different seed - captures "current task gradient" variability)
    g_task = _reference_gradient_mnist(baseline, x4, y4,
                                       UPDATE_GEN.sgd_inner_batch, seed + 12345)
    g_past = _past_gradient(baseline, tasks[:4], seed + 22222,
                            batch=UPDATE_GEN.sgd_inner_batch)
    g_rand = torch.randn_like(g_ref)

    probes = build_probes(tx_norm, ty_all, PROBE.n_probes,
                          seed=seed + 10_000, subset_size=3000)
    ref_ftd = probe_ftd(baseline, probes, PROBE.probe_steps, PROBE.probe_lr,
                        PROBE.probe_batch, seed=seed)
    past_tasks = tasks[:4]
    ref_past = [eval_loss(baseline, t["test_x"], t["test_y"]) for t in past_tasks]

    theta_before = flat_params(baseline)
    rows, deltas_saved = [], []
    for uid, u in enumerate(updates):
        m = deepcopy(baseline)
        set_flat_params(m, theta_before + u.delta)
        ftd_after = probe_ftd(m, probes, PROBE.probe_steps, PROBE.probe_lr,
                              PROBE.probe_batch, seed=seed)
        ftd = ftd_after - ref_ftd
        after_past = [eval_loss(m, t["test_x"], t["test_y"]) for t in past_tasks]
        ptd = float(sum(a - r for a, r in zip(after_past, ref_past)))
        rows.append({
            "uid": uid, "source": u.source, "mag": float(u.delta.norm().item()),
            "d_dir_g":    cos(u.delta, g_ref),
            "d_dir_task": cos(u.delta, g_task),
            "d_dir_past": cos(u.delta, g_past),
            "d_dir_rand": cos(u.delta, g_rand),
            "ctb": float(u.ctb), "ptd": ptd, "ftd": ftd,
        })
        deltas_saved.append(u.delta.detach().clone())
    df = pd.DataFrame(rows)
    return df, deltas_saved, {"g_ref": g_ref, "g_task": g_task,
                               "g_past": g_past, "g_rand": g_rand,
                               "student": baseline, "probes": probes}


# ============ N1: multi-descriptor regression ============

def experiment_n1():
    print("\n=== N1: multi-descriptor beta4 regression ===")
    df, _, _ = _make_update_frame_with_multi(seed=PRIMARY_SEED, per_source=15)
    df.to_csv(os.path.join(RESULTS_TABLES, "e06_n1_measurements.csv"), index=False)

    covariates = ["mag", "ctb", "ptd", "d_dir_g", "d_dir_task", "d_dir_past", "d_dir_rand"]
    vif = compute_vif(df, covariates)
    print(f"[N1] VIF: {vif}")
    r = fit_beta4_regression(df, covariates=covariates, outcome="ftd")
    print(f"[N1] Full regression n={r.n_obs}  R^2={r.r_squared:.4f}")
    for cov in covariates:
        print(f"       beta_{cov} = {r.coefs[cov]:+.5f}  "
              f"CI=[{r.ci_low[cov]:+.5f},{r.ci_high[cov]:+.5f}]  "
              f"p={r.pvalues[cov]:.4g}")

    # Caliper-matched multi-descriptor regression
    matched_results = {}
    for cal in STATS.ptd_calipers:
        matched = ptd_caliper_match(df, caliper=cal)
        if len(matched) < 30:
            continue
        rr = fit_beta4_regression(matched, covariates=covariates, outcome="ftd")
        matched_results[cal] = {
            "n": int(len(matched)),
            "coefs": {k: float(rr.coefs[k]) for k in covariates},
            "ci_low": {k: float(rr.ci_low[k]) for k in covariates},
            "ci_high": {k: float(rr.ci_high[k]) for k in covariates},
            "p": {k: float(rr.pvalues[k]) for k in covariates},
            "excludes_zero": {k: bool((rr.ci_low[k] > 0) or (rr.ci_high[k] < 0))
                              for k in covariates},
        }
        print(f"[N1] caliper={cal}, n={len(matched)}:")
        for cov in covariates:
            excl = (rr.ci_low[cov] > 0) or (rr.ci_high[cov] < 0)
            marker = "*" if excl else " "
            print(f"       {marker} beta_{cov}={rr.coefs[cov]:+.5f} "
                  f"CI=[{rr.ci_low[cov]:+.5f},{rr.ci_high[cov]:+.5f}] p={rr.pvalues[cov]:.3g}")

    return {"vif": _stringify(vif),
            "full": {"coefs": _stringify(r.coefs), "ci_low": _stringify(r.ci_low),
                     "ci_high": _stringify(r.ci_high), "p": _stringify(r.pvalues),
                     "r_squared": r.r_squared},
            "matched_calipers": matched_results}


# ============ N2: directional signature stability ============

def experiment_n2():
    print("\n=== N2: directional-signature stability across seeds ===")
    per_seed_results = {}
    u_bad_by_seed = {}
    for seed in BOOTSTRAP_SEEDS:
        print(f"\n[N2] seed={seed}")
        df, deltas, ctx = _make_update_frame_with_multi(seed=seed, per_source=10)
        # β_d_dir_g regression under PTD caliper 0.02 (primary)
        matched = ptd_caliper_match(df, caliper=STATS.ptd_primary_caliper)
        if len(matched) < 20:
            print(f"[N2] seed {seed}: insufficient matched samples")
            continue
        r = fit_beta4_regression(
            matched, covariates=["mag", "ctb", "ptd", "d_dir_g"], outcome="ftd",
        )
        beta = float(r.coefs["d_dir_g"])
        # Estimate top FTD-weighted harmful direction (empirical u_bad)
        ftds = df["ftd"].values.tolist()
        from src.constraint import estimate_harmful_direction
        u_bad = estimate_harmful_direction(deltas, ftds, top_k=1)
        u_bad_by_seed[seed] = u_bad
        # cos with g_ref
        u_bad_vs_gref = float((u_bad @ ctx["g_ref"]).item()
                              / (u_bad.norm() * ctx["g_ref"].norm() + 1e-12))
        per_seed_results[seed] = {
            "n_matched": int(len(matched)),
            "beta_d_dir_g": beta,
            "ci": (float(r.ci_low["d_dir_g"]), float(r.ci_high["d_dir_g"])),
            "p": float(r.pvalues["d_dir_g"]),
            "excludes_zero": bool((r.ci_low["d_dir_g"] > 0) or (r.ci_high["d_dir_g"] < 0)),
            "u_bad_vs_gref_cos": u_bad_vs_gref,
        }
        print(f"[N2] seed {seed}: beta_d_dir_g={beta:+.5f} "
              f"CI=[{r.ci_low['d_dir_g']:+.5f},{r.ci_high['d_dir_g']:+.5f}] "
              f"u_bad vs g_ref cos={u_bad_vs_gref:+.4f}")

    # Pairwise cosine of u_bad across seeds
    seeds = list(u_bad_by_seed.keys())
    pair_cos = {}
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            si, sj = seeds[i], seeds[j]
            ui, uj = u_bad_by_seed[si], u_bad_by_seed[sj]
            c = float((ui @ uj / (ui.norm() * uj.norm() + 1e-12)).item())
            pair_cos[f"{si}_vs_{sj}"] = c
            print(f"[N2] cos(u_bad_{si}, u_bad_{sj}) = {c:+.4f}")

    # Sign consistency across seeds
    betas = [v["beta_d_dir_g"] for v in per_seed_results.values()]
    excludes_all = [v["excludes_zero"] for v in per_seed_results.values()]
    sign_stable = (all(b > 0 for b in betas) or all(b < 0 for b in betas)) if betas else False
    print(f"[N2] sign_stable_across_seeds={sign_stable}")
    print(f"[N2] fraction_seeds_significant={sum(excludes_all)}/{len(excludes_all)}")
    return {"per_seed": per_seed_results, "pairwise_u_bad_cos": pair_cos,
            "sign_stable": bool(sign_stable),
            "frac_seeds_significant": float(sum(excludes_all) / max(len(excludes_all), 1))}


# ============ N3: empirical-u_bad method rerun ============

def experiment_n3():
    print("\n=== N3: rerun projection method with empirical top harmful direction ===")
    df, deltas, ctx = _make_update_frame_with_multi(seed=PRIMARY_SEED, per_source=15)
    from src.constraint import estimate_harmful_direction
    ftds = df["ftd"].values.tolist()
    u_bad_emp = estimate_harmful_direction(deltas, ftds, top_k=1)
    u_bad_vs_gref = float((u_bad_emp @ ctx["g_ref"]
                           / (u_bad_emp.norm() * ctx["g_ref"].norm() + 1e-12)).item())
    print(f"[N3] empirical u_bad vs g_ref cos = {u_bad_vs_gref:+.4f}")

    # Run 3-arm compare: A0 SGD, A1 project +g_ref (E05 baseline), A3 project u_bad_emp
    tasks, tx_norm, ty_all = _load_and_split_mnist()
    probes = build_probes(tx_norm, ty_all, PROBE.n_probes,
                          seed=PRIMARY_SEED + 10_000, subset_size=3000)
    from experiments.e05_candidate_a_method import train_arm

    def fresh_student():
        seed_everything(PRIMARY_SEED + 7000)
        m = MNISTStudent(input_dim=784,
                         hidden=SPLIT_MNIST.student_hidden_width,
                         depth=SPLIT_MNIST.student_hidden_depth,
                         n_out=10)
        return m

    arm_summaries = {}
    for name, u in [("A0_sgd", None),
                    ("A1_project_gref", ctx["g_ref"] / (ctx["g_ref"].norm() + 1e-12)),
                    ("A3_project_uemp", u_bad_emp)]:
        model = fresh_student()
        result = train_arm(
            model, tasks, lr=SPLIT_MNIST.train_lr,
            epochs=SPLIT_MNIST.train_epochs_per_task,
            batch=SPLIT_MNIST.train_batch, seed=PRIMARY_SEED,
            projection_mode=("none" if u is None else "fixed"),
            u_bad=u, probes=probes, probe_cfg=PROBE, verbose=False,
        )
        arm_summaries[name] = {
            "mean_final_acc": result["mean_final_acc"],
            "final_accs": result["final_accs_all_tasks"],
            "final_probe_ftd": result["per_task_probe_ftd"][-1],
        }
        print(f"[N3] {name}: mean_final_acc={result['mean_final_acc']:.4f}  "
              f"final_probeFTD={result['per_task_probe_ftd'][-1]:.4f}")

    return {"u_bad_vs_gref_cos": u_bad_vs_gref, "arms": _stringify(arm_summaries)}


def main() -> None:
    seed_everything(PRIMARY_SEED)
    out = {}

    out["N1_multi_descriptor"] = experiment_n1()
    out["N2_signature_stability"] = experiment_n2()
    out["N3_empirical_method"] = experiment_n3()

    out = _stringify(out)
    with open(os.path.join(RESULTS_TABLES, "e06_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[E06] wrote {os.path.join(RESULTS_TABLES, 'e06_summary.json')}")


if __name__ == "__main__":
    main()
