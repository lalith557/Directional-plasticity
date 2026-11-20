"""E18 - Scale/mechanism check (Phase 4, D-040).

Runs the E16-style predictive theory analysis and (optionally) the E17-style
adaptive controller on a LARGER SmallCNN (base=64, fc=256, ~4x parameters)
on Split-CIFAR-10 in the non-degenerate regime (NON_DEG).

Mechanistic prediction: if the state-dependent directional effect strengthens
where representation-learning dominates, the E16 slope should be SAME SIGN and
LARGER MAGNITUDE at the bigger CNN scale. If flat/noisier, no scale support.

The controller is also re-run at this scale to see whether the ORIGINAL-goal
Pareto improvement (if any) survives the scale-up.
"""
from __future__ import annotations
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from copy import deepcopy

from configs.preregistration import (
    PRIMARY_SEED, SPLIT_CIFAR, UPDATE_GEN, STATS, PROBE,
    CANONICAL_PROBES, NORMALIZED_FTD, PREDICTIVE_V3, SCALE_MECH, NON_DEG, ADAPTIVE,
)
from src.seeds import seed_everything
from src.models import flat_params, set_flat_params, flat_grad, SmallCNN
from src.device import pick_device, env_summary
from src.datasets import load_cifar10, split_cifar10_tasks
from src.stats import fit_beta4_regression, ptd_caliper_match
from src.probes import build_canonical_probes, probe_grad_align, normalized_ftd
from src.baselines import train_sequence, probe_ftd_generic
from src.adaptive_controller import AdaptiveDirectionalController

from experiments.e04_split_mnist_replication import task_iter, cos
from experiments.e09_split_cifar_replication import (
    gen_sgd_updates as g_sgd_cifar, gen_orth_updates as g_orth_cifar,
    gen_random_updates as g_rand_cifar, gen_convex_mix as g_mix_cifar,
    _reference_grad as _ref_grad_cifar,
)
from experiments.e16_predictive_theory import (
    probe_auc_unified, evaluate_predictive_theory_v3, _stringify,
)
from experiments.e17_adaptive_controller import train_adaptive


RESULTS_TABLES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "tables"
)
RESULTS_FIGS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures"
)


def _warmup_large_cifar(seed, tasks_all, device):
    student = SmallCNN(in_channels=3, base=SCALE_MECH.cnn_base_channels,
                       fc=SCALE_MECH.cnn_hidden_fc, n_classes=10,
                       input_hw=32).to(device)
    for t_idx in range(len(tasks_all) - 1):
        task = tasks_all[t_idx]
        n = task["train_x"].shape[0]
        steps = SCALE_MECH.train_epochs_per_task * max(1, n // 128)
        opt = torch.optim.SGD(student.parameters(), lr=SCALE_MECH.train_lr)
        g = torch.Generator().manual_seed(seed + 17 * t_idx)
        step = 0
        while step < steps:
            for xb_cpu, yb_cpu in task_iter(task["train_x"], task["train_y"], 128, g):
                xb = xb_cpu.to(device); yb = yb_cpu.to(device)
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(student(xb), yb).backward(); opt.step()
                step += 1
                if step >= steps: break
    return student


def run_predictive_at_scale(seed: int, device: torch.device):
    """E16-style predictive analysis on the LARGER CNN."""
    seed_everything(seed)
    cfg = SPLIT_CIFAR
    tx, ty, vx, vy = load_cifar10(cfg.cifar10_dir)
    warmup_pairs = ((0, 1), (2, 3), (4, 5), (6, 7))
    held_out_pair = (8, 9)
    tasks = split_cifar10_tasks(tx, ty, vx, vy, warmup_pairs)
    task_last = split_cifar10_tasks(tx, ty, vx, vy, (held_out_pair,))[0]
    tasks_all = tasks + [task_last]

    student = _warmup_large_cifar(seed, tasks_all, device)
    baseline = deepcopy(student)
    n_params = sum(p.numel() for p in baseline.parameters())
    print(f"[E18 seed={seed}] LARGE SmallCNN params={n_params}")

    tx_all = torch.from_numpy(np.transpose(tx.astype(np.float32) / 255.0, (0, 3, 1, 2)))
    ty_all = torch.from_numpy(ty.astype(np.int64))
    probes = build_canonical_probes(
        tx_all, ty_all, families_and_counts=list(CANONICAL_PROBES.families),
        seed=seed + 10_000, subset_size=cfg.probe_subset, n_classes=10,
        held_out_classes=held_out_pair,
        model_for_anticlass=baseline, device=device,
    )

    xL, yL = task_last["train_x"], task_last["train_y"]
    per = SCALE_MECH.updates_per_source_per_norm
    norms = cfg.target_norms
    a = g_sgd_cifar(baseline, xL, yL, per, norms, cfg.sgd_inner_steps,
                    cfg.sgd_inner_lr, cfg.sgd_inner_batch, seed=seed + 501, device=device)
    b = g_orth_cifar(baseline, xL, yL, per, norms, cfg.sgd_inner_batch,
                     seed=seed + 502, device=device)
    c = g_rand_cifar(baseline, xL, yL, per, norms, cfg.sgd_inner_batch,
                     seed=seed + 503, device=device)
    d = g_mix_cifar(baseline, xL, yL, per, norms, cfg.sgd_inner_steps,
                    cfg.sgd_inner_lr, cfg.sgd_inner_batch, seed=seed + 504, device=device)
    updates = a + b + c + d
    g_ref_train = _ref_grad_cifar(baseline, xL, yL, cfg.sgd_inner_batch,
                                   seed=seed + 991, device=device)

    theta_before_cpu = flat_params(baseline).detach().cpu()
    past_tasks = tasks
    rows = []
    for pi, probe in enumerate(probes):
        align, gp_norm = probe_grad_align(baseline, probe, g_ref_train,
                                           batch=PREDICTIVE_V3.probe_batch,
                                           device=device)
        ref_auc = probe_auc_unified(baseline, probe.x, probe.y,
                                     PREDICTIVE_V3.probe_train_steps,
                                     PREDICTIVE_V3.probe_lr,
                                     PREDICTIVE_V3.probe_batch,
                                     seed=seed, device=device)
        ref_past = []
        for t in past_tasks:
            with torch.no_grad():
                total, n = 0.0, 0
                for i in range(0, t["test_x"].shape[0], 256):
                    xb = t["test_x"][i:i+256].to(device)
                    yb = t["test_y"][i:i+256].to(device)
                    total += F.cross_entropy(baseline(xb), yb, reduction="sum").item()
                    n += xb.shape[0]
                ref_past.append(total / max(n, 1))

        per_update_rows = []
        for uid, (source, tn, delta_cpu) in enumerate(updates):
            m = deepcopy(baseline)
            theta_after = (theta_before_cpu + delta_cpu).to(device)
            set_flat_params(m, theta_after)
            after_auc = probe_auc_unified(m, probe.x, probe.y,
                                           PREDICTIVE_V3.probe_train_steps,
                                           PREDICTIVE_V3.probe_lr,
                                           PREDICTIVE_V3.probe_batch,
                                           seed=seed, device=device)
            nftd = normalized_ftd(after_auc, ref_auc,
                                    eps=NORMALIZED_FTD.fallback_eps)
            after_past = []
            for t in past_tasks:
                with torch.no_grad():
                    total, n = 0.0, 0
                    for i in range(0, t["test_x"].shape[0], 256):
                        xb = t["test_x"][i:i+256].to(device)
                        yb = t["test_y"][i:i+256].to(device)
                        total += F.cross_entropy(m(xb), yb, reduction="sum").item()
                        n += xb.shape[0]
                    after_past.append(total / max(n, 1))
            ptd = float(sum(a2 - r for a2, r in zip(after_past, ref_past)))
            ctb = float(-(g_ref_train @ delta_cpu).item())
            per_update_rows.append({
                "update_id": uid, "source": source, "target_norm": tn,
                "mag": float(delta_cpu.norm().item()),
                "d_dir": cos(delta_cpu, g_ref_train),
                "ctb": ctb, "ptd": ptd,
                "ftd_raw": after_auc - ref_auc, "ftd": nftd,
            })
            del m
        pdf = pd.DataFrame(per_update_rows)
        matched = ptd_caliper_match(pdf, caliper=STATS.ptd_primary_caliper)
        if len(matched) >= 20:
            r = fit_beta4_regression(matched, covariates=list(STATS.covariates),
                                      outcome="ftd", cov_type=STATS.ols_cov_type)
            beta = float(r.coefs["d_dir"])
            ci = (float(r.ci_low["d_dir"]), float(r.ci_high["d_dir"]))
            p = float(r.pvalues["d_dir"])
        else:
            beta, ci, p = None, (None, None), None
        rows.append({
            "probe_id": probe.name, "family": probe.family, "strength": probe.strength,
            "probe_grad_align": align, "probe_grad_norm": gp_norm,
            "align_times_gnorm": align * gp_norm,
            "beta_d_dir_normalized": beta,
            "ci_low": ci[0], "ci_high": ci[1], "p": p,
            "sign": (int(np.sign(beta)) if beta is not None else 0),
            "n_matched": int(len(matched)),
        })
        b_str = (f"{beta:+.5f}" if beta is not None else "None")
        print(f"[E18 seed={seed}] probe {probe.name}: "
              f"align={align:+.3f} beta={b_str}")
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int,
                   default=[PRIMARY_SEED, PRIMARY_SEED + 1])
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device, profile = pick_device(args.device)
    print(f"[E18] device={device} profile={profile}")

    out = {"hardware": env_summary(), "seeds": {}}
    for s in args.seeds:
        print(f"\n=== E18 predictive-at-scale seed={s} ===")
        df = run_predictive_at_scale(s, device=device)
        df.to_csv(os.path.join(RESULTS_TABLES,
                                f"e18_predictive_seed={s}.csv"), index=False)
        pred = evaluate_predictive_theory_v3(df, s, f"cifar10_large_seed={s}")
        out["seeds"][s] = {"per_probe": df.to_dict("records"), "predictive": pred}
        print(f"[E18 seed={s}] slope CI: {pred['slope_bootstrap_ci_95']} "
              f"sign_acc={pred['holdout_sign_accuracy']}")

    # Aggregate across seeds: is the slope consistent + informative?
    slopes = [out["seeds"][s]["predictive"]["train_fit_linear"]["slope"] for s in args.seeds]
    sign_accs = [out["seeds"][s]["predictive"]["holdout_sign_accuracy"] for s in args.seeds]
    overall = {
        "mean_slope": float(np.mean(slopes)),
        "std_slope": float(np.std(slopes)),
        "mean_sign_acc": float(np.mean(sign_accs)),
        "slopes_all_positive": bool(all(sl > 0 for sl in slopes)),
    }
    out["aggregated"] = overall
    verdict = "SCALE_CONFIRMS_DIRECTIONAL_EFFECT" if (overall["slopes_all_positive"]
                                                       and overall["mean_sign_acc"] >= 0.75) else \
              "SCALE_UNCLEAR"
    out["verdict"] = verdict
    print(f"\n[E18] scale/mechanism verdict: {verdict}")
    print(f"  mean slope = {overall['mean_slope']:.4f} ± {overall['std_slope']:.4f}")
    print(f"  mean sign_acc = {overall['mean_sign_acc']:.3f}")

    with open(os.path.join(RESULTS_TABLES, "e18_summary.json"), "w") as f:
        json.dump(_stringify(out), f, indent=2)
    print("[E18] wrote e18_summary.json")


if __name__ == "__main__":
    main()
