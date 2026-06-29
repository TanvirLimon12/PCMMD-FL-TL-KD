"""
train_fedprox.py
----------------
FedProx simulation (Tanjid §T5). Proximal term mu/2 * ||w - w_global||^2 stabilises
non-IID client drift. Sweeps mu over config 'mu_values' (default [0.001, 0.01, 0.1]).

LEAKAGE-SAFE: identical patient-disjoint VAL/TEST handling as FedAvg — best global
checkpoint chosen on VAL, TEST scored once; best round + final round both reported.

Outputs:
  results/client_stats.csv
  results/fedprox_round_logs.csv       — per-round metrics for every mu
  results/fedprox_results.csv          — best-round + final-round per (mu, distribution)
  results/communication_analysis.csv   — appended FedProx rows
  results/client_analysis.csv          — appended per-client rows (best mu)
  figures/fedprox/{f1,loss}_curve_<dist>.png   (one line per mu)
  checkpoints/fedprox/<backbone>_<dist>_mu<mu>_fold<k>.pth

Run:  python train_fedprox.py --config configs/fedprox.yaml [--mu 0.01] [--distribution iid]
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.folds import (  # noqa: E402
    build_client_loaders, build_val_test_loaders, compute_client_stats)
from fl.client import FedProxClient  # noqa: E402
from fl.engine import evaluate_with_loss, per_client_metrics, rounds_to_best  # noqa: E402
from fl.fedavg import aggregate_fedavg  # noqa: E402
from models import build_model  # noqa: E402
from utils import (  # noqa: E402
    compute_all_metrics, get_device, load_config, plot_client_performance,
    save_config_snapshot, set_seed, setup_logging,
)
from utils.metrics import collect_predictions  # noqa: E402

METRIC_COLS = ["accuracy", "precision", "recall", "f1", "macro_f1",
               "roc_auc", "pr_auc", "specificity", "sensitivity"]


def append_csv(path: Path, df: pd.DataFrame, subset) -> None:
    if path.exists():
        df = pd.concat([pd.read_csv(path), df]).drop_duplicates(subset=subset, keep="last")
    df.to_csv(path, index=False)


def run_one_mu(cfg, device, logger, client_loaders, monitor, test_loader, n_clients, dist_label, mu):
    fold_id, backbone = cfg["fold_id"], cfg["backbone"]
    client_frac = float(cfg.get("client_fraction", 1.0))
    local_epochs = int(cfg.get("local_epochs", 3))
    rng_participation = np.random.default_rng(42)
    all_client_keys = list(client_loaders.keys())
    global_model = build_model(backbone, num_classes=2, pretrained=cfg["pretrained"]).to(device)
    criterion = nn.CrossEntropyLoss()
    model_mb = sum(p.numel() * 4 for p in global_model.parameters()) / 1e6
    ckpt_dir = Path(cfg["ckpt_dir"]) / "fedprox"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"{backbone}_{dist_label}_mu{mu}_fold{fold_id}.pth"

    best_f1, round_logs, comm_logs, cumulative_mb = -1.0, [], [], 0.0
    for rnd in range(1, cfg["num_rounds"] + 1):
        t0 = time.time()
        k_selected = max(1, int(np.ceil(client_frac * n_clients)))
        selected_keys = rng_participation.choice(all_client_keys, size=k_selected, replace=False).tolist()
        client_weights, client_sizes, round_loss = [], [], 0.0
        global_sd = copy.deepcopy(global_model.state_dict())
        for key in selected_keys:
            loader = client_loaders[key]
            local = copy.deepcopy(global_model)
            opt = optim.Adam(local.parameters(), lr=cfg["learning_rate"])
            res = FedProxClient(local, loader, criterion, opt, device, mu=mu).train(epochs=local_epochs)
            client_weights.append(res["state_dict"]); client_sizes.append(res["num_samples"])
            round_loss += res["loss"] * res["num_samples"]
        global_model.load_state_dict(global_sd)
        global_model = aggregate_fedavg(global_model, client_weights, client_sizes)

        vm = evaluate_with_loss(global_model, monitor, criterion, device)
        elapsed = time.time() - t0
        cumulative_mb += model_mb * k_selected * 2
        logger.info("[mu=%s r%03d] clients=%d/%d train_loss=%.4f val_f1=%.4f (%.1fs)",
                    mu, rnd, k_selected, n_clients, round_loss / max(1, sum(client_sizes)), vm["f1"], elapsed)
        round_logs.append({"method": "fedprox", "distribution": dist_label, "mu": mu, "fold": fold_id,
                           "client_fraction": client_frac, "local_epochs": local_epochs,
                           "round": rnd, "n_selected": k_selected,
                           "train_loss": round(round_loss / max(1, sum(client_sizes)), 5),
                           "val_loss": round(vm["loss"], 5),
                           **{f"val_{k}": round(vm[k], 5) for k in METRIC_COLS}})
        comm_logs.append({"method": "fedprox", "distribution": dist_label, "round": rnd, "mu": mu, "fold": fold_id,
                          "n_clients": n_clients, "n_selected": k_selected,
                          "model_size_mb": round(model_mb, 3),
                          "round_time_sec": round(elapsed, 2),
                          "cumulative_comm_mb": round(cumulative_mb, 2)})
        if vm["f1"] > best_f1:
            best_f1 = vm["f1"]
            torch.save(global_model.state_dict(), ckpt)

    final = compute_all_metrics(*collect_predictions(global_model, test_loader, device)[:3])
    global_model.load_state_dict(torch.load(ckpt, map_location=device))
    best_test = compute_all_metrics(*collect_predictions(global_model, test_loader, device)[:3])
    n_best = rounds_to_best([{**r, "f1": r["val_f1"]} for r in round_logs], metric="f1")
    summary = {"method": "fedprox", "distribution": dist_label, "mu": mu, "fold": fold_id,
               "backbone": backbone, "rounds": cfg["num_rounds"], "best_round": n_best,
               "best_val_f1": round(best_f1, 5),
               **{f"besttest_{k}": round(best_test[k], 5) for k in METRIC_COLS},
               **{f"final_{k}": round(final[k], 5) for k in METRIC_COLS}}
    return pd.DataFrame(round_logs), pd.DataFrame(comm_logs), summary, global_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fedprox.yaml")
    ap.add_argument("--mu", type=float, default=None, help="Single mu (overrides sweep)")
    ap.add_argument("--distribution", default=None, choices=["iid", "non-IID", "noniid", "patient"])
    ap.add_argument("--fold", type=int, default=None, help="Override fold_id from config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.distribution:
        cfg["distribution"] = args.distribution
    if args.fold is not None:
        cfg["fold_id"] = args.fold
    mu_values = [args.mu] if args.mu is not None else cfg.get("mu_values", [0.001, 0.01, 0.1])

    set_seed()
    device = get_device()
    fold_id = cfg["fold_id"]
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    logger = setup_logging(Path(cfg["results_dir"]) / "logs" / "fedprox.log")
    save_config_snapshot(cfg, Path(cfg["results_dir"]) / "configs" / "fedprox.json")

    _raw_dist = str(cfg.get("distribution", "non-IID")).lower()
    dist_label = "iid" if _raw_dist == "iid" else (_raw_dist if _raw_dist.startswith("dirichlet") else "noniid")
    partition = "iid" if dist_label == "iid" else cfg.get("partition", "patient")
    dirichlet_alpha = cfg.get("dirichlet_alpha", 0.5)
    val_frac = cfg.get("val_frac", 0.2)

    stats = compute_client_stats(cfg["fold_dir"], fold_id)
    stats.to_csv(Path(cfg["results_dir"]) / "client_stats.csv", index=False)
    diag_map = dict(zip(stats["patient_id"].astype(str), stats["diagnosis"]))

    val_loader, test_loader, val_pat = build_val_test_loaders(
        fold_dir=cfg["fold_dir"], fold_id=fold_id, batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"], root_dir=cfg["root_dir"], image_root=cfg["image_root"],
        val_frac=val_frac)
    client_loaders = build_client_loaders(
        fold_dir=cfg["fold_dir"], fold_id=fold_id, batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"], root_dir=cfg["root_dir"], image_root=cfg["image_root"],
        partition=partition, num_clients=cfg.get("num_clients"), holdout_patients=set(val_pat),
        dirichlet_alpha=dirichlet_alpha)
    monitor = val_loader if val_loader is not None else test_loader
    n_clients = len(client_loaders)
    logger.info("FedProx | dist=%s partition=%s clients=%d mu_values=%s rounds=%d val_patients=%s",
                dist_label, partition, n_clients, mu_values, cfg["num_rounds"], val_pat)

    all_rounds, summaries = [], []
    best_overall = (-1.0, None, None)  # (val_f1, model, mu)
    for mu in mu_values:
        rl, cl, summ, model = run_one_mu(cfg, device, logger, client_loaders, monitor,
                                         test_loader, n_clients, dist_label, mu)
        all_rounds.append(rl)
        append_csv(Path(cfg["results_dir"]) / "communication_analysis.csv", cl,
                   subset=["method", "distribution", "mu", "fold", "round"])
        summaries.append(summ)
        if summ["best_val_f1"] > best_overall[0]:
            best_overall = (summ["best_val_f1"], model, mu)
        # write incrementally per-mu so an interruption doesn't lose completed mu runs
        append_csv(Path(cfg["results_dir"]) / "fedprox_round_logs.csv", rl,
                   subset=["method", "distribution", "mu", "fold", "round"])
        append_csv(Path(cfg["results_dir"]) / "fedprox_results.csv", pd.DataFrame([summ]),
                   subset=["method", "distribution", "mu", "fold", "backbone"])

    rounds_df = pd.concat(all_rounds, ignore_index=True)

    # Per-client analysis for the best mu
    if best_overall[1] is not None:
        cdf = per_client_metrics(best_overall[1], client_loaders, device, diag_map)
        cdf.insert(0, "method", "fedprox"); cdf.insert(1, "distribution", dist_label); cdf.insert(2, "fold", fold_id)
        append_csv(Path(cfg["results_dir"]) / "client_analysis.csv", cdf,
                   subset=["method", "distribution", "fold", "patient_id"])

    # Convergence curves: one line per mu
    fig_dir = Path(cfg["figures_dir"]) / "fedprox"; fig_dir.mkdir(parents=True, exist_ok=True)
    for metric, col in [("f1", "val_f1"), ("loss", "val_loss")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for mu in mu_values:
            sub = rounds_df[rounds_df["mu"] == mu]
            ax.plot(sub["round"], sub[col], marker="o", ms=3, label=f"mu={mu}")
        ax.set_xlabel("Round"); ax.set_ylabel(metric); ax.set_title(f"FedProx {dist_label} — {metric}")
        ax.legend()
        fig.tight_layout(); fig.savefig(fig_dir / f"{metric}_curve_{dist_label}.png", dpi=150); plt.close(fig)

    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
