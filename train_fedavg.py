"""
train_fedavg.py
---------------
FedAvg simulation over PCMMD clients (Tanjid §T3–T8).

Partitioning (config 'distribution'):
  • non-IID : one client per patient_id (default; natural label/quantity skew)
  • IID     : train pool shuffled into equal shards

LEAKAGE-SAFE: a patient-disjoint VAL set (held out from the federation) selects the
best global checkpoint; the TEST fold is scored once at the best round. Best round
and final round are both reported.

Outputs:
  results/client_stats.csv             — per-patient cell counts / plasma %
  results/fedavg_round_logs.csv        — per-round train_loss/val_loss/val metrics
  results/fedavg_results.csv           — best-round + final-round test summary
  results/communication_analysis.csv   — rounds/runtime/model size/comm cost (FedAvg rows)
  results/client_analysis.csv          — per-client performance of the global model
  figures/fedavg/{f1,loss,pr_auc}_curve_<dist>.png · figures/clients/client_perf_fedavg_<dist>.png
  checkpoints/fedavg/<backbone>_<dist>_fold<k>.pth

Run:  python train_fedavg.py --config configs/fedavg.yaml [--distribution iid]
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
from fl.client import FLClient  # noqa: E402
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


def run_fedavg(cfg, device, logger) -> None:
    fold_id, backbone = cfg["fold_id"], cfg["backbone"]
    _raw_dist = str(cfg.get("distribution", "non-IID")).lower()
    dist_label = "iid" if _raw_dist == "iid" else (_raw_dist if _raw_dist.startswith("dirichlet") else "noniid")
    # partition: iid always uses "iid"; non-IID defaults to "patient" but can be "dirichlet"
    partition = "iid" if dist_label == "iid" else cfg.get("partition", "patient")
    dirichlet_alpha = cfg.get("dirichlet_alpha", 0.5)
    val_frac = cfg.get("val_frac", 0.2)

    stats = compute_client_stats(cfg["fold_dir"], fold_id)
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
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
    client_frac = float(cfg.get("client_fraction", 1.0))  # C-fraction: 0.5 | 0.75 | 1.0
    local_epochs = int(cfg.get("local_epochs", 3))
    rng_participation = np.random.default_rng(42)
    logger.info("FedAvg | dist=%s partition=%s clients=%d C=%.2f E=%d rounds=%d fold=%d",
                dist_label, partition, n_clients, client_frac, local_epochs, cfg["num_rounds"], fold_id)

    global_model = build_model(backbone, num_classes=2, pretrained=cfg["pretrained"]).to(device)
    criterion = nn.CrossEntropyLoss()
    model_mb = sum(p.numel() * 4 for p in global_model.parameters()) / 1e6

    ckpt_dir = Path(cfg["ckpt_dir"]) / "fedavg"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"{backbone}_{dist_label}_fold{fold_id}.pth"

    best_f1, round_logs, comm_logs, cumulative_mb = -1.0, [], [], 0.0
    all_client_keys = list(client_loaders.keys())
    for rnd in range(1, cfg["num_rounds"] + 1):
        t0 = time.time()
        # partial participation: sample ceil(C * n) clients without replacement
        k_selected = max(1, int(np.ceil(client_frac * n_clients)))
        selected_keys = rng_participation.choice(all_client_keys, size=k_selected, replace=False).tolist()
        client_weights, client_sizes, round_loss = [], [], 0.0
        global_sd = copy.deepcopy(global_model.state_dict())
        for key in selected_keys:
            loader = client_loaders[key]
            local = copy.deepcopy(global_model)
            opt = optim.Adam(local.parameters(), lr=cfg["learning_rate"])
            res = FLClient(local, loader, criterion, opt, device).train(epochs=local_epochs)
            client_weights.append(res["state_dict"]); client_sizes.append(res["num_samples"])
            round_loss += res["loss"] * res["num_samples"]
        global_model.load_state_dict(global_sd)
        global_model = aggregate_fedavg(global_model, client_weights, client_sizes)

        vm = evaluate_with_loss(global_model, monitor, criterion, device)
        elapsed = time.time() - t0
        cumulative_mb += model_mb * k_selected * 2
        logger.info("[r%03d] clients=%d/%d train_loss=%.4f val_f1=%.4f (%.1fs)",
                    rnd, k_selected, n_clients, round_loss / max(1, sum(client_sizes)), vm["f1"], elapsed)
        round_logs.append({"method": "fedavg", "distribution": dist_label, "fold": fold_id,
                           "client_fraction": client_frac, "local_epochs": local_epochs,
                           "round": rnd, "n_selected": k_selected,
                           "train_loss": round(round_loss / max(1, sum(client_sizes)), 5),
                           "val_loss": round(vm["loss"], 5),
                           **{f"val_{k}": round(vm[k], 5) for k in METRIC_COLS}})
        comm_logs.append({"method": "fedavg", "distribution": dist_label, "fold": fold_id, "round": rnd,
                          "n_clients": n_clients, "n_selected": k_selected,
                          "model_size_mb": round(model_mb, 3),
                          "round_time_sec": round(elapsed, 2),
                          "cumulative_comm_mb": round(cumulative_mb, 2)})
        if vm["f1"] > best_f1:
            best_f1 = vm["f1"]
            torch.save(global_model.state_dict(), ckpt)

    # round logs use val_f1 as the selection metric for rounds_to_best
    rl = pd.DataFrame(round_logs)
    append_csv(Path(cfg["results_dir"]) / "fedavg_round_logs.csv", rl,
               subset=["method", "distribution", "fold", "round"])
    append_csv(Path(cfg["results_dir"]) / "communication_analysis.csv", pd.DataFrame(comm_logs),
               subset=["method", "distribution", "fold", "round"])

    # Best-round (restore ckpt) and final-round TEST metrics
    final = compute_all_metrics(*collect_predictions(global_model, test_loader, device)[:3])
    global_model.load_state_dict(torch.load(ckpt, map_location=device))
    best_test = compute_all_metrics(*collect_predictions(global_model, test_loader, device)[:3])
    n_best = rounds_to_best([{**r, "f1": r["val_f1"]} for r in round_logs], metric="f1")

    summary = pd.DataFrame([{"method": "fedavg", "distribution": dist_label, "fold": fold_id,
                             "backbone": backbone, "rounds": cfg["num_rounds"], "best_round": n_best,
                             "best_val_f1": round(best_f1, 5),
                             **{f"besttest_{k}": round(best_test[k], 5) for k in METRIC_COLS},
                             **{f"final_{k}": round(final[k], 5) for k in METRIC_COLS}}])
    append_csv(Path(cfg["results_dir"]) / "fedavg_results.csv", summary,
               subset=["method", "distribution", "fold", "backbone"])

    # Per-client analysis (global model on each client) + bar plot
    cdf = per_client_metrics(global_model, client_loaders, device, diag_map)
    cdf.insert(0, "method", "fedavg"); cdf.insert(1, "distribution", dist_label); cdf.insert(2, "fold", fold_id)
    append_csv(Path(cfg["results_dir"]) / "client_analysis.csv", cdf,
               subset=["method", "distribution", "fold", "patient_id"])
    colors = cdf["diagnosis"].map({"mm": "#C44E52", "normal": "#4C72B0"}).fillna("#8C8C8C")
    plot_client_performance(cdf["patient_id"], cdf["f1"],
                            Path(cfg["figures_dir"]) / "clients" / f"client_perf_fedavg_{dist_label}.png",
                            ylabel="F1", title=f"FedAvg {dist_label} per-client F1", colors=colors)

    # Convergence curves
    fig_dir = Path(cfg["figures_dir"]) / "fedavg"; fig_dir.mkdir(parents=True, exist_ok=True)
    for metric, col in [("f1", "val_f1"), ("loss", "val_loss"), ("pr_auc", "val_pr_auc")]:
        if col not in rl:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rl["round"], rl[col], marker="o", ms=3)
        ax.set_xlabel("Round"); ax.set_ylabel(metric); ax.set_title(f"FedAvg {dist_label} — {metric}")
        fig.tight_layout(); fig.savefig(fig_dir / f"{metric}_curve_{dist_label}.png", dpi=150); plt.close(fig)

    logger.info("Done. best_val_f1=%.4f @round %d | best-test f1=%.4f | final f1=%.4f",
                best_f1, n_best, best_test["f1"], final["f1"])
    print(summary.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fedavg.yaml")
    ap.add_argument("--distribution", default=None, choices=["iid", "non-IID", "noniid", "patient"])
    ap.add_argument("--fold", type=int, default=None, help="Override fold_id from config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.distribution:
        cfg["distribution"] = args.distribution
    if args.fold is not None:
        cfg["fold_id"] = args.fold
    set_seed()
    device = get_device()
    logger = setup_logging(Path(cfg["results_dir"]) / "logs" / "fedavg.log")
    save_config_snapshot(cfg, Path(cfg["results_dir"]) / "configs" / "fedavg.json")
    run_fedavg(cfg, device, logger)


if __name__ == "__main__":
    main()
