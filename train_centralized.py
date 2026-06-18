"""
train_centralized.py
--------------------
Centralized 5-fold cross-validation baseline — the upper bound for FL comparison.

Backbones: resnet50 | efficientnet_b0 | mobilenet_v3  (config or --backbone)
Optimizer: AdamW + CosineAnnealingLR
Loss:      ce | weighted_ce | focal   (config 'loss'; reported in the snapshot)
TL mode:   full | frozen | partial    (config 'finetune_mode'; proposal §7.2/E3)

LEAKAGE-SAFE: early stopping & best-checkpoint selection use a patient-disjoint
VALIDATION set carved from the train patients. The TEST fold is evaluated exactly
once, after restoring the best (val-selected) checkpoint.

Outputs:
  results/centralized_results.csv     — one row per fold (full metric suite, incl. ECE)
  results/centralized_summary.csv     — mean ± std (+95% CI) across folds
  results/centralized_<backbone>_fold<k>_history.csv
  results/curves/<backbone>_fold<k>_{roc,pr}_points.csv
  figures/centralized/<backbone>_fold<k>_{confusion,roc,pr,reliability,history}.png
  checkpoints/centralized/<backbone>_fold<k>.pth
  results/configs/centralized_<backbone>.json

Run:  python train_centralized.py --config configs/centralized.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import PCMMDDataset  # noqa: E402
from data.folds import get_fold_loaders, validate_fold  # noqa: E402
from models import build_model  # noqa: E402
from utils import (  # noqa: E402
    bootstrap_ci, build_loss, collect_predictions, compute_all_metrics,
    expected_calibration_error, get_device, load_config, plot_confusion_matrix,
    plot_pr_curve, plot_reliability_diagram, plot_roc_curve, plot_training_history,
    save_config_snapshot, set_seed, setup_logging, summarise_folds,
)

METRIC_COLS = ["accuracy", "precision", "recall", "f1", "macro_f1",
               "roc_auc", "pr_auc", "specificity", "sensitivity", "ece"]


def make_loss(cfg, fold_dir, fold_id, device):
    name = cfg.get("loss", "ce")
    weights = None
    if name in ("weighted_ce", "weighted", "focal"):
        train_ds = PCMMDDataset(Path(fold_dir) / f"fold_{fold_id}.csv", split="train",
                                image_root=cfg["image_root"], root_dir=cfg["root_dir"])
        weights = train_ds.class_weights().to(device)
    return build_loss(name, class_weights=weights, focal_gamma=cfg.get("focal_gamma", 2.0))


def evaluate_split(model, loader, device):
    y_true, y_pred, y_prob, pids = collect_predictions(model, loader, device)
    m = compute_all_metrics(y_true, y_pred, y_prob)
    ece, rel = expected_calibration_error(y_true, y_prob)
    m["ece"] = ece
    return m, (y_true, y_pred, y_prob, pids, rel)


def train_one_fold(cfg, fold_id, device, logger) -> dict:
    backbone = cfg["backbone"]
    rep = validate_fold(cfg["fold_dir"], fold_id, val_frac=cfg.get("val_frac", 0.2))
    if rep["patient_leakage"]:
        logger.warning("Fold %d PATIENT LEAKAGE train/test: %s", fold_id, rep["patient_leakage"][:10])
    logger.info("Fold %d: train=%d test=%d | clients=%d val_patients=%d (%s)",
                fold_id, rep["n_train"], rep["n_test"], rep["n_train_clients"],
                rep["n_val_patients"], rep["val_patients"])

    train_loader, val_loader, test_loader = get_fold_loaders(
        fold_dir=cfg["fold_dir"], fold_id=fold_id, batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"], root_dir=cfg["root_dir"], image_root=cfg["image_root"],
        use_weighted_sampler=cfg["use_weighted_sampler"], val_frac=cfg.get("val_frac", 0.2))
    monitor = val_loader if val_loader is not None else test_loader
    if val_loader is None:
        logger.warning("Fold %d: no val patients available -> monitoring on TEST (report cautiously).", fold_id)

    model = build_model(backbone, num_classes=2, pretrained=cfg["pretrained"],
                        finetune_mode=cfg.get("finetune_mode", "full")).to(device)
    criterion = make_loss(cfg, cfg["fold_dir"], fold_id, device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    ckpt_dir = Path(cfg["ckpt_dir"]) / "centralized"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{backbone}_fold{fold_id}.pth"

    best_f1, patience_ctr, history = -1.0, 0, []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running, total = 0.0, 0
        for batch in train_loader:
            imgs, lbls = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0); total += imgs.size(0)
        scheduler.step()

        vm, _ = evaluate_split(model, monitor, device)
        epoch_loss = running / max(1, total)
        history.append({"epoch": epoch, "train_loss": round(epoch_loss, 5),
                        "val_f1": round(vm["f1"], 5), "val_accuracy": round(vm["accuracy"], 5),
                        "val_auc": round(vm["roc_auc"], 5)})
        logger.info("  [%s f%d] ep %03d/%d loss=%.4f val_f1=%.4f val_acc=%.4f val_auc=%.4f",
                    backbone, fold_id, epoch, cfg["epochs"], epoch_loss,
                    vm["f1"], vm["accuracy"], vm["roc_auc"])
        if vm["f1"] > best_f1:
            best_f1, patience_ctr = vm["f1"], 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["patience"]:
                logger.info("  early stop @ epoch %d (no val-F1 gain for %d)", epoch, cfg["patience"])
                break

    # Restore best (val-selected) checkpoint, evaluate TEST once
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    tm, (y_true, y_pred, y_prob, _, rel) = evaluate_split(model, test_loader, device)

    fig_dir = Path(cfg["figures_dir"]) / "centralized"
    cur_dir = Path(cfg["results_dir"]) / "curves"; cur_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    y_true_pos = (np.asarray(y_true) == 0).astype(int)
    plot_confusion_matrix(cm, fig_dir / f"{backbone}_fold{fold_id}_confusion.png", title=f"{backbone} f{fold_id}")
    plot_roc_curve(y_true_pos, y_prob, fig_dir / f"{backbone}_fold{fold_id}_roc.png",
                   title=f"ROC {backbone} f{fold_id}",
                   save_points_csv=cur_dir / f"{backbone}_fold{fold_id}_roc_points.csv")
    plot_pr_curve(y_true_pos, y_prob, fig_dir / f"{backbone}_fold{fold_id}_pr.png",
                  title=f"PR {backbone} f{fold_id}",
                  save_points_csv=cur_dir / f"{backbone}_fold{fold_id}_pr_points.csv")
    plot_reliability_diagram(rel, tm["ece"], fig_dir / f"{backbone}_fold{fold_id}_reliability.png",
                             title=f"Reliability {backbone} f{fold_id}")
    plot_training_history(history, fig_dir / f"{backbone}_fold{fold_id}_history.png",
                          title=f"History {backbone} f{fold_id}")
    pd.DataFrame(history).to_csv(
        Path(cfg["results_dir"]) / f"centralized_{backbone}_fold{fold_id}_history.csv", index=False)

    ci = bootstrap_ci(y_true, y_pred, y_prob, metric="f1")
    row = {"fold": fold_id, "backbone": backbone, "loss": cfg.get("loss", "ce"),
           "finetune_mode": cfg.get("finetune_mode", "full"), "n_test": len(y_true), "seed": 42}
    row.update({k: round(tm[k], 5) for k in METRIC_COLS})
    row.update({"f1_ci_low": ci["ci_low"], "f1_ci_high": ci["ci_high"]})
    logger.info("Fold %d TEST f1=%.4f acc=%.4f auc=%.4f ece=%.4f (best val_f1=%.4f)",
                fold_id, tm["f1"], tm["accuracy"], tm["roc_auc"], tm["ece"], best_f1)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/centralized.yaml")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--folds", default=None, help="Comma list e.g. 1,2,3")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.backbone:
        cfg["backbone"] = args.backbone
    if args.folds:
        cfg["folds"] = [int(x) for x in args.folds.split(",")]

    set_seed()
    device = get_device()
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    logger = setup_logging(Path(cfg["results_dir"]) / "logs" / "centralized.log")
    save_config_snapshot(cfg, Path(cfg["results_dir"]) / "configs" / f"centralized_{cfg['backbone']}.json")
    logger.info("Centralized CV | backbone=%s folds=%s loss=%s ft=%s | device=%s",
                cfg["backbone"], cfg["folds"], cfg.get("loss", "ce"),
                cfg.get("finetune_mode", "full"), device)

    rows = [train_one_fold(cfg, f, device, logger) for f in cfg["folds"]]
    per_fold = pd.DataFrame(rows)

    res_path = Path(cfg["results_dir"]) / "centralized_results.csv"
    if res_path.exists():
        prev = pd.read_csv(res_path)
        per_fold = pd.concat([prev, per_fold]).drop_duplicates(subset=["backbone", "fold"], keep="last")
    per_fold.to_csv(res_path, index=False)

    summ = summarise_folds(per_fold[per_fold["backbone"] == cfg["backbone"]], METRIC_COLS)
    summ.insert(0, "backbone", cfg["backbone"])
    sum_path = Path(cfg["results_dir"]) / "centralized_summary.csv"
    if sum_path.exists():
        prev = pd.read_csv(sum_path)
        summ = pd.concat([prev, summ]).drop_duplicates(subset=["backbone", "metric"], keep="last")
    summ.to_csv(sum_path, index=False)
    logger.info("Saved %s and %s", res_path, sum_path)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
