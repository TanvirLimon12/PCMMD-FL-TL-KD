"""
eval_centralized.py
-------------------
Eval-only: load existing checkpoint for a backbone/fold and compute test metrics.
Appends rows to results/centralized_results.csv without re-training.

Run:  python eval_centralized.py --backbone resnet50 --folds 1,2,3,4
      python eval_centralized.py --backbone efficientnet_b0 --folds 1,2,3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.folds import get_fold_loaders, validate_fold
from models import build_model
from utils import (
    bootstrap_ci, collect_predictions, compute_all_metrics,
    expected_calibration_error, get_device, load_config, set_seed, setup_logging,
    summarise_folds,
)

METRIC_COLS = ["accuracy", "precision", "recall", "f1", "macro_f1",
               "roc_auc", "pr_auc", "specificity", "sensitivity", "ece"]


def eval_one_fold(cfg, fold_id, device, logger) -> dict:
    backbone = cfg["backbone"]
    ckpt_path = Path(cfg["ckpt_dir"]) / "centralized" / f"{backbone}_fold{fold_id}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    rep = validate_fold(cfg["fold_dir"], fold_id, val_frac=cfg.get("val_frac", 0.2))
    logger.info("Fold %d: n_test=%d | ckpt=%s", fold_id, rep["n_test"], ckpt_path.name)

    _, _, test_loader = get_fold_loaders(
        fold_dir=cfg["fold_dir"], fold_id=fold_id, batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"], root_dir=cfg["root_dir"], image_root=cfg["image_root"],
        use_weighted_sampler=False, val_frac=cfg.get("val_frac", 0.2))

    model = build_model(backbone, num_classes=2, pretrained=False,
                        finetune_mode=cfg.get("finetune_mode", "full")).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    y_true, y_pred, y_prob, _ = collect_predictions(model, test_loader, device)
    tm = compute_all_metrics(y_true, y_pred, y_prob)
    ece, _ = expected_calibration_error(y_true, y_prob)
    tm["ece"] = ece

    ci = bootstrap_ci(y_true, y_pred, y_prob, metric="f1")
    row = {"fold": fold_id, "backbone": backbone, "loss": cfg.get("loss", "ce"),
           "finetune_mode": cfg.get("finetune_mode", "full"), "n_test": len(y_true), "seed": 42}
    row.update({k: round(tm[k], 5) for k in METRIC_COLS})
    row.update({"f1_ci_low": ci["ci_low"], "f1_ci_high": ci["ci_high"]})
    logger.info("Fold %d TEST f1=%.4f acc=%.4f auc=%.4f ece=%.4f",
                fold_id, tm["f1"], tm["accuracy"], tm["roc_auc"], tm["ece"])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/centralized.yaml")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--folds", required=True, help="Comma list e.g. 1,2,3,4")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["backbone"] = args.backbone
    folds = [int(x) for x in args.folds.split(",")]

    set_seed()
    device = get_device()
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    logger = setup_logging(Path(cfg["results_dir"]) / "logs" / "eval_centralized.log")
    logger.info("Eval-only | backbone=%s folds=%s | device=%s", args.backbone, folds, device)

    rows = [eval_one_fold(cfg, f, device, logger) for f in folds]
    per_fold = pd.DataFrame(rows)

    res_path = Path(cfg["results_dir"]) / "centralized_results.csv"
    if res_path.exists():
        prev = pd.read_csv(res_path)
        per_fold = pd.concat([prev, per_fold]).drop_duplicates(
            subset=["backbone", "fold", "finetune_mode"], keep="last")
    per_fold.to_csv(res_path, index=False)

    summ = summarise_folds(per_fold[per_fold["backbone"] == args.backbone], METRIC_COLS)
    summ.insert(0, "backbone", args.backbone)
    sum_path = Path(cfg["results_dir"]) / "centralized_summary.csv"
    if sum_path.exists():
        prev = pd.read_csv(sum_path)
        summ = pd.concat([prev, summ]).drop_duplicates(subset=["backbone", "metric"], keep="last")
    summ.to_csv(sum_path, index=False)

    logger.info("Saved %s and %s", res_path, sum_path)
    print(per_fold[["fold", "backbone", "f1", "roc_auc", "ece"]].to_string(index=False))


if __name__ == "__main__":
    main()
