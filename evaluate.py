"""
evaluate.py
-----------
Load a checkpoint, evaluate it on a chosen fold's TEST split, and save:
  results/eval_<tag>_fold<k>_metrics.csv      — global metric suite (+ECE, +F1 bootstrap CI)
  results/eval_<tag>_fold<k>_predictions.csv  — patient_id, target, pred, prob_plasma
  results/eval_<tag>_fold<k>_per_patient.csv  — per-patient metrics
  results/curves/eval_<tag>_fold<k>_{roc,pr}_points.csv
  figures/eval/<tag>_fold<k>_{confusion,roc,pr,reliability}.png
  figures/eval/<tag>_fold<k>_per_patient_f1.png

Run:
  python evaluate.py --config configs/centralized.yaml \
      --weights checkpoints/centralized/resnet50_fold1.pth --fold 1 --tag resnet50
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import build_loader  # noqa: E402
from data.transforms import get_val_transforms  # noqa: E402
from models import build_model  # noqa: E402
from utils import (  # noqa: E402
    bootstrap_ci, collect_predictions, compute_all_metrics, expected_calibration_error,
    get_device, load_config, plot_client_performance, plot_confusion_matrix,
    plot_pr_curve, plot_reliability_diagram, plot_roc_curve, set_seed,
)

METRIC_COLS = ["accuracy", "precision", "recall", "f1", "macro_f1",
               "roc_auc", "pr_auc", "specificity", "sensitivity"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/centralized.yaml")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--tag", default="model")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed()
    device = get_device()
    fold_id = args.fold or cfg["fold_id"]
    backbone = args.backbone or cfg["backbone"]
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    csv_path = Path(cfg["fold_dir"]) / f"fold_{fold_id}.csv"
    print(f"Evaluating {args.weights} on {csv_path} (backbone={backbone})")
    test_loader = build_loader(
        csv_path=csv_path, split="test", transform=get_val_transforms(),
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
        root_dir=cfg["root_dir"], image_root=cfg["image_root"], return_meta=True)

    model = build_model(backbone, num_classes=2, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    y_true, y_pred, y_prob, pids = collect_predictions(model, test_loader, device)
    m = compute_all_metrics(y_true, y_pred, y_prob)
    ece, rel = expected_calibration_error(y_true, y_prob)
    m["ece"] = ece
    ci = bootstrap_ci(y_true, y_pred, y_prob, metric="f1")

    res = Path(cfg["results_dir"]); res.mkdir(parents=True, exist_ok=True)
    cur = res / "curves"; cur.mkdir(parents=True, exist_ok=True)
    stem = f"eval_{args.tag}_fold{fold_id}"

    pd.DataFrame([{"tag": args.tag, "fold": fold_id, "backbone": backbone, "n_test": len(y_true),
                   **{k: round(m[k], 5) for k in METRIC_COLS}, "ece": round(ece, 5),
                   "f1_ci_low": ci["ci_low"], "f1_ci_high": ci["ci_high"]}]
                 ).to_csv(res / f"{stem}_metrics.csv", index=False)

    preds = pd.DataFrame({"patient_id": pids if pids else ["?"] * len(y_true),
                          "target": y_true, "pred": y_pred, "prob_plasma": np.round(y_prob, 5)})
    preds.to_csv(res / f"{stem}_predictions.csv", index=False)

    per_patient = None
    if pids:
        rows = []
        for pid in sorted(set(pids)):
            sub = preds[preds["patient_id"] == pid]
            mm = compute_all_metrics(sub["target"].values, sub["pred"].values, sub["prob_plasma"].values)
            rows.append({"patient_id": pid, "n": len(sub),
                         **{k: round(mm[k], 5) for k in ["accuracy", "f1", "sensitivity", "specificity"]}})
        per_patient = pd.DataFrame(rows)
        per_patient.to_csv(res / f"{stem}_per_patient.csv", index=False)

    # Figures
    fig = Path(cfg["figures_dir"]) / "eval"
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    y_true_pos = (np.asarray(y_true) == 0).astype(int)
    plot_confusion_matrix(cm, fig / f"{stem}_confusion.png", title=f"{args.tag} f{fold_id}")
    plot_roc_curve(y_true_pos, y_prob, fig / f"{stem}_roc.png", title=f"ROC {args.tag} f{fold_id}",
                   save_points_csv=cur / f"{stem}_roc_points.csv")
    plot_pr_curve(y_true_pos, y_prob, fig / f"{stem}_pr.png", title=f"PR {args.tag} f{fold_id}",
                  save_points_csv=cur / f"{stem}_pr_points.csv")
    plot_reliability_diagram(rel, ece, fig / f"{stem}_reliability.png", title=f"Reliability {args.tag}")
    if per_patient is not None and len(per_patient) > 1:
        plot_client_performance(per_patient["patient_id"], per_patient["f1"],
                                fig / f"{stem}_per_patient_f1.png", ylabel="F1",
                                title=f"Per-patient F1 — {args.tag} f{fold_id}")

    print(f"\nacc={m['accuracy']:.4f} f1={m['f1']:.4f} (95%CI {ci['ci_low']}-{ci['ci_high']}) "
          f"auc={m['roc_auc']:.4f} sens={m['sensitivity']:.4f} spec={m['specificity']:.4f} ece={ece:.4f}")
    print(f"Confusion (plasma=0, non_plasma=1):\n{cm}")
    print(f"Saved: {res / (stem + '_metrics.csv')}")


if __name__ == "__main__":
    main()
