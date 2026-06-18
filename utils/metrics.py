"""
utils/metrics.py
----------------
Full classification metric suite for the PCMMD binary task.

Positive class = "plasma" (idx 0) — the clinically relevant cell type.
  sensitivity = recall of plasma   (TP / (TP+FN))
  specificity = recall of non_plasma (TN / (TN+FP))
ROC-AUC / PR-AUC use the predicted probability of the positive (plasma) class.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

# plasma=0 is the positive class (see data/dataset.py CLASS_TO_IDX)
POSITIVE_CLASS_IDX = 0


@torch.no_grad()
def collect_predictions(model, loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Run the model over a loader and return:
        y_true  : (N,) int labels
        y_pred  : (N,) int argmax predictions
        y_prob  : (N,) probability of the POSITIVE (plasma) class
        pids    : list of patient_id strings (empty if loader has no meta)
    Works whether the dataset yields (img, lbl) or (img, lbl, pid, path).
    """
    model.eval()
    ys, ps, probs, pids = [], [], [], []
    for batch in loader:
        imgs = batch[0].to(device)
        lbls = batch[1]
        logits = model(imgs)
        prob = F.softmax(logits, dim=1)[:, POSITIVE_CLASS_IDX]
        pred = logits.argmax(dim=1)
        ys.extend(lbls.numpy().tolist())
        ps.extend(pred.cpu().numpy().tolist())
        probs.extend(prob.cpu().numpy().tolist())
        if len(batch) > 2:
            pids.extend([str(p) for p in batch[2]])
    return np.asarray(ys), np.asarray(ps), np.asarray(probs), pids


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Returns: accuracy, precision, recall, f1 (positive=plasma), macro_f1,
             roc_auc, pr_auc, specificity, sensitivity.
    Robust to single-class batches (AUC -> nan instead of crashing).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[POSITIVE_CLASS_IDX], average="binary",
        pos_label=POSITIVE_CLASS_IDX, zero_division=0)
    macro_f1 = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)[2]

    # Confusion matrix with explicit label order [positive(0), negative(1)]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tp, fn = cm[0, 0], cm[0, 1]
    fp, tn = cm[1, 0], cm[1, 1]
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # AUCs need both classes present; sklearn expects prob of label-1 for roc_auc.
    if len(np.unique(y_true)) < 2:
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        # y_prob is P(plasma=0). Convert true labels so positive=1 for sklearn.
        y_true_pos = (y_true == POSITIVE_CLASS_IDX).astype(int)
        roc_auc = roc_auc_score(y_true_pos, y_prob)
        pr_auc = average_precision_score(y_true_pos, y_prob)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "specificity": float(specificity),
        "sensitivity": float(sensitivity),
    }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """
    ECE for the positive (plasma) class + a reliability table.
    y_true : raw int labels (0=plasma positive). y_prob : P(plasma).
    Returns (ece: float, reliability_df: DataFrame[bin_lo,bin_hi,confidence,accuracy,count]).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    y_true_pos = (y_true == POSITIVE_CLASS_IDX).astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows, ece, n = [], 0.0, len(y_true_pos)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob > lo) & (y_prob <= hi) if i > 0 else (y_prob >= lo) & (y_prob <= hi)
        cnt = int(mask.sum())
        if cnt:
            conf = float(y_prob[mask].mean())
            acc = float(y_true_pos[mask].mean())
            ece += (cnt / n) * abs(acc - conf)
        else:
            conf, acc = 0.0, 0.0
        rows.append({"bin_lo": round(lo, 3), "bin_hi": round(hi, 3),
                     "confidence": round(conf, 5), "accuracy": round(acc, 5), "count": cnt})
    return float(ece), pd.DataFrame(rows)


def bootstrap_ci(values_true, values_pred, values_prob, metric: str = "f1",
                 n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    """Non-parametric bootstrap 95% CI for a single metric over a prediction set."""
    yt, yp, pr = np.asarray(values_true), np.asarray(values_pred), np.asarray(values_prob)
    rng = np.random.default_rng(seed)
    n = len(yt)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            stats.append(compute_all_metrics(yt[idx], yp[idx], pr[idx])[metric])
        except Exception:
            continue
    stats = np.asarray([s for s in stats if s == s])  # drop NaN
    if len(stats) == 0:
        return {"metric": metric, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    return {"metric": metric,
            "mean": round(float(np.mean(stats)), 5),
            "ci_low": round(float(np.quantile(stats, alpha / 2)), 5),
            "ci_high": round(float(np.quantile(stats, 1 - alpha / 2)), 5)}


def summarise_folds(per_fold: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """mean ± std (and 95% CI half-width) across folds for each metric."""
    rows = []
    n = len(per_fold)
    for col in metric_cols:
        vals = pd.to_numeric(per_fold[col], errors="coerce").dropna().values
        mean = float(np.mean(vals)) if len(vals) else float("nan")
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        rows.append({"metric": col, "mean": round(mean, 5), "std": round(std, 5),
                     "ci95_halfwidth": round(ci95, 5), "n_folds": n})
    return pd.DataFrame(rows)
