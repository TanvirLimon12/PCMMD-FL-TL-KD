"""
utils/plots.py
--------------
Matplotlib figure helpers. Uses the non-interactive 'Agg' backend so it runs
headless on Kaggle / CI. Every function saves a PNG and closes the figure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    auc,
    precision_recall_curve,
    roc_curve,
)

CLASS_NAMES = ["plasma", "non_plasma"]


def _ensure(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_confusion_matrix(cm: np.ndarray, out_path: str | Path,
                          classes: Sequence[str] = CLASS_NAMES, title: str = "Confusion Matrix") -> None:
    out_path = _ensure(out_path)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), labels=classes)
    ax.set_yticks(range(len(classes)), labels=classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true_pos: np.ndarray, y_prob: np.ndarray, out_path: str | Path,
                   title: str = "ROC Curve", save_points_csv: Optional[str | Path] = None) -> float:
    out_path = _ensure(out_path)
    fpr, tpr, thr = roc_curve(y_true_pos, y_prob)
    roc_auc = auc(fpr, tpr)
    if save_points_csv:
        pd.DataFrame({"fpr": fpr, "tpr": tpr,
                      "threshold": np.append(thr, np.nan)[:len(fpr)]}).to_csv(
            _ensure(save_points_csv), index=False)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return float(roc_auc)


def plot_pr_curve(y_true_pos: np.ndarray, y_prob: np.ndarray, out_path: str | Path,
                  title: str = "Precision-Recall Curve", save_points_csv: Optional[str | Path] = None) -> float:
    out_path = _ensure(out_path)
    prec, rec, thr = precision_recall_curve(y_true_pos, y_prob)
    pr_auc = auc(rec, prec)
    if save_points_csv:
        pd.DataFrame({"recall": rec, "precision": prec}).to_csv(
            _ensure(save_points_csv), index=False)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(rec, prec, label=f"AP = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return float(pr_auc)


def plot_reliability_diagram(reliability_df, ece: float, out_path: str | Path,
                             title: str = "Reliability Diagram") -> None:
    """Reliability diagram from utils.metrics.expected_calibration_error output."""
    out_path = _ensure(out_path)
    df = reliability_df[reliability_df["count"] > 0]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="perfect")
    ax.plot(df["confidence"], df["accuracy"], "o-", color="C3", label=f"model (ECE={ece:.3f})")
    ax.set_xlabel("Confidence (P plasma)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_efficiency_performance(labels: Sequence[str], x_vals: Sequence[float],
                                y_vals: Sequence[float], out_path: str | Path,
                                xlabel: str = "Model size (MB)", ylabel: str = "F1",
                                title: str = "Efficiency vs Performance") -> None:
    out_path = _ensure(out_path)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x_vals, y_vals, s=60, color="C0")
    for lab, x, y in zip(labels, x_vals, y_vals):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_client_performance(labels: Sequence[str], values: Sequence[float], out_path: str | Path,
                            ylabel: str = "F1", title: str = "Per-client performance",
                            colors: Optional[Sequence] = None) -> None:
    out_path = _ensure(out_path)
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.4), 4))
    ax.bar([str(x) for x in labels], values, color=colors if colors is not None else "#4C72B0")
    ax.set_ylabel(ylabel); ax.set_xlabel("Client (patient)"); ax.set_title(title)
    plt.xticks(rotation=90)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_training_history(history: List[Dict], out_path: str | Path,
                          title: str = "Training History") -> None:
    out_path = _ensure(out_path)
    if not history:
        return
    epochs = [h.get("epoch", i + 1) for i, h in enumerate(history)]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    if "train_loss" in history[0]:
        ax1.plot(epochs, [h["train_loss"] for h in history], "C0-", label="train_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="C0")
    ax2 = ax1.twinx()
    for key, style in (("f1", "C1-"), ("accuracy", "C2--")):
        if key in history[0]:
            ax2.plot(epochs, [h[key] for h in history], style, label=key)
    ax2.set_ylabel("Metric", color="C1")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
