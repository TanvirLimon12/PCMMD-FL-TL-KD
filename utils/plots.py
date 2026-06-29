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


def plot_comm_vs_accuracy(
    labels: Sequence[str],
    comm_mb: Sequence[float],
    f1_scores: Sequence[float],
    out_path: str | Path,
    title: str = "Communication Cost vs F1",
) -> None:
    """Scatter: cumulative communication (MB) on x-axis, best-test F1 on y-axis.
    One point per method (FedAvg-IID, FedAvg-nonIID, FedProx-best-mu, FedBN, KD-student, etc.)
    """
    out_path = _ensure(out_path)
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for lab, x, y, c in zip(labels, comm_mb, f1_scores, colors):
        ax.scatter(x, y, color=c, s=80, zorder=5)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("Cumulative communication (MB)")
    ax.set_ylabel("Best-round Test F1")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dirichlet_distribution(
    client_label_counts: Dict[str, Dict[str, int]],
    out_path: str | Path,
    title: str = "Label Distribution per Client (Dirichlet)",
) -> None:
    """Stacked bar chart showing plasma / non_plasma counts per client."""
    out_path = _ensure(out_path)
    clients = sorted(client_label_counts)
    plasma_counts = [client_label_counts[c].get("plasma", 0) for c in clients]
    non_plasma_counts = [client_label_counts[c].get("non_plasma", 0) for c in clients]
    x = np.arange(len(clients))
    fig, ax = plt.subplots(figsize=(max(6, len(clients) * 0.5), 4))
    ax.bar(x, plasma_counts, label="plasma", color="#C44E52")
    ax.bar(x, non_plasma_counts, bottom=plasma_counts, label="non_plasma", color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(clients, rotation=90, fontsize=7)
    ax.set_xlabel("Client")
    ax.set_ylabel("Sample count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_multi(curves: Dict[str, tuple], out_path: str | Path,
                   title: str = "ROC Curves") -> None:
    """Overlay ROC curves. curves = {label: (fpr, tpr)} or {label: (y_true, y_prob)}."""
    out_path = _ensure(out_path)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    for label, vals in curves.items():
        if len(vals[0]) == len(vals[1]) and vals[0][0] in (0.0, 1.0) and max(vals[0]) <= 1.0:
            fpr, tpr = vals[0], vals[1]
            roc_auc = auc(fpr, tpr)
        else:
            fpr, tpr, _ = roc_curve(vals[0], vals[1])
            roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_pr_multi(curves: Dict[str, tuple], out_path: str | Path,
                  title: str = "Precision-Recall Curves") -> None:
    """Overlay PR curves. curves = {label: (precision, recall)} or {label: (y_true, y_prob)}."""
    out_path = _ensure(out_path)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for label, vals in curves.items():
        if len(vals) == 2 and np.max(vals[0]) <= 1.01:
            prec, rec = vals[0], vals[1]
            ap = auc(rec[::-1], prec[::-1]) if len(rec) > 1 else float(np.mean(prec))
        else:
            prec, rec, _ = precision_recall_curve(vals[0], vals[1])
            ap = auc(rec, prec)
        ax.plot(rec, prec, label=f"{label} (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title); ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_grouped_bar(
    groups: Sequence[str],
    series: Dict[str, Sequence[float]],
    out_path: str | Path,
    ylabel: str = "Score",
    title: str = "Comparison",
    err: Optional[Dict[str, Sequence[float]]] = None,
    ylim: Optional[tuple] = None,
    legend_loc: str = "upper right",
) -> None:
    """Grouped bar chart. groups = x-labels; series = {series_name: [values]}."""
    out_path = _ensure(out_path)
    n_groups, n_series = len(groups), len(series)
    width = 0.8 / n_series
    x = np.arange(n_groups)
    fig, ax = plt.subplots(figsize=(max(6, n_groups * 1.2), 4.5))
    for i, (name, vals) in enumerate(series.items()):
        offset = (i - n_series / 2 + 0.5) * width
        yerr = err[name] if err and name in err else None
        ax.bar(x + offset, vals, width=width * 0.9, label=name, yerr=yerr,
               error_kw={"elinewidth": 1, "capsize": 3})
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(loc=legend_loc, fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_confusion_matrix_grid(
    cms: Dict[str, np.ndarray],
    out_path: str | Path,
    classes: Sequence[str] = ("plasma", "non_plasma"),
    title: str = "Confusion Matrices",
) -> None:
    """Side-by-side confusion matrices, one per key in cms."""
    out_path = _ensure(out_path)
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    if n == 1:
        axes = [axes]
    for ax, (label, cm) in zip(axes, cms.items()):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("True", fontsize=8)
        ax.set_title(label, fontsize=9)
        thresh = cm.max() / 2.0 if cm.max() else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_radar_chart(
    methods: Sequence[str],
    metrics: Sequence[str],
    values: np.ndarray,
    out_path: str | Path,
    title: str = "Method Comparison",
) -> None:
    """Radar/spider chart. values shape: (n_methods, n_metrics)."""
    out_path = _ensure(out_path)
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(5.5, 5), subplot_kw={"polar": True})
    for i, method in enumerate(methods):
        vals = list(values[i]) + [values[i][0]]
        ax.plot(angles, vals, "o-", linewidth=1.5, label=method)
        ax.fill(angles, vals, alpha=0.1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels=metrics, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_heatmap(
    data: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    out_path: str | Path,
    title: str = "Heatmap",
    fmt: str = ".3f",
    cmap: str = "RdYlGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Generic heatmap with annotated cells."""
    out_path = _ensure(out_path)
    fig, ax = plt.subplots(figsize=(max(5, len(col_labels) * 1.1), max(4, len(row_labels) * 0.7)))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, format(v, fmt) if not np.isnan(v) else "—",
                    ha="center", va="center", fontsize=7.5,
                    color="black" if 0.2 < (v - (vmin or 0)) / max((vmax or 1) - (vmin or 0), 1e-9) < 0.8 else "white")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_stacked_bar(
    categories: Sequence[str],
    stacks: Dict[str, Sequence[float]],
    out_path: str | Path,
    xlabel: str = "",
    ylabel: str = "Count",
    title: str = "Distribution",
    colors: Optional[Sequence[str]] = None,
    rotate_x: int = 0,
) -> None:
    """Stacked bar chart. stacks = {stack_label: [values per category]}."""
    out_path = _ensure(out_path)
    fig, ax = plt.subplots(figsize=(max(6, len(categories) * 0.8), 4))
    bottoms = np.zeros(len(categories))
    palette = colors or [f"C{i}" for i in range(len(stacks))]
    for (name, vals), col in zip(stacks.items(), palette):
        ax.bar(categories, vals, bottom=bottoms, label=name, color=col)
        bottoms += np.array(vals, dtype=float)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    if rotate_x:
        plt.xticks(rotation=rotate_x, ha="right")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_line_with_bands(
    x: Sequence[float],
    series: Dict[str, tuple],
    out_path: str | Path,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str = "Curve",
    markers: bool = False,
) -> None:
    """Line chart with optional ±std bands. series = {label: (mean_arr, std_arr)}."""
    out_path = _ensure(out_path)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, (label, (mean, std)) in enumerate(series.items()):
        mean, std = np.asarray(mean), np.asarray(std)
        ax.plot(x, mean, label=label, color=f"C{i}", marker="o" if markers else None,
                markersize=3 if markers else None)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=f"C{i}")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
