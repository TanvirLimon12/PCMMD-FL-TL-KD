"""
Compile key numbers for the paper tables from all results CSVs.
Run after all experiments complete.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")

RESULTS = Path("results")

BACKBONE_LABELS = {
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenet_v3":    "MobileNetV3",
    "resnet50":        "ResNet50",
}

def fmt(mean, std=None):
    if std is not None:
        return f"{mean:.4f} ± {std:.4f}"
    return f"{mean:.4f}"


def table_centralized():
    path = RESULTS / "centralized_results.csv"
    if not path.exists():
        print("centralized_results.csv not found"); return
    df = pd.read_csv(path)
    full = df[df["finetune_mode"] == "full"]
    print("\n" + "="*70)
    print("TABLE 1 — Centralized backbone comparison (5-fold, full fine-tune)")
    print("="*70)
    cols = ["f1", "roc_auc", "pr_auc", "sensitivity", "specificity", "ece"]
    rows = []
    for bb in ["efficientnet_b0", "mobilenet_v3", "resnet50"]:
        sub = full[full["backbone"] == bb]
        row = {"Backbone": BACKBONE_LABELS[bb], "N folds": len(sub)}
        for c in cols:
            row[c.upper()] = fmt(sub[c].mean(), sub[c].std())
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))


def table_tl_ablation():
    path = RESULTS / "centralized_results.csv"
    if not path.exists():
        print("centralized_results.csv not found"); return
    df = pd.read_csv(path)
    modes = df["finetune_mode"].unique()
    if len(modes) < 2:
        print("TL ablation results not yet available"); return
    print("\n" + "="*70)
    print("TABLE — TL mode ablation (fold 1)")
    print("="*70)
    sub = df[df["fold"] == 1].copy()
    sub["Backbone"] = sub["backbone"].map(BACKBONE_LABELS)
    pivot = sub.pivot_table(index="Backbone", columns="finetune_mode", values="f1")
    print(pivot.to_string())


def table_kd():
    path = RESULTS / "kd_results.csv"
    if not path.exists():
        print("kd_results.csv not found — KD not complete yet"); return
    df = pd.read_csv(path)
    print("\n" + "="*70)
    print("TABLE — KD temperature × alpha ablation")
    print("="*70)
    if "model" in df.columns and "baseline" in df["model"].values:
        bl = df[df["model"] == "baseline"]["f1"].values[0]
        print(f"Baseline (no distillation) F1: {bl:.4f}")
    dist = df[df["model"].str.startswith("distilled")].copy() if "model" in df.columns else df
    dist = dist.dropna(subset=["temperature", "alpha"])
    dist["temperature"] = dist["temperature"].astype(float)
    dist["alpha"]       = dist["alpha"].astype(float)
    pivot = dist.pivot_table(index="temperature", columns="alpha", values="f1")
    print("F1 by (temperature, alpha):")
    print(pivot.to_string())
    best = dist.loc[dist["f1"].idxmax()]
    print(f"Best KD: T={best['temperature']} α={best['alpha']} F1={best['f1']:.4f}")


def table_kd_5fold():
    path = RESULTS / "kd_results_5fold.csv"
    if not path.exists():
        print("kd_results_5fold.csv not found — 5-fold KD extension not complete yet"); return
    df = pd.read_csv(path)
    print("\n" + "="*70)
    print("TABLE — KD 5-fold extension (best config T=1.0, alpha=0.7)")
    print("="*70)
    for model_type, label in [("teacher", "Teacher (EfficientNet-B0)"),
                               ("baseline", "Baseline student (no KD)"),
                               ("distilled_T1.0_a0.7", "KD student")]:
        sub = df[df["model"] == model_type]
        if len(sub):
            print(f"{label}: F1={sub['f1'].mean():.4f} ± {sub['f1'].std():.4f}  (n={len(sub)})")
    try:
        from scipy import stats
        t = df[df.model == "teacher"].set_index("fold")["f1"]
        b = df[df.model == "baseline"].set_index("fold")["f1"]
        k = df[df.model == "distilled_T1.0_a0.7"].set_index("fold")["f1"]
        _, p_kb = stats.ttest_rel(k, b)
        _, p_kt = stats.ttest_rel(k, t)
        print(f"KD vs baseline paired t-test p={p_kb:.4f} | KD vs teacher p={p_kt:.4f}")
    except ImportError:
        pass


def table_fewshot():
    path = RESULTS / "fewshot_results.csv"
    if not path.exists():
        print("fewshot_results.csv not found — few-shot not complete yet"); return
    df = pd.read_csv(path)
    print("\n" + "="*70)
    print("TABLE — Few-shot data efficiency (fold 1)")
    print("="*70)
    df["Backbone"] = df["backbone"].map(BACKBONE_LABELS)
    pivot = df.pivot_table(index="data_pct", columns="Backbone", values="f1")
    print(pivot.to_string())


def table_fl():
    path = RESULTS / "fedavg_results.csv"
    if not path.exists():
        print("fedavg_results.csv not found — FL not complete yet"); return
    df = pd.read_csv(path)
    print("\n" + "="*70)
    print("TABLE — Federated Learning results")
    print("="*70)
    cols = ["f1", "roc_auc", "sensitivity", "specificity"]
    for partition in df.get("partition", pd.Series([])).unique():
        sub = df[df["partition"] == partition]
        print(f"\nPartition: {partition}")
        for c in cols:
            print(f"  {c}: {sub[c].mean():.4f} ± {sub[c].std():.4f}")


def summary_stats():
    path = RESULTS / "statistical_analysis.csv"
    if not path.exists():
        print("statistical_analysis.csv not found"); return
    df = pd.read_csv(path)
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS — mean ± std (95% CI)")
    print("="*70)
    print(df[["group", "metric", "mean", "std", "ci95_halfwidth"]].to_string(index=False))


if __name__ == "__main__":
    table_centralized()
    table_tl_ablation()
    table_kd()
    table_kd_5fold()
    table_fewshot()
    table_fl()
    summary_stats()
