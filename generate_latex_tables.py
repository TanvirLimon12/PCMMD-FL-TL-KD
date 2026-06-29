"""Generate LaTeX tables from results CSVs."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

RESULTS = Path("results")
OUT     = Path("results/paper_tables.tex")

BACKBONE_LABELS = {
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenet_v3":    "MobileNetV3",
    "resnet50":        "ResNet50",
}

def pm(mean, std):
    return f"${mean:.4f} \\pm {std:.4f}$"


def table_centralized(lines):
    path = RESULTS / "centralized_results.csv"
    if not path.exists():
        return
    df   = pd.read_csv(path)
    full = df[df["finetune_mode"] == "full"]
    cols = ["f1", "roc_auc", "pr_auc", "sensitivity", "specificity", "ece"]
    lines += [
        "",
        "% ─── Table 1: Centralized backbone comparison ───────────────────────",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Centralized baseline: 5-fold patient-disjoint cross-validation (mean~$\pm$~std).}",
        r"\label{tab:centralized}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Backbone & F1 & AUC-ROC & AUC-PR & Sensitivity & Specificity & ECE \\",
        r"\midrule",
    ]
    for bb in ["efficientnet_b0", "mobilenet_v3", "resnet50"]:
        sub   = full[full["backbone"] == bb]
        parts = [BACKBONE_LABELS[bb]]
        for c in cols:
            parts.append(pm(sub[c].mean(), sub[c].std()))
        lines.append(" & ".join(parts) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]


def table_tl_ablation(lines):
    path = RESULTS / "centralized_results.csv"
    if not path.exists():
        return
    df    = pd.read_csv(path)
    mode_order = ["frozen", "partial", "full"]
    modes = [m for m in mode_order if m in df["finetune_mode"].dropna().unique()]
    if len(modes) < 2:
        return
    sub = df[df["fold"] == 1].copy()
    lines += [
        "% ─── Table: TL mode ablation ────────────────────────────────────────",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Transfer-learning strategy ablation on fold~1 (F1 score).}",
        r"\label{tab:tl_ablation}",
        r"\begin{tabular}{l" + "c" * len(modes) + "}",
        r"\toprule",
        "Backbone & " + " & ".join(m.capitalize() for m in modes) + r" \\",
        r"\midrule",
    ]
    for bb in ["efficientnet_b0", "mobilenet_v3", "resnet50"]:
        parts = [BACKBONE_LABELS[bb]]
        for m in modes:
            row = sub[(sub["backbone"] == bb) & (sub["finetune_mode"] == m)]
            parts.append(f"${row['f1'].values[0]:.4f}$" if len(row) else "--")
        lines.append(" & ".join(parts) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]


def table_kd(lines):
    path = RESULTS / "kd_results.csv"
    if not path.exists():
        lines.append("% kd_results.csv not yet available")
        return
    df = pd.read_csv(path)
    # filter to distilled rows only (have numeric temperature + alpha)
    df = df[df["model"].str.startswith("distilled")].copy() if "model" in df.columns else df
    df = df.dropna(subset=["temperature", "alpha"])
    df["temperature"] = df["temperature"].astype(float)
    df["alpha"]       = df["alpha"].astype(float)
    temps  = sorted(df["temperature"].unique())
    alphas = sorted(df["alpha"].unique())
    lines += [
        "% ─── Table: KD ablation ─────────────────────────────────────────────",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Knowledge distillation: F1 score by temperature $T$ and mixing weight $\alpha$.}",
        r"\label{tab:kd}",
        r"\begin{tabular}{l" + "c" * len(alphas) + "}",
        r"\toprule",
        "$T$ & " + " & ".join(f"$\\alpha={a}$" for a in alphas) + r" \\",
        r"\midrule",
    ]
    for t in temps:
        parts = [f"$T={t}$"]
        for a in alphas:
            row = df[(df["temperature"] == t) & (df["alpha"] == a)]
            parts.append(f"${row['f1'].values[0]:.4f}$" if len(row) else "--")
        lines.append(" & ".join(parts) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]


def table_kd_5fold(lines):
    path = RESULTS / "kd_results_5fold.csv"
    if not path.exists():
        lines.append("% kd_results_5fold.csv not yet available")
        return
    df = pd.read_csv(path)
    rows_out = []
    for model_type, label in [("teacher", "Teacher (EfficientNet-B0)"),
                               ("baseline", "Baseline student (no KD)"),
                               ("distilled_T1.0_a0.7", "KD student ($T{=}1,\\alpha{=}0.7$)")]:
        sub = df[df["model"] == model_type]
        if len(sub):
            rows_out.append((label, sub["f1"].mean(), sub["f1"].std()))
    lines += [
        "% ─── Table: KD 5-fold extension (best config) ──────────────────────",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Knowledge distillation, best configuration ($T{=}1,\alpha{=}0.7$) "
        r"extended to all 5 folds (mean~$\pm$~std). Pairwise differences are not "
        r"statistically significant (paired $t$-test/Wilcoxon, $n{=}5$, $p>0.7$).}",
        r"\label{tab:kd_5fold}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Model & F1 \\",
        r"\midrule",
    ]
    for label, mean, std in rows_out:
        lines.append(f"{label} & {pm(mean, std)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]


def table_fewshot(lines):
    path = RESULTS / "fewshot_results.csv"
    if not path.exists():
        lines.append("% fewshot_results.csv not yet available")
        return
    df = pd.read_csv(path)
    if "data_pct" not in df.columns and "pct" in df.columns:
        df = df.rename(columns={"pct": "data_pct"})
    backbones = [b for b in ["efficientnet_b0", "mobilenet_v3"] if b in df["backbone"].values]
    pcts      = sorted(df["data_pct"].unique())
    lines += [
        "% ─── Table: Few-shot data efficiency ───────────────────────────────",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Data efficiency: F1 score vs.\ fraction of labeled training data (fold~1).}",
        r"\label{tab:fewshot}",
        r"\begin{tabular}{l" + "c" * len(backbones) + "}",
        r"\toprule",
        "Data (\\%) & " + " & ".join(BACKBONE_LABELS[b] for b in backbones) + r" \\",
        r"\midrule",
    ]
    for p in pcts:
        parts = [f"{p}\\%"]
        for b in backbones:
            row = df[(df["data_pct"] == p) & (df["backbone"] == b)]
            parts.append(f"${row['f1'].values[0]:.4f}$" if len(row) else "--")
        lines.append(" & ".join(parts) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]


def table_fl(lines):
    path = RESULTS / "fedavg_results.csv"
    if not path.exists():
        lines.append("% fedavg_results.csv not yet available")
        return
    df = pd.read_csv(path)
    f1_col    = "besttest_f1"    if "besttest_f1"    in df.columns else "f1"
    auc_col   = "besttest_roc_auc" if "besttest_roc_auc" in df.columns else "roc_auc"
    round_col = "best_round"     if "best_round"     in df.columns else None
    dists = [d for d in ["noniid", "iid"] if d in df.get("distribution", pd.Series([])).values] \
        if "distribution" in df.columns else df["fold"].unique().tolist()
    lines += [
        "% ─── Table: FL FedAvg results ────────────────────────────────────────",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Federated learning (FedAvg) results --- fold~1, 20 communication rounds, 6 clients.}",
        r"\label{tab:fl}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Distribution & Best F1 & Best AUC-ROC & Best Round \\",
        r"\midrule",
    ]
    for dist in (dists if dists else [None]):
        sub = df[df["distribution"] == dist] if dist and "distribution" in df.columns else df
        if len(sub) == 0:
            continue
        best = sub.sort_values(f1_col, ascending=False).iloc[0]
        f1v   = f"${best[f1_col]:.4f}$"
        aucv  = f"${best[auc_col]:.4f}$" if auc_col in best else "--"
        rnv   = str(int(best[round_col])) if round_col and round_col in best else "--"
        label = dist.capitalize() if dist else "All"
        lines.append(f"{label} & {f1v} & {aucv} & {rnv} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]


if __name__ == "__main__":
    lines = [r"% Auto-generated LaTeX tables — do not edit manually", ""]
    table_centralized(lines)
    table_tl_ablation(lines)
    table_fl(lines)
    table_kd(lines)
    table_kd_5fold(lines)
    table_fewshot(lines)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"Saved {OUT} ({len(lines)} lines)")
    print("\n".join(lines))
