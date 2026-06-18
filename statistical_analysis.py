"""
statistical_analysis.py
-----------------------
Statistical validation across the result CSVs (Tanjid §T9, proposal §8.2).

Produces:
  results/statistical_analysis.csv  — mean ± std + 95% CI per (source, group, metric)
  results/statistical_pairwise.csv  — paired tests (Wilcoxon signed-rank + paired t,
                                       Cohen's d) on shared folds, with interpretation

Paired comparisons (fold-aligned):
  • centralized backbones vs each other
  • centralized best vs FedAvg(non-IID)            (if fold-level rows exist)
  • centralized best vs FedProx(best mu, non-IID)
  • FedAvg vs FedProx

Reads whichever of these exist: centralized_results.csv, fedavg_results.csv,
fedprox_results.csv. FL summary rows carry both best-round (besttest_*) and final
(final_*) metrics; we use besttest_* for comparison.

Run:  python statistical_analysis.py --config configs/centralized.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_config  # noqa: E402

try:
    from scipy import stats as scipy_stats  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

METRICS = ["accuracy", "f1", "roc_auc", "pr_auc", "sensitivity", "specificity"]


def _summary_rows(df, source, group_col, metric_cols):
    rows = []
    for g, sub in df.groupby(group_col):
        for col in metric_cols:
            if col not in sub.columns:
                continue
            vals = pd.to_numeric(sub[col], errors="coerce").dropna().values
            if len(vals) == 0:
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ci = 1.96 * std / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            rows.append({"source": source, "group": str(g), "metric": col,
                         "mean": round(mean, 5), "std": round(std, 5),
                         "ci95_halfwidth": round(ci, 5), "n": len(vals)})
    return rows


def _cohens_d(a, b):
    diff = np.asarray(a, float) - np.asarray(b, float)
    sd = np.std(diff, ddof=1) if len(diff) > 1 else 0.0
    return float(np.mean(diff) / sd) if sd > 0 else 0.0


def _bootstrap_paired_ci(a, b, n_boot=2000, seed=42):
    """Bootstrap 95% CI of the paired mean difference (a-b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    diffs = a - b
    boot = [np.mean(diffs[rng.integers(0, len(diffs), len(diffs))]) for _ in range(n_boot)]
    return round(float(np.quantile(boot, 0.025)), 5), round(float(np.quantile(boot, 0.975)), 5)


def _paired_tests(name, a, b, col):
    """Wilcoxon + paired t + Cohen's d + bootstrap CI with a plain-English interpretation."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mean_diff = float(np.mean(a - b))
    t, pt, w, pw = (np.nan,) * 4
    if _HAVE_SCIPY and len(a) >= 2:
        t, pt = scipy_stats.ttest_rel(a, b)
        if np.any(a - b != 0):
            try:
                w, pw = scipy_stats.wilcoxon(a, b)
            except Exception:
                w, pw = np.nan, np.nan
    lo, hi = _bootstrap_paired_ci(a, b) if len(a) >= 2 else (np.nan, np.nan)
    p_use = pw if pw == pw else pt
    if p_use == p_use:
        sig = "significant" if p_use < 0.05 else "not significant"
        better = name.split("_vs_")[0] if mean_diff > 0 else name.split("_vs_")[-1]
        interp = f"{better} higher by {abs(mean_diff):.4f} ({sig}, p={p_use:.3f})"
    else:
        interp = f"mean diff {mean_diff:+.4f} (p-value unavailable; install scipy)"
    return {"comparison": name, "metric": col, "n_folds": len(a),
            "mean_diff": round(mean_diff, 5),
            "wilcoxon_p": round(float(pw), 4) if pw == pw else None,
            "ttest_p": round(float(pt), 4) if pt == pt else None,
            "cohens_d": round(_cohens_d(a, b), 4),
            "diff_ci_low": lo, "diff_ci_high": hi, "interpretation": interp}


def _fold_series(df, metric):
    """{fold: value} for a single-group df, picking besttest_/final_/raw column."""
    for cand in (f"besttest_{metric}", metric, f"final_{metric}"):
        if cand in df.columns:
            return dict(zip(df["fold"], pd.to_numeric(df[cand], errors="coerce")))
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/centralized.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = Path(cfg["results_dir"])

    summary_rows, pairwise_rows = [], []
    cen = fa = fp = None

    cen_path = res / "centralized_results.csv"
    if cen_path.exists():
        cen = pd.read_csv(cen_path)
        summary_rows += _summary_rows(cen, "centralized", "backbone", METRICS)
        for b1, b2 in combinations(sorted(cen["backbone"].unique()), 2):
            d1, d2 = _fold_series(cen[cen.backbone == b1], "f1"), _fold_series(cen[cen.backbone == b2], "f1")
            shared = sorted(set(d1) & set(d2))
            for col in METRICS:
                s1 = _fold_series(cen[cen.backbone == b1], col)
                s2 = _fold_series(cen[cen.backbone == b2], col)
                sh = sorted(set(s1) & set(s2))
                if len(sh) >= 2:
                    pairwise_rows.append(_paired_tests(f"{b1}_vs_{b2}", [s1[f] for f in sh],
                                                       [s2[f] for f in sh], col))

    fa_path = res / "fedavg_results.csv"
    if fa_path.exists():
        fa = pd.read_csv(fa_path)
        fa = fa.assign(group=fa["distribution"])
        summary_rows += _summary_rows(fa.rename(columns=lambda c: c.replace("besttest_", "")),
                                      "fedavg", "group", METRICS)

    fp_path = res / "fedprox_results.csv"
    if fp_path.exists():
        fp = pd.read_csv(fp_path)
        fp["group"] = fp["distribution"].astype(str) + "_mu" + fp["mu"].astype(str)
        summary_rows += _summary_rows(fp.rename(columns=lambda c: c.replace("besttest_", "")),
                                      "fedprox", "group", METRICS)

    # Cross-method paired tests on shared folds (non-IID), using besttest_* metrics
    def central_best_series(metric):
        if cen is None:
            return {}
        means = cen.groupby("backbone")["f1"].mean() if "f1" in cen else None
        best_bb = means.idxmax() if means is not None and len(means) else None
        return _fold_series(cen[cen.backbone == best_bb], metric) if best_bb else {}

    if fa is not None:
        fa_ni = fa[fa["distribution"] == "noniid"]
        for col in METRICS:
            cser, fser = central_best_series(col), _fold_series(fa_ni, col)
            sh = sorted(set(cser) & set(fser))
            if len(sh) >= 2:
                pairwise_rows.append(_paired_tests("centralized_vs_fedavg", [cser[f] for f in sh],
                                                   [fser[f] for f in sh], col))
    if fp is not None:
        # best mu per fold for non-IID
        fp_ni = fp[fp["distribution"] == "noniid"]
        if len(fp_ni):
            idx = fp_ni.groupby("fold")["best_val_f1"].idxmax()
            fp_best = fp_ni.loc[idx]
            for col in METRICS:
                cser, fser = central_best_series(col), _fold_series(fp_best, col)
                sh = sorted(set(cser) & set(fser))
                if len(sh) >= 2:
                    pairwise_rows.append(_paired_tests("centralized_vs_fedprox", [cser[f] for f in sh],
                                                       [fser[f] for f in sh], col))
            if fa is not None:
                fa_ni = fa[fa["distribution"] == "noniid"]
                for col in METRICS:
                    aser, bser = _fold_series(fa_ni, col), _fold_series(fp_best, col)
                    sh = sorted(set(aser) & set(bser))
                    if len(sh) >= 2:
                        pairwise_rows.append(_paired_tests("fedavg_vs_fedprox", [aser[f] for f in sh],
                                                           [bser[f] for f in sh], col))

    if not summary_rows:
        print("No result CSVs found yet. Run training first.")
        return
    res.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(res / "statistical_analysis.csv", index=False)
    if pairwise_rows:
        pd.DataFrame(pairwise_rows).to_csv(res / "statistical_pairwise.csv", index=False)
    if not _HAVE_SCIPY:
        print("[note] scipy missing → Wilcoxon/t-test p-values are None (pip install scipy).")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\nSaved: {res / 'statistical_analysis.csv'}"
          + (f" and {res / 'statistical_pairwise.csv'}" if pairwise_rows else ""))


if __name__ == "__main__":
    main()
