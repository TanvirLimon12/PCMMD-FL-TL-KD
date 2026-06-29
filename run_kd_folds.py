"""
run_kd_folds.py
----------------
Runs the best KD config (T=1.0, alpha=0.7, from the fold-1 ablation) across
folds 2-5 to get a 5-fold mean/std for the paper's KD vs. teacher comparison.

Each train_kd.py run OVERWRITES results/kd_results.csv (only fold-1 keeps the
full T×alpha grid, backed up separately), so this script captures each fold's
3 rows (teacher, baseline, distilled_T1.0_a0.7) right after it finishes and
appends them into results/kd_results_5fold.csv. At the end, the full fold-1
grid is restored to results/kd_results.csv so downstream figure/table scripts
keep working off the complete ablation.

Usage: python run_kd_folds.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

RESULTS = Path("results")
GRID_BACKUP = RESULTS / "kd_results_fold1_grid_backup.csv"
MULTIFOLD = RESULTS / "kd_results_5fold.csv"


def main() -> None:
    if not GRID_BACKUP.exists():
        print(f"ERROR: {GRID_BACKUP} not found — back up fold-1 grid first.")
        sys.exit(1)

    grid = pd.read_csv(GRID_BACKUP)
    best = grid[(grid.get("temperature") == 1.0) & (grid.get("alpha") == 0.7)]
    fold1_rows = pd.concat([
        grid[grid["model"].isin(["teacher", "baseline"])],
        best,
    ])
    master_rows = [fold1_rows]
    print(f"Fold 1 (from grid backup): {len(fold1_rows)} rows")

    for fold in [2, 3, 4, 5]:
        print(f"\n{'='*60}\nFold {fold}: training best KD config (T=1.0, alpha=0.7)\n{'='*60}")
        cmd = [sys.executable, "train_kd.py", "--config", "configs/kd_best.yaml", "--fold", str(fold)]
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"[FAIL] fold {fold} — exit code {result.returncode}")
            continue
        fold_df = pd.read_csv(RESULTS / "kd_results.csv")
        master_rows.append(fold_df)
        combined = pd.concat(master_rows, ignore_index=True)
        combined.to_csv(MULTIFOLD, index=False)
        print(f"[DONE] fold {fold} — appended {len(fold_df)} rows to {MULTIFOLD}")

    # Restore the full fold-1 ablation grid as the canonical kd_results.csv
    grid.to_csv(RESULTS / "kd_results.csv", index=False)
    print(f"\nRestored fold-1 full grid to {RESULTS / 'kd_results.csv'}")

    final = pd.read_csv(MULTIFOLD)
    print(f"\n{'='*60}\n5-FOLD KD SUMMARY (best config T=1.0, alpha=0.7)\n{'='*60}")
    for model_type, label in [("teacher", "Teacher (EfficientNet-B0)"),
                               ("baseline", "Baseline student (no KD)"),
                               ("distilled_T1.0_a0.7", "KD student (T=1,a=0.7)")]:
        sub = final[final["model"] == model_type]
        if len(sub):
            print(f"{label}: F1={sub['f1'].mean():.4f} ± {sub['f1'].std():.4f}  "
                  f"(n={len(sub)})  AUC={sub['roc_auc'].mean():.4f} ± {sub['roc_auc'].std():.4f}")


if __name__ == "__main__":
    main()
