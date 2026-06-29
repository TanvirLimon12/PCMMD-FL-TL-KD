"""
run_fl_folds.py
---------------
Runs FedAvg, FedProx, and FedBN for all 5 folds (IID + non-IID).
After all folds complete, updates statistical_analysis.csv with full FL stats.

Usage:
  python run_fl_folds.py                      # all methods, folds 1-5, both distributions
  python run_fl_folds.py --folds 2 3 4 5      # skip fold 1 (already done)
  python run_fl_folds.py --method fedavg      # only FedAvg
  python run_fl_folds.py --dist noniid        # only non-IID

Progress is logged to results/logs/run_fl_folds.log.
Already-completed (fold, method, dist) rows in CSVs are skipped automatically
because each training script uses append_csv with drop_duplicates.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


METHODS = {
    "fedavg":   ("train_fedavg.py",   "configs/fedavg.yaml"),
    "fedprox":  ("train_fedprox.py",  "configs/fedprox.yaml"),
    "fedbn":    ("train_fedbn.py",    "configs/fedbn.yaml"),
}

DISTRIBUTIONS = ["non-IID", "iid"]


def _done(results_csv: str, method: str, dist: str, fold: int) -> bool:
    """Return True if this (method, dist, fold) row already exists in CSV."""
    import pandas as pd
    p = Path(results_csv)
    if not p.exists():
        return False
    df = pd.read_csv(p)
    dist_label = "iid" if dist.lower() == "iid" else "noniid"
    mask = (df.get("method", "") == method) & \
           (df.get("distribution", "") == dist_label) & \
           (df.get("fold", -1) == fold)
    return bool(mask.any())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(1, 6)))
    ap.add_argument("--method", choices=list(METHODS) + ["all"], default="all")
    ap.add_argument("--dist",   choices=["iid", "noniid", "non-IID", "both"], default="both")
    ap.add_argument("--dry_run", action="store_true", help="Print commands without running")
    args = ap.parse_args()

    methods = list(METHODS) if args.method == "all" else [args.method]
    dists = DISTRIBUTIONS if args.dist in ("both", "all") else [args.dist]

    total = len(methods) * len(dists) * len(args.folds)
    done_count = 0
    skipped = 0
    t_start = time.time()

    print(f"FL sweep: {len(methods)} methods × {len(dists)} dists × {len(args.folds)} folds = {total} runs")
    print(f"Folds: {args.folds} | Methods: {methods} | Distributions: {dists}\n")

    for fold in args.folds:
        for method in methods:
            script, config = METHODS[method]
            # results CSV to check for existing rows
            results_csv = f"results/{method}_results.csv"
            for dist in dists:
                dist_label = "iid" if dist.lower() == "iid" else "noniid"
                run_id = f"{method}|{dist_label}|fold{fold}"

                if _done(results_csv, method, dist_label, fold):
                    print(f"[SKIP] {run_id} — already in {results_csv}")
                    skipped += 1
                    continue

                cmd = [
                    sys.executable, script,
                    "--config", config,
                    "--distribution", dist,
                    "--fold", str(fold),
                ]
                print(f"[RUN ] {run_id}")
                print(f"       {' '.join(cmd)}")
                if args.dry_run:
                    continue

                t0 = time.time()
                result = subprocess.run(cmd, capture_output=False)
                elapsed = time.time() - t0
                if result.returncode != 0:
                    print(f"[FAIL] {run_id} — exit code {result.returncode} after {elapsed:.0f}s")
                else:
                    done_count += 1
                    remaining = total - done_count - skipped
                    eta = (time.time() - t_start) / max(1, done_count) * remaining
                    print(f"[DONE] {run_id} — {elapsed:.0f}s | ETA {eta/60:.1f} min\n")

    elapsed_total = time.time() - t_start
    print(f"\nComplete. Ran {done_count} | Skipped {skipped} | Total {elapsed_total/60:.1f} min")
    print("\nNext: python statistical_analysis.py --config configs/centralized.yaml")


if __name__ == "__main__":
    main()
