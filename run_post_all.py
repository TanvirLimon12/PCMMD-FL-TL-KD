"""
Master post-processing script.
Run once all experiments complete to regenerate all figures, tables, and stats.

Usage:
  python run_post_all.py [--skip-tl-restore] [--skip-fl] [--skip-kd] [--skip-fewshot]
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

RESULTS = Path("results")


def run(cmd: list[str], label: str) -> bool:
    print(f"\n{'='*60}")
    print(f">>> {label}")
    print(f"    {' '.join(cmd)}")
    print(f"{'='*60}")
    ret = subprocess.run(cmd)
    ok = ret.returncode == 0
    if not ok:
        print(f"  [WARN] {label} exited {ret.returncode}")
    return ok


def check(path: Path, label: str) -> bool:
    if not path.exists():
        print(f"  [SKIP] {label}: {path} not found")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tl-restore", action="store_true")
    ap.add_argument("--skip-fl", action="store_true")
    ap.add_argument("--skip-kd", action="store_true")
    ap.add_argument("--skip-fewshot", action="store_true")
    args = ap.parse_args()
    py = sys.executable

    # ── 1. Restore fold-1 checkpoints + fix mobilenet_v3 frozen row ──────────
    if not args.skip_tl_restore:
        run([py, "restore_after_tl.py"], "Restore checkpoints + fix mobilenet_v3 frozen row")
    else:
        print("\n[SKIP] TL restore")

    # ── 2. Calibration comparison (needs correct fold-1 checkpoints) ─────────
    run([py, "generate_calibration_comparison.py"], "Calibration comparison figures")

    # ── 3. t-SNE (needs correct fold-1 checkpoints) ──────────────────────────
    run([py, "generate_tsne.py"], "t-SNE feature visualization")

    # ── 4. FL post-processing ────────────────────────────────────────────────
    if not args.skip_fl:
        if check(RESULTS / "fedavg_results.csv", "FL"):
            run([py, "plot_fl_convergence.py"], "FL convergence curves")
            if check(RESULTS / "fedavg_round_logs.csv", "FL round logs"):
                run([py, "run_rounds_ablation.py"], "Rounds ablation analysis")
                run([py, "plot_communication_analysis.py"], "Communication analysis")
    else:
        print("\n[SKIP] FL post-processing")

    # ── 5. KD post-processing ────────────────────────────────────────────────
    if not args.skip_kd:
        if not check(RESULTS / "kd_results.csv", "KD"):
            print("  Re-run after KD completes")
    else:
        print("\n[SKIP] KD post-processing")

    # ── 6. Few-shot post-processing ──────────────────────────────────────────
    if not args.skip_fewshot:
        check(RESULTS / "fewshot_results.csv", "Few-shot")
    else:
        print("\n[SKIP] Few-shot post-processing")

    # ── 7. Statistical analysis (all available data) ─────────────────────────
    run([py, "statistical_analysis.py"], "Statistical analysis (all data)")

    # ── 8. Error analysis (all folds × backbones) ────────────────────────────
    run([py, "generate_error_analysis_figure.py"], "Error analysis figures")

    # ── 9. Regenerate all figures ─────────────────────────────────────────────
    run([py, "generate_all_figures.py"], "Regenerate all 30+ figures")

    # ── 10. LaTeX tables ──────────────────────────────────────────────────────
    run([py, "generate_latex_tables.py"], "Generate LaTeX tables")

    # ── 11. Paper numbers summary ─────────────────────────────────────────────
    run([py, "compile_paper_numbers.py"], "Compile paper numbers")

    print("\n" + "="*60)
    print("run_post_all.py COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
