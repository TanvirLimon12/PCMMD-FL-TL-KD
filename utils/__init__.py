"""
utils — shared helpers for the PCMMD project.

  common  : seed, device, logging, config, model stats (params/size/FLOPs/latency)
  metrics : full metric suite, calibration (ECE), bootstrap CI, prediction collection
  plots   : confusion / ROC / PR / reliability / efficiency / client / history figures
  losses  : ce | weighted_ce | focal factory
"""
from .common import (
    SEED,
    set_seed,
    get_device,
    setup_logging,
    load_config,
    save_config_snapshot,
    count_parameters,
    model_size_mb,
    measure_latency,
    compute_flops,
    model_complexity_report,
)
from .metrics import (
    POSITIVE_CLASS_IDX,
    collect_predictions,
    compute_all_metrics,
    expected_calibration_error,
    bootstrap_ci,
    summarise_folds,
)
from .plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_reliability_diagram,
    plot_efficiency_performance,
    plot_client_performance,
    plot_training_history,
    plot_comm_vs_accuracy,
    plot_dirichlet_distribution,
    plot_roc_multi,
    plot_pr_multi,
    plot_grouped_bar,
    plot_confusion_matrix_grid,
    plot_radar_chart,
    plot_heatmap,
    plot_stacked_bar,
    plot_line_with_bands,
)
from .losses import FocalLoss, build_loss

__all__ = [
    "SEED", "set_seed", "get_device", "setup_logging", "load_config", "save_config_snapshot",
    "count_parameters", "model_size_mb", "measure_latency", "compute_flops",
    "model_complexity_report",
    "POSITIVE_CLASS_IDX", "collect_predictions", "compute_all_metrics",
    "expected_calibration_error", "bootstrap_ci", "summarise_folds",
    "plot_confusion_matrix", "plot_roc_curve", "plot_pr_curve",
    "plot_reliability_diagram", "plot_efficiency_performance", "plot_client_performance",
    "plot_training_history", "plot_comm_vs_accuracy", "plot_dirichlet_distribution",
    "plot_roc_multi", "plot_pr_multi", "plot_grouped_bar", "plot_confusion_matrix_grid",
    "plot_radar_chart", "plot_heatmap", "plot_stacked_bar", "plot_line_with_bands",
    "FocalLoss", "build_loss",
]
