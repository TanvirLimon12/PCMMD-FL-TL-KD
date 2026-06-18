"""
utils/common.py
---------------
Reproducibility, device, logging, config snapshots and model-complexity tooling
(parameter count, on-disk size, FLOPs/MACs, inference latency).

These are shared by every train/eval script so behaviour is identical everywhere.
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

SEED = 42  # project-wide fixed seed — DO NOT change

# Defaults merged under every loaded config so scripts never KeyError.
_CONFIG_DEFAULTS: Dict[str, Any] = {
    "data_root": "./",          # base; fold_dir/image_root derived from it if unset
    "fold_dir": None,           # dir holding fold_*.csv (defaults to data_root)
    "image_root": None,         # dir holding raw images (basename index); None -> path/patient_cells
    "root_dir": "./",           # legacy relative-path root
    "folds": [1, 2, 3, 4, 5],   # 5-fold CV by default
    "fold_id": 1,               # single-fold scripts (FL) use this
    "backbone": "resnet50",
    "pretrained": True,
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "patience": 5,              # early-stopping patience (epochs)
    "num_workers": 2,
    "use_weighted_sampler": True,
    "use_weighted_loss": False,
    "results_dir": "results",
    "figures_dir": "figures",
    "ckpt_dir": "checkpoints",
}


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config, merge defaults, and resolve data_root → fold_dir/image_root."""
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    cfg = {**_CONFIG_DEFAULTS, **user}
    if cfg.get("fold_dir") is None:
        cfg["fold_dir"] = cfg["data_root"]
    if cfg.get("image_root") is None:
        cfg["image_root"] = user.get("image_root")  # stays None if not provided
    # normalise: a single fold_id implies folds=[fold_id] unless folds given explicitly
    if "folds" not in user and "fold_id" in user:
        cfg["folds"] = [user["fold_id"]]
    return cfg


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int = SEED, deterministic: bool = True) -> None:
    """Seed python / numpy / torch (+CUDA) for reproducible training."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """CUDA if available, else CPU (Kaggle GPU / local CPU fallback)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(log_path: str | Path, name: str = "pcmmd") -> logging.Logger:
    """Logger that writes to both stdout and a file."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


# ── Config snapshots ──────────────────────────────────────────────────────────
def save_config_snapshot(cfg: Dict[str, Any], out_path: str | Path) -> None:
    """Persist the exact config used for a run (reproducibility audit trail)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    snap = dict(cfg)
    snap["_torch_version"] = torch.__version__
    snap["_cuda"] = torch.cuda.is_available()
    snap["_seed"] = SEED
    with open(out_path, "w") as f:
        json.dump(snap, f, indent=2, default=str)


# ── Model complexity ──────────────────────────────────────────────────────────
def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """On-disk size of float32 weights + buffers, in MB."""
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    n_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return n_bytes / (1024 ** 2)


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    device: torch.device,
    input_size: tuple = (1, 3, 224, 224),
    n_warmup: int = 5,
    n_runs: int = 30,
) -> Dict[str, float]:
    """Mean / std single-sample forward latency in milliseconds."""
    model.eval().to(device)
    x = torch.randn(*input_size, device=device)
    for _ in range(n_warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return {"latency_ms_mean": float(arr.mean()), "latency_ms_std": float(arr.std()),
            "throughput_img_s": float(1000.0 / arr.mean())}


def compute_flops(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> Dict[str, Optional[float]]:
    """
    FLOPs / MACs via `thop` if installed, else returns None values gracefully
    (so the pipeline never crashes when thop is unavailable on Kaggle/CPU).
    """
    try:
        from thop import profile  # type: ignore
        x = torch.randn(*input_size)
        macs, params = profile(model, inputs=(x,), verbose=False)
        return {"macs_g": macs / 1e9, "flops_g": 2 * macs / 1e9, "params_m": params / 1e6}
    except Exception:
        return {"macs_g": None, "flops_g": None, "params_m": None}


def model_complexity_report(
    model: nn.Module,
    device: torch.device,
    input_size: tuple = (1, 3, 224, 224),
) -> Dict[str, Any]:
    """One-call deployment summary used by KD analysis."""
    rep: Dict[str, Any] = {
        "params_total": count_parameters(model),
        "params_total_m": round(count_parameters(model) / 1e6, 3),
        "model_size_mb": round(model_size_mb(model), 3),
    }
    rep.update({k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in compute_flops(model, input_size).items()})
    rep.update({k: round(v, 3) for k, v in measure_latency(model, device, input_size).items()})
    return rep
