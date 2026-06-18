"""
data/folds.py
-------------
Loads frozen fold CSV manifests (fold_1.csv … fold_5.csv) and few-shot subsets.
STRICTLY NO random_split / train_test_split — the train/test split is frozen by
the EDA stage and patient-disjoint.

LEAKAGE-SAFE VALIDATION
-----------------------
The frozen folds only carry role ∈ {train, test}. Selecting the best checkpoint
on the *test* fold would be optimistic (test-set model selection). So we carve a
patient-disjoint VALIDATION set out of the TRAIN patients (stratified by
diagnosis, seed 42). Early stopping / checkpoint selection use VAL only; TEST is
touched once, for the final reported number. The same carve is deterministic, so
the centralized, few-shot and federated tracks all share identical val/test sets.

Provides:
  • validate_fold        — column / label / patient-leakage integrity check
  • patient_val_split    — deterministic patient-disjoint train/val patient sets
  • get_fold_loaders     — (train, val, test) DataLoaders for one fold
  • build_client_loaders — FL client partition (patient non-IID | IID), holding out val patients
  • build_val_test_loaders — val (carved) + test loaders for FL global-model selection
  • compute_client_stats — per-client DataFrame for client_stats.csv
  • get_fewshot_loader   — frozen few-shot subset loader with anti-leakage filter vs the test fold
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .dataset import CLASS_TO_IDX, PCMMDDataset, build_loader
from .transforms import get_train_transforms, get_val_transforms

REQUIRED_COLS = {"path", "label"}
VALID_LABELS = set(CLASS_TO_IDX)
DEFAULT_VAL_FRAC = 0.2
SEED = 42


def _split_col(df: pd.DataFrame) -> str:
    return "role" if "role" in df.columns else "split"


def load_fold_df(fold_dir: str | Path, fold_id: int) -> pd.DataFrame:
    path = Path(fold_dir) / f"fold_{fold_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Fold CSV not found: {path}")
    df = pd.read_csv(path)
    missing = (REQUIRED_COLS | {_split_col(df)}) - set(df.columns)
    if missing:
        raise ValueError(f"fold_{fold_id}.csv missing columns: {missing}")
    return df


def patient_val_split(
    train_df: pd.DataFrame,
    val_frac: float = DEFAULT_VAL_FRAC,
    seed: int = SEED,
    diag_col: str = "patient_diagnosis",
) -> Tuple[List[str], List[str]]:
    """
    Deterministic patient-disjoint split of TRAIN patients into (train, val).
    Stratified by diagnosis so val keeps both diseased/healthy patients.
    A diagnosis group with a single patient contributes 0 to val (never empties a class).
    """
    patients = sorted(map(str, train_df["patient_id"].dropna().unique()))
    if val_frac <= 0 or len(patients) < 2:
        return patients, []

    groups: Dict[str, List[str]] = defaultdict(list)
    if diag_col in train_df.columns:
        diag = (train_df.drop_duplicates("patient_id")
                .assign(patient_id=lambda d: d["patient_id"].astype(str))
                .set_index("patient_id")[diag_col])
        for p in patients:
            groups[str(diag.get(p, "unknown"))].append(p)
    else:
        groups["all"] = patients

    val: List[str] = []
    for g in sorted(groups):
        ps = sorted(groups[g])
        rng = np.random.default_rng(seed + hash(g) % 10_000)
        order = rng.permutation(len(ps))
        k = int(round(len(ps) * val_frac))
        if len(ps) > 1:
            k = max(1, k)
        for i in order[:k]:
            val.append(ps[i])
    val_set = set(val)
    train = [p for p in patients if p not in val_set]
    # guard: never let train collapse
    if not train:
        return patients, []
    return sorted(train), sorted(val_set)


def validate_fold(fold_dir: str | Path, fold_id: int, val_frac: float = DEFAULT_VAL_FRAC) -> Dict[str, object]:
    """Integrity report: label sanity, train/test/val patient disjointness, leakage flag."""
    df = load_fold_df(fold_dir, fold_id)
    sc = _split_col(df)

    bad = set(df["label"].dropna().unique()) - VALID_LABELS
    if bad:
        raise ValueError(f"fold_{fold_id}.csv has unexpected labels: {bad}")

    train = df[df[sc] == "train"]
    test = df[df[sc] == "test"]
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"fold_{fold_id}: empty train/test split.")

    leakage: List[str] = []
    tr_pat, val_pat = [], []
    if "patient_id" in df.columns:
        overlap = set(train["patient_id"].dropna().astype(str)) & set(test["patient_id"].dropna().astype(str))
        leakage = sorted(overlap)
        tr_pat, val_pat = patient_val_split(train, val_frac)

    return {
        "fold_id": fold_id,
        "n_train": len(train), "n_test": len(test),
        "train_plasma": int((train["label"] == "plasma").sum()),
        "train_non_plasma": int((train["label"] == "non_plasma").sum()),
        "test_plasma": int((test["label"] == "plasma").sum()),
        "test_non_plasma": int((test["label"] == "non_plasma").sum()),
        "n_patients": int(df["patient_id"].nunique()) if "patient_id" in df else -1,
        "n_train_clients": len(tr_pat), "n_val_patients": len(val_pat),
        "val_patients": val_pat,
        "patient_leakage": leakage,            # [] == clean (train/test patient-disjoint)
    }


def get_fold_loaders(
    fold_dir: str | Path,
    fold_id: int,
    batch_size: int = 32,
    num_workers: int = 2,
    root_dir: Optional[str | Path] = None,
    image_root: Optional[str | Path] = None,
    use_weighted_sampler: bool = True,
    val_frac: float = DEFAULT_VAL_FRAC,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).
    train/val are patient-disjoint slices of the TRAIN split; test is the TEST split.
    val_loader is None only if the fold lacks patient_id or has too few patients.
    """
    csv_path = Path(fold_dir) / f"fold_{fold_id}.csv"
    train_df = load_fold_df(fold_dir, fold_id)
    sc = _split_col(train_df)
    train_only = train_df[train_df[sc] == "train"]

    tr_pat, val_pat = patient_val_split(train_only, val_frac) if "patient_id" in train_df.columns else (None, [])

    train_loader = build_loader(
        csv_path=csv_path, split="train", transform=get_train_transforms(),
        batch_size=batch_size, num_workers=num_workers,
        use_weighted_sampler=use_weighted_sampler, root_dir=root_dir, image_root=image_root,
        keep_patients=set(tr_pat) if tr_pat is not None else None, is_train=True,
    )
    val_loader = None
    if val_pat:
        val_loader = build_loader(
            csv_path=csv_path, split="train", transform=get_val_transforms(),
            batch_size=batch_size, num_workers=num_workers, use_weighted_sampler=False,
            root_dir=root_dir, image_root=image_root, return_meta=True,
            keep_patients=set(val_pat), is_train=False,
        )
    test_loader = build_loader(
        csv_path=csv_path, split="test", transform=get_val_transforms(),
        batch_size=batch_size, num_workers=num_workers, use_weighted_sampler=False,
        root_dir=root_dir, image_root=image_root, return_meta=True,
    )
    return train_loader, val_loader, test_loader


def build_val_test_loaders(
    fold_dir: str | Path,
    fold_id: int,
    batch_size: int = 32,
    num_workers: int = 2,
    root_dir: Optional[str | Path] = None,
    image_root: Optional[str | Path] = None,
    val_frac: float = DEFAULT_VAL_FRAC,
) -> Tuple[Optional[DataLoader], DataLoader, List[str]]:
    """val + test loaders (both return_meta) + the list of held-out val patients."""
    _, val_loader, test_loader = get_fold_loaders(
        fold_dir, fold_id, batch_size, num_workers, root_dir, image_root,
        use_weighted_sampler=False, val_frac=val_frac)
    train_df = load_fold_df(fold_dir, fold_id)
    sc = _split_col(train_df)
    _, val_pat = patient_val_split(train_df[train_df[sc] == "train"], val_frac) \
        if "patient_id" in train_df.columns else (None, [])
    return val_loader, test_loader, val_pat


def build_client_loaders(
    fold_dir: str | Path,
    fold_id: int,
    batch_size: int = 16,
    num_workers: int = 2,
    root_dir: Optional[str | Path] = None,
    image_root: Optional[str | Path] = None,
    partition: str = "patient",        # "patient" (non-IID) | "iid"
    num_clients: Optional[int] = None,
    holdout_patients: Optional[set] = None,   # val patients excluded from federation
    seed: int = SEED,
) -> Dict[str, DataLoader]:
    """
    {client_id: DataLoader} over TRAIN patients (minus holdout_patients).

    partition="patient": one client per patient_id → natural non-IID.
    partition="iid"    : pooled train shuffled into equal shards.
    """
    csv_path = Path(fold_dir) / f"fold_{fold_id}.csv"
    keep = None
    if holdout_patients:
        full = load_fold_df(fold_dir, fold_id)
        sc = _split_col(full)
        all_train_pat = set(full[full[sc] == "train"]["patient_id"].astype(str))
        keep = {p for p in all_train_pat if p not in {str(x) for x in holdout_patients}}

    full_ds = PCMMDDataset(
        csv_path=csv_path, split="train", transform=get_train_transforms(),
        root_dir=root_dir, image_root=image_root, keep_patients=keep)

    loaders: Dict[str, DataLoader] = {}
    if partition == "patient":
        for pid in sorted(full_ds.df["patient_id"].unique().tolist(), key=lambda x: str(x)):
            subset = full_ds.patient_subset(pid)
            loaders[str(pid)] = DataLoader(
                subset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True, drop_last=False)
    elif partition == "iid":
        n = num_clients or full_ds.df["patient_id"].nunique()
        idx = np.arange(len(full_ds))
        np.random.default_rng(seed).shuffle(idx)
        for c, shard in enumerate(np.array_split(idx, n)):
            subset = full_ds.index_subset(shard.tolist())
            loaders[f"client_{c:02d}"] = DataLoader(
                subset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True, drop_last=False)
    else:
        raise ValueError(f"Unknown partition '{partition}'. Use 'patient' or 'iid'.")
    return loaders


def compute_client_stats(fold_dir: str | Path, fold_id: int) -> pd.DataFrame:
    """Per-patient TRAIN-split stats: patient_id, diagnosis, total/plasma/non_plasma, plasma_%."""
    df = load_fold_df(fold_dir, fold_id)
    sc = _split_col(df)
    train = df[df[sc] == "train"]
    diag_col = "patient_diagnosis" if "patient_diagnosis" in train.columns else None
    rows = []
    for pid, g in train.groupby("patient_id"):
        plasma = int((g["label"] == "plasma").sum())
        non_plasma = int((g["label"] == "non_plasma").sum())
        total = plasma + non_plasma
        diag = (g[diag_col].dropna().mode()[0]
                if diag_col and g[diag_col].notna().any() else "unknown")
        rows.append({"patient_id": str(pid), "diagnosis": diag, "total_cells": total,
                     "plasma_cells": plasma, "non_plasma_cells": non_plasma,
                     "plasma_percentage": round(100.0 * plasma / total, 2) if total else 0.0})
    return pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)


def _test_identity_keys(fold_dir: str | Path, fold_id: int) -> set:
    """Image-identity keys (md5 else basename) of the fold TEST split — for leakage filtering."""
    df = load_fold_df(fold_dir, fold_id)
    sc = _split_col(df)
    test = df[df[sc] == "test"]
    if "md5" in test.columns and test["md5"].notna().any():
        return set(test["md5"].astype(str))
    return set(test["path"].map(lambda p: Path(str(p)).name))


def get_fewshot_loader(
    fold_dir: str | Path,
    n_shot: int,
    fold_id: int,
    batch_size: int = 32,
    num_workers: int = 2,
    root_dir: Optional[str | Path] = None,
    image_root: Optional[str | Path] = None,
) -> Tuple[DataLoader, int]:
    """
    Frozen few-shot subset loader (fewshot_5/10/20/50/100.csv).
    ANTI-LEAKAGE: drops any few-shot row whose image identity collides with the
    chosen fold's TEST split, so few-shot training never sees test images.
    Returns (loader, n_dropped).
    """
    csv_path = Path(fold_dir) / f"fewshot_{n_shot}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Few-shot CSV not found: {csv_path}")
    test_keys = _test_identity_keys(fold_dir, fold_id)
    before = len(pd.read_csv(csv_path))
    loader = build_loader(
        csv_path=csv_path, split=None, transform=get_train_transforms(),
        batch_size=batch_size, num_workers=num_workers, use_weighted_sampler=True,
        root_dir=root_dir, image_root=image_root, exclude_md5=test_keys, is_train=True)
    n_dropped = before - len(loader.dataset)
    return loader, n_dropped
