"""
data/dataset.py
---------------
PCMMDDataset — loads cropped plasma cells from frozen fold CSV manifests.

Image resolution (in priority order):
  1. image_root given  -> resolve by *basename* against a cached filename index
                          (matches the EDA pipeline, which stores absolute EDA-machine
                          paths that DON'T exist on Kaggle — only the basenames do).
  2. row["path"] is absolute and exists.
  3. root_dir / row["path"].
  4. root_dir / patient_cells / <patient_id> / <basename>.

Target label = "label" column ∈ {plasma, non_plasma}.
NEVER use "patient_diagnosis" (patient-level mm/normal) as the target — it leaks
patient identity (constant per patient). Guarded here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ── Label mapping (fixed; plasma=positive=0) ──────────────────────────────────
CLASS_TO_IDX: Dict[str, int] = {"plasma": 0, "non_plasma": 1}
IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}
NUM_CLASSES = 2

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_INDEX_CACHE: Dict[str, Dict[str, Path]] = {}


def build_image_index(image_root: str | Path) -> Dict[str, Path]:
    """Walk image_root ONCE (cached) and map basename -> full Path."""
    key = str(Path(image_root).resolve())
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]
    root = Path(image_root)
    index: Dict[str, Path] = {}
    if root.exists():
        for fp in root.rglob("*"):
            if fp.suffix.lower() in _IMAGE_EXTS:
                index[fp.name] = fp  # last path wins on duplicate names
    _INDEX_CACHE[key] = index
    return index


class PCMMDDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        split: Optional[str] = "train",
        transform: Optional[Callable] = None,
        root_dir: Optional[str | Path] = None,
        image_root: Optional[str | Path] = None,
        label_col: str = "label",
        return_meta: bool = False,
        keep_patients: Optional[set] = None,
        exclude_md5: Optional[set] = None,
    ) -> None:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Fold CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self._validate_columns(label_col)

        split_col = "role" if "role" in self.df.columns else "split"
        if split is not None and split_col in self.df.columns:
            self.df = self.df[self.df[split_col] == split].reset_index(drop=True)

        # Patient-aware sub-filter (used to carve a leakage-free val set from train)
        if keep_patients is not None and "patient_id" in self.df.columns:
            keep = {str(p) for p in keep_patients}
            self.df = self.df[self.df["patient_id"].astype(str).isin(keep)].reset_index(drop=True)

        # Anti-leakage filter: drop rows whose image identity (md5, else basename)
        # appears in a held-out set (e.g. few-shot pool rows that collide with test).
        if exclude_md5:
            ex = {str(v) for v in exclude_md5}
            if "md5" in self.df.columns and self.df["md5"].notna().any():
                self.df = self.df[~self.df["md5"].astype(str).isin(ex)].reset_index(drop=True)
            else:
                base = self.df["path"].map(lambda p: Path(str(p)).name)
                self.df = self.df[~base.isin(ex)].reset_index(drop=True)

        valid_mask = self.df[label_col].isin(CLASS_TO_IDX)
        dropped = int((~valid_mask).sum())
        if dropped:
            print(f"[PCMMDDataset] Dropped {dropped} rows with unknown label.")
        self.df = self.df[valid_mask].reset_index(drop=True)

        self.root_dir = Path(root_dir) if root_dir else csv_path.parent
        self.transform = transform
        self.label_col = label_col
        self.return_meta = return_meta
        self.image_index = build_image_index(image_root) if image_root else None
        self._labels: List[int] = [CLASS_TO_IDX[l] for l in self.df[label_col]]

    def identity_keys(self) -> set:
        """Image-identity keys (md5 if present, else basename) — for leakage checks."""
        if "md5" in self.df.columns and self.df["md5"].notna().any():
            return set(self.df["md5"].astype(str))
        return set(self.df["path"].map(lambda p: Path(str(p)).name))

    def _validate_columns(self, label_col: str) -> None:
        required = {"path", label_col}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Fold CSV missing required columns: {missing}")
        if label_col == "patient_diagnosis":
            raise ValueError("patient_diagnosis must NOT be used as the target (leakage).")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, row) -> Path:
        raw = Path(str(row["path"]))
        name = raw.name
        # 1. basename index (Kaggle-safe)
        if self.image_index is not None and name in self.image_index:
            return self.image_index[name]
        # 2. absolute path that exists
        if raw.is_absolute() and raw.exists():
            return raw
        # 3. root_dir / path
        cand = self.root_dir / raw
        if cand.exists():
            return cand
        # 4. patient_cells fallback
        pid = str(row.get("patient_id", ""))
        fb = self.root_dir / "patient_cells" / pid / name
        if fb.exists():
            return fb
        raise FileNotFoundError(
            f"Image not found for basename '{name}'. Tried index/{raw}/{cand}/{fb}. "
            f"Set 'image_root' in the config to the folder holding the raw images."
        )

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        label = self._labels[idx]
        img_path = self._resolve_path(row)
        try:
            img = Image.open(img_path).convert("RGB")
            img = ImageOps.exif_transpose(img)
        except Exception as exc:
            raise RuntimeError(f"Cannot open image at index {idx}: {img_path}") from exc
        if self.transform:
            img = self.transform(img)
        if self.return_meta:
            return img, label, str(row.get("patient_id", "unknown")), str(img_path)
        return img, label

    @property
    def labels(self) -> List[int]:
        return self._labels

    def class_weights(self) -> torch.Tensor:
        counts = self.df[self.label_col].map(CLASS_TO_IDX).value_counts().sort_index()
        total = float(counts.sum())
        return torch.tensor(
            [total / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)],
            dtype=torch.float32,
        )

    def weighted_sampler(self) -> WeightedRandomSampler:
        cls_w = self.class_weights()
        sample_w = torch.tensor([cls_w[l] for l in self._labels], dtype=torch.float32)
        return WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    def patient_subset(self, patient_id: str) -> "PCMMDDataset":
        sub_df = self.df[self.df["patient_id"] == patient_id].reset_index(drop=True)
        if len(sub_df) == 0:
            raise ValueError(f"No rows found for patient_id='{patient_id}'")
        return self._clone_with(sub_df)

    def index_subset(self, indices: List[int]) -> "PCMMDDataset":
        """Subset by row indices (used for IID client partitioning)."""
        sub_df = self.df.iloc[indices].reset_index(drop=True)
        return self._clone_with(sub_df)

    def _clone_with(self, sub_df: pd.DataFrame) -> "PCMMDDataset":
        s = object.__new__(PCMMDDataset)
        s.df = sub_df
        s.root_dir = self.root_dir
        s.transform = self.transform
        s.label_col = self.label_col
        s.return_meta = self.return_meta
        s.image_index = self.image_index
        s._labels = [CLASS_TO_IDX[l] for l in sub_df[self.label_col]]
        return s


def build_loader(
    csv_path: str | Path,
    split: str,
    transform: Callable,
    batch_size: int = 32,
    num_workers: int = 2,
    use_weighted_sampler: bool = False,
    root_dir: Optional[str | Path] = None,
    image_root: Optional[str | Path] = None,
    return_meta: bool = False,
    keep_patients: Optional[set] = None,
    exclude_md5: Optional[set] = None,
    is_train: Optional[bool] = None,
) -> DataLoader:
    ds = PCMMDDataset(
        csv_path=csv_path, split=split, transform=transform,
        root_dir=root_dir, image_root=image_root, return_meta=return_meta,
        keep_patients=keep_patients, exclude_md5=exclude_md5,
    )
    # `is_train` lets callers mark a loader as training even when split != "train"
    # (e.g. a val loader carved from the train split, or a few-shot pool).
    train_mode = is_train if is_train is not None else (split == "train")
    sampler = ds.weighted_sampler() if (use_weighted_sampler and train_mode) else None
    shuffle = train_mode and (sampler is None)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=train_mode,
    )
