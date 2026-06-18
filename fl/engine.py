"""
fl/engine.py
------------
Shared evaluation helpers for the FedAvg / FedProx drivers so both report the
same numbers (val loss for early-round monitoring, per-client metrics, etc.).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_with_loss(model, loader, criterion, device) -> Dict[str, float]:
    """Full metric suite + mean loss over a loader (loader may return meta)."""
    from utils import compute_all_metrics  # local import avoids cycles
    model.eval()
    ys, ps, probs, running, total = [], [], [], 0.0, 0
    for batch in loader:
        imgs = batch[0].to(device)
        lbls = batch[1].to(device)
        logits = model(imgs)
        running += criterion(logits, lbls).item() * imgs.size(0)
        total += imgs.size(0)
        prob = F.softmax(logits, dim=1)[:, 0]      # P(plasma)
        ys.extend(lbls.cpu().numpy().tolist())
        ps.extend(logits.argmax(1).cpu().numpy().tolist())
        probs.extend(prob.cpu().numpy().tolist())
    m = compute_all_metrics(np.asarray(ys), np.asarray(ps), np.asarray(probs))
    m["loss"] = running / max(1, total)
    return m


@torch.no_grad()
def per_client_metrics(model, client_loaders: Dict[str, object], device,
                       diag_map: Dict[str, str] | None = None) -> pd.DataFrame:
    """
    Evaluate the global model separately on each client's data (Tanjid §T7).
    Returns patient_id, diagnosis, n, precision, recall, f1, plasma_recall.
    """
    from utils import compute_all_metrics
    rows: List[dict] = []
    model.eval()
    for cid, loader in client_loaders.items():
        ys, ps, probs = [], [], []
        for batch in loader:
            imgs = batch[0].to(device)
            lbls = batch[1]
            logits = model(imgs)
            ys.extend(lbls.numpy().tolist())
            ps.extend(logits.argmax(1).cpu().numpy().tolist())
            probs.extend(F.softmax(logits, dim=1)[:, 0].cpu().numpy().tolist())
        if not ys:
            continue
        m = compute_all_metrics(np.asarray(ys), np.asarray(ps), np.asarray(probs))
        rows.append({"patient_id": str(cid),
                     "diagnosis": (diag_map or {}).get(str(cid), "unknown"),
                     "n": len(ys),
                     "precision": round(m["precision"], 5),
                     "recall": round(m["recall"], 5),
                     "f1": round(m["f1"], 5),
                     "plasma_recall": round(m["sensitivity"], 5)})
    return pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)


def rounds_to_best(round_logs: List[dict], metric: str = "f1") -> int:
    """Number of rounds to reach the best value of `metric`."""
    if not round_logs:
        return 0
    best_i = int(np.argmax([r.get(metric, float("-inf")) for r in round_logs]))
    return int(round_logs[best_i].get("round", best_i + 1))
