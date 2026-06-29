"""
fl/fedbn.py
-----------
FedBN aggregation: aggregate all parameters EXCEPT batch-norm running stats.
Each client keeps its own BN statistics, preventing domain shift from averaging
statistics across heterogeneous clients.

Reference: Li et al. (2021) FedBN: Federated Learning on Non-IID Features via
Local Batch Normalization. ICLR 2021.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

# Keys in state_dict that belong to BatchNorm tracking — kept local, not aggregated
_BN_LOCAL_KEYS = ("running_mean", "running_var", "num_batches_tracked")


def _is_bn_local(key: str) -> bool:
    return any(key.endswith(k) for k in _BN_LOCAL_KEYS)


def aggregate_fedbn(
    global_model: nn.Module,
    client_weights_list: List[Dict[str, torch.Tensor]],
    client_data_sizes: List[int],
) -> nn.Module:
    """
    Weighted average of client state_dicts, excluding BN running statistics.
    BN running stats are left as they are in global_model (or set from first client).
    Modifies global_model in-place and returns it.
    """
    if not client_weights_list:
        raise ValueError("aggregate_fedbn: empty client_weights_list")

    total_samples = sum(client_data_sizes)
    if total_samples == 0:
        raise ValueError("aggregate_fedbn: total sample count is 0")

    global_sd = global_model.state_dict()

    # Aggregate on CPU to avoid device mismatch (MPS ↔ CPU) across clients
    aggregated: Dict[str, torch.Tensor] = {}
    for k, v in global_sd.items():
        if _is_bn_local(k):
            aggregated[k] = v.clone().cpu()
        else:
            aggregated[k] = torch.zeros_like(v.cpu(), dtype=torch.float32)

    for weights, n in zip(client_weights_list, client_data_sizes):
        factor = n / total_samples
        for k in aggregated:
            if not _is_bn_local(k):
                aggregated[k] += weights[k].float().cpu() * factor

    global_model.load_state_dict(aggregated)
    return global_model
