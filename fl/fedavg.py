"""
fl/fedavg.py
------------
FedAvg aggregation: weighted average by number of local samples.
"""
from __future__ import annotations
from typing import Dict, List
import torch
import torch.nn as nn


def aggregate_fedavg(
    global_model: nn.Module,
    client_weights_list: List[Dict[str, torch.Tensor]],
    client_data_sizes: List[int],
) -> nn.Module:
    """
    Weighted average of client state_dicts, proportional to dataset size.
    Modifies global_model in-place and also returns it.
    """
    if not client_weights_list:
        raise ValueError("aggregate_fedavg: empty client_weights_list")

    total_samples = sum(client_data_sizes)
    if total_samples == 0:
        raise ValueError("aggregate_fedavg: total sample count is 0")

    aggregated: Dict[str, torch.Tensor] = {
        k: torch.zeros_like(v, dtype=torch.float32)
        for k, v in global_model.state_dict().items()
    }

    for weights, n in zip(client_weights_list, client_data_sizes):
        factor = n / total_samples
        for k in aggregated:
            aggregated[k] += weights[k].float() * factor

    global_model.load_state_dict(aggregated)
    return global_model
