"""
fl/fedprox.py
-------------
FedProxClient re-exported here for clean import path.
Aggregation reuses aggregate_fedavg — FedProx only differs in local training.
"""
from .client import FedProxClient
from .fedavg import aggregate_fedavg

__all__ = ["FedProxClient", "aggregate_fedavg"]
