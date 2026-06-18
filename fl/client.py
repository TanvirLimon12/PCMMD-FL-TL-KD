"""
fl/client.py
------------
FLClient: one patient node. Local SGD training, returns state_dict + stats.
FedProxClient: adds proximal term mu/2 * ||w - w_global||^2.
"""
from __future__ import annotations
import copy
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FLClient:
    """
    Parameters
    ----------
    model      : deep copy of global model for this round
    dataloader : patient's local DataLoader (train split only)
    criterion  : loss function
    optimizer  : local optimizer
    device     : torch.device
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> None:
        self.model      = model.to(device)
        self.dataloader = dataloader
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.device     = device

    def train(self, epochs: int = 3) -> Dict[str, Any]:
        """
        Returns
        -------
        dict with keys:
            state_dict  : CPU state dict
            loss        : avg task loss over all local batches
            num_samples : total samples seen (used for weighted FedAvg)
        """
        self.model.train()
        running_loss  = 0.0
        total_samples = 0

        for _ in range(epochs):
            for batch in self.dataloader:
                # Unpack safely — dataset may return (img, lbl) or (img, lbl, pid, path)
                images, labels = batch[0], batch[1]
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss    = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss  += loss.item() * images.size(0)
                total_samples += images.size(0)

        avg_loss = running_loss / max(1, total_samples)
        return {
            "state_dict" : {k: v.cpu() for k, v in self.model.state_dict().items()},
            "loss"       : avg_loss,
            "num_samples": total_samples,
        }


class FedProxClient(FLClient):
    """
    FedProx: adds proximal regulariser  mu/2 * ||w - w_global||^2
    to pull local weights toward global model — stabilises non-IID training.
    """

    def __init__(self, *args: Any, mu: float = 0.01, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.mu = mu

    def train(self, epochs: int = 3) -> Dict[str, Any]:  # type: ignore[override]
        self.model.train()

        # Snapshot global weights ONCE before any local update
        global_weights = {
            name: param.clone().detach().to(self.device)
            for name, param in self.model.named_parameters()
        }

        running_loss  = 0.0
        total_samples = 0

        for _ in range(epochs):
            for batch in self.dataloader:
                images, labels = batch[0], batch[1]
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                outputs   = self.model(images)
                task_loss = self.criterion(outputs, labels)

                # Proximal term: mu/2 * ||w - w_global||^2
                prox = torch.tensor(0.0, device=self.device)
                for name, param in self.model.named_parameters():
                    prox = prox + (param - global_weights[name]).norm(2) ** 2
                prox_loss = (self.mu / 2.0) * prox

                loss = task_loss + prox_loss
                loss.backward()
                self.optimizer.step()

                # Log task loss only (not prox) for fair comparison with FedAvg logs
                running_loss  += task_loss.item() * images.size(0)
                total_samples += images.size(0)

        avg_loss = running_loss / max(1, total_samples)
        return {
            "state_dict" : {k: v.cpu() for k, v in self.model.state_dict().items()},
            "loss"       : avg_loss,
            "num_samples": total_samples,
        }
