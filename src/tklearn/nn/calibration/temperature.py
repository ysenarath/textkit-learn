from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from tklearn.nn.base.module import Module
from tklearn.nn.models.classifier.multiclass import (
    LinearMulticlassClassifier,
)
from tklearn.utils.array import move_to_device


class TemperatureScaling(nn.Module):
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = self.temperature.unsqueeze(1).expand(
            logits.size(0), logits.size(1)
        )
        return logits / temperature


class TemperatureTrainer:
    def __init__(
        self,
        model: Module,
        dataloader: DataLoader,
        max_iters: int = 50,
        device: Optional[str] = None,
    ):
        self.base_model = model
        self.temperature_model = TemperatureScaling()
        self.validation_loader = dataloader
        self.max_iters = max_iters
        self.device = device or model.device
        # Move models to device
        self.base_model.to(self.device)
        self.temperature_model.to(self.device)

    def collect_logits(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Collect all logits and labels from validation set."""
        self.base_model.eval()
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for b, batch in enumerate(self.validation_loader):
                batch = move_to_device(batch, self.device, non_blocking=True)
                output = self.base_model.predict_step(batch, batch_idx=b)
                labels, logits = batch["labels"], output["logits"]
                logits_list.append(logits)
                labels_list.append(labels)
            logits = torch.cat(logits_list)
            labels = torch.cat(labels_list)
        return logits, labels

    def train(self) -> Dict[str, Any]:
        """Train the temperature scaling model."""
        logits, labels = self.collect_logits()

        # Switch to Adam optimizer
        optimizer = torch.optim.Adam(
            self.temperature_model.parameters(), lr=0.01
        )

        # Track initial temperature
        initial_temp = self.temperature_model.temperature.item()

        # Training loop
        for i in range(self.max_iters):
            optimizer.zero_grad()
            scaled_logits = self.temperature_model(logits)
            loss = self.base_model.compute_loss(
                {"labels": labels}, {"logits": scaled_logits}
            )
            loss.backward()
            optimizer.step()

        # Return training info
        final_temp = self.temperature_model.temperature.item()
        final_loss = loss.item()

        return {
            "temperature": final_temp,
            "final_nll": final_loss,
            "initial_temp": initial_temp,
        }


class CalibratedModule(Module):
    """Wrapper for a model with temperature scaling calibration."""

    def __init__(
        self,
        base_model: LinearMulticlassClassifier,
        temperature_model: TemperatureScaling,
        info: Dict[str, Any],
    ):
        super().__init__()
        self.base_model = base_model
        self.temperature_model = temperature_model
        self.info = info

    def forward(self, batch):
        logits = self.base_model(batch)
        return self.temperature_model(logits)

    def compute_loss(self, batch, output):
        return self.base_model.compute_loss(batch, output)

    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        output = self.base_model.predict_step(
            batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx
        )
        output["logits"] = self.temperature_model(output["logits"])
        return output

    def compute_metric_inputs(self, batch, output, **kwargs) -> dict:
        return self.base_model.compute_metric_inputs(batch, output, **kwargs)

    @property
    def temperature(self):
        return self.temperature_model.temperature.item()


def calibrate_model(
    model: Module,
    dataloader: DataLoader,
    max_iters: int = 50,
    device: Optional[str] = None,
) -> CalibratedModule:
    """
    Calibrate a model using temperature scaling.

    Returns
    -------
    tuple
        A tuple containing the calibrated model and calibration info.
    """
    trainer = TemperatureTrainer(
        model=model,
        dataloader=dataloader,
        max_iters=max_iters,
        device=device,
    )
    info = trainer.train()
    calibrated_model = CalibratedModule(model, trainer.temperature_model, info)
    return calibrated_model
