from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from tklearn.nn.base.module import Module
from tklearn.utils.array import move_to_device


class TemperatureScaling(nn.Module):
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature


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
            for batch in self.validation_loader:
                batch = move_to_device(batch, self.device, non_blocking=True)
                output = self.base_model.predict_step(batch)
                logits = output["logits"]
                labels = (
                    batch[1]
                    if isinstance(batch, (tuple, list))
                    else batch["labels"]
                )
                logits_list.append(logits)
                labels_list.append(labels)
        return torch.cat(logits_list), torch.cat(labels_list)

    def train(self) -> Dict[str, Any]:
        """Train the temperature scaling model."""
        logits, labels = self.collect_logits()

        # Create optimizer and NLL criterion
        optimizer = torch.optim.LBFGS(
            [self.temperature_model.temperature],
            lr=0.01,
            max_iter=self.max_iters,
        )
        nll_criterion = nn.CrossEntropyLoss()

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.temperature_model(logits)
            loss = nll_criterion(scaled_logits, labels)
            loss.backward()
            return loss

        # Train temperature scaling
        optimizer.step(eval_loss)

        # Return training info
        with torch.no_grad():
            final_temp = self.temperature_model.temperature.item()
            final_loss = eval_loss().item()

        return {"temperature": final_temp, "final_nll": final_loss}


class CalibratedModule(Module):
    """Wrapper for a model with temperature scaling calibration."""

    def __init__(
        self, base_model: Module, temperature_model: TemperatureScaling
    ):
        super().__init__()
        self.base_model = base_model
        self.temperature_model = temperature_model

    def forward(self, batch):
        logits = self.base_model(batch)
        return self.temperature_model(logits)

    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        output = self.base_model.predict_step(batch, batch_idx, dataloader_idx)
        if isinstance(output, dict) and "logits" in output:
            output["logits"] = self.temperature_model(output["logits"])
        return output

    def compute_loss(self, batch, output):
        return self.base_model.compute_loss(batch, output)


def calibrate_model(
    model: Module,
    dataloader: DataLoader,
    max_iters: int = 50,
    device: Optional[str] = None,
) -> tuple[CalibratedModule, Dict[str, Any]]:
    """
    Calibrate a model using temperature scaling.

    Returns:
        Tuple of (calibrated model, calibration info)
    """
    temperature_model = TemperatureScaling()
    trainer = TemperatureTrainer(
        base_model=model,
        temperature_model=temperature_model,
        dataloader=dataloader,
        max_iters=max_iters,
        device=device,
    )
    calibration_info = trainer.train()
    calibrated_model = CalibratedModule(model, temperature_model)
    return calibrated_model, calibration_info


# Example usage:
"""
# After training your base model:
calibrated_model, info = calibrate_model(trained_model, validation_loader)
print(f"Optimal temperature: {info['temperature']:.3f}")

# Use calibrated model with existing Predictor/Evaluator
predictor = Predictor(
    model=calibrated_model,
    dataloader=test_loader,
    callbacks=callbacks
)
predictions = predictor.predict()
"""
