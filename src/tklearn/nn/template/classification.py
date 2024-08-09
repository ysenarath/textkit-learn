from __future__ import annotations

from typing import Any

import torch
from typing_extensions import Protocol, runtime_checkable

from tklearn.nn.base import Module
from tklearn.nn.template.loss import BatchPrototypeLoss


@runtime_checkable
class OutputWithPooling(Protocol):
    pooler_output: torch.Tensor


class TemplateBasedClassifier(Module):
    def __init__(self, backbone: Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.loss_func = BatchPrototypeLoss()

    def predict_on_batch(self, batch: Any) -> OutputWithPooling:
        pooler_output = self.backbone.predict_on_batch(batch)
        return pooler_output

    def compute_loss(self, batch: Any, output: OutputWithPooling, **kwargs):
        y_true = batch.y or batch.x["labels"]
        if y_true is None:
            raise ValueError("no labels found in the batch")
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true)
        if y_true.dim() > 1:
            # convert to long format if one-hot encoded
            y_true = y_true.argmax(dim=1)
        # convert to long tensor
        targets = y_true.to(torch.long)
        return self.loss_func(output.pooler_output, targets)
