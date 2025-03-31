from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from tklearn.nn.utils.collections import TensorDict
from tklearn.utils.targets import TargetType, type_of_target

__all__ = [
    "LossDict",
    "TargetBasedLoss",
]


class LossDict(TensorDict):
    def __init__(self, *args: Any, **kwargs: Any):
        if len(args) == 1 and not isinstance(args[0], Mapping):
            args = ({"loss": args[0]},)
        super().__init__(*args, **kwargs)


class TargetBasedLoss(torch.nn.Module):
    def __init__(self, target_type: str | TargetType, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        self.target_type = type_of_target(target_type)
        self._loss_func = None

    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if self._loss_func is None:
            if self.target_type.label == "continuous":
                loss_fct = MSELoss()
            elif self.target_type.label == "multiclass":
                # LogSoftmax on an input, followed by NLLLoss
                # - input must be a Tensor of size either (minibatch, C) or
                #   (minibatch, C, d1, d2, ..., dK)
                # - target must be a class index in [0, C-1] (long)
                loss_fct = CrossEntropyLoss()
            elif self.target_type.label in ["binary", "multilabel-indicator"]:
                # Sigmoid layer and the BCELoss
                # - input and target must have same shape
                loss_fct = BCEWithLogitsLoss()
            else:
                msg = (
                    f"target type '{self.target_type.label}' is not supported"
                )
                raise ValueError(msg)
            self._loss_func = loss_fct
        if isinstance(self._loss_func, MSELoss):
            if self.num_labels == 1:
                loss = self._loss_func(
                    input.squeeze(),
                    target.squeeze(),
                )
            else:
                loss = self._loss_func(input, target)
        elif isinstance(self._loss_func, CrossEntropyLoss):
            # >>> # Example of target with class indices
            # >>> loss = nn.CrossEntropyLoss()
            # >>> input = torch.randn(3, 5, requires_grad=True)
            # >>> target = torch.empty(3, dtype=torch.long).random_(5)
            # >>> output = loss(input, target)
            # >>> output.backward()
            # >>>
            # >>> # Example of target with class probabilities
            # >>> input = torch.randn(3, 5, requires_grad=True)
            # >>> target = torch.randn(3, 5).softmax(dim=1)
            # >>> output = loss(input, target)
            # >>> output.backward()
            loss = self._loss_func(
                input.view(-1, self.num_labels), target.view(-1)
            )
        else:  # BCEWithLogitsLoss
            # >>> loss = nn.BCEWithLogitsLoss()
            # >>> input = torch.randn(3, requires_grad=True)
            # >>> target = torch.empty(3).random_(2)
            # >>> output = loss(input, target)
            # >>> output.backward()
            # shape of input and target must be the same
            if len(target.shape) == 1:  # binary case
                # 1 is the number of classes
                target = target.view(-1, 1)
            target = target.to(dtype=input.dtype)
            loss = self._loss_func(input, target)
        return loss
