from __future__ import annotations

from typing import Any, Generic, Mapping, TypeVar, Union

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing_extensions import Self

from octoflow import logging
from tklearn.utils.targets import TargetType, type_of_target

__all__ = [
    "LossDict",
]

T = TypeVar("T", torch.Tensor, float)

logger = logging.get_logger(__name__)


class LossDict(Mapping[str, T], Generic[T]):
    def __init__(self, *args, **kwargs):
        super().__init__()
        losses: Mapping[str, T] = {}
        for key, value in dict(*args, **kwargs).items():
            if value is None:
                msg = f"loss for '{key}' is None"
                logger.warn(msg)
                continue
            losses[key] = value
        self.losses = losses

    def backward(self) -> None:
        sum(self.losses.values()).backward()

    def __add__(self, other: Any) -> Self:
        if isinstance(other, Mapping):
            losses = {}
            for key in set(self.keys()).union(other.keys()):
                if key in self.losses and key in other:
                    losses[key] = self.losses[key] + other[key]
                elif key in self.losses:
                    losses[key] = self.losses[key]
                else:
                    losses[key] = other[key]
        elif other is None:
            losses = self.losses
        else:
            raise NotImplementedError
        return self.__class__(losses)

    def __truediv__(self, other: Any) -> Self:
        losses = {}
        if isinstance(other, Mapping):
            raise NotImplementedError
        for key, value in self.losses.items():
            losses[key] = value / other
        return self.__class__(losses)

    def __getitem__(self, __key: str) -> torch.Tensor:
        return self.losses[__key]

    def __iter__(self):
        return iter(self.losses)

    def __len__(self):
        return len(self.losses)

    def __contains__(self, __key: str) -> bool:
        return __key in self.losses

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.losses})"

    @classmethod
    def from_dict(cls, loss: Union[Mapping[str, T], T]) -> LossDict[T]:
        if not isinstance(loss, Mapping):
            loss = {"loss": loss}
        return cls(loss)

    def detach(self) -> LossDict[T]:
        """Detach all tensors."""
        c = {}
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                value = value.detach()
            c[key] = value
        return LossDict(c)

    def to(
        self, device: Union[torch.device, str], non_blocking: bool = False
    ) -> LossDict[T]:
        """Move all tensors to device."""
        c = {}
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                value = value.to(device, non_blocking=non_blocking)
            c[key] = value
        return LossDict(c)

    def cuda(self, non_blocking: bool = False) -> LossDict[T]:
        """Move all tensors to cuda."""
        return self.to("cuda", non_blocking=non_blocking)

    def cpu(self) -> LossDict[T]:
        """Move all tensors to cpu."""
        return self.to("cpu")

    def item(self) -> LossDict[float]:
        """Convert all tensors to float."""
        c = {}
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            c[key] = value
        return LossDict(c)

    def to_dict(self) -> dict:
        return dict(self)


class TargetLossFunction(torch.nn.Module):
    def __init__(self, target_type: Union[str, TargetType], num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        if isinstance(target_type, str):
            target_type = type_of_target(target_type)
        self.target_type = target_type
        self._loss_func = None

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.target_type is None:
            if self.num_labels == 1:
                target_type = "continuous"
            elif self.num_labels > 1 and (
                target.dtype == torch.long or target.dtype == torch.int
            ):
                target_type = "multiclass"
            else:
                target_type = "multilabel-indicator"
            self.target_type = target_type
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
                raise ValueError(
                    f"loss function for '{self.target_type.label}' not found"
                )
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
                input.view(-1, self.num_labels),
                target.view(-1),
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
