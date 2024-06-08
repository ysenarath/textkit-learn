from __future__ import annotations

from typing import Any, Generic, Mapping, TypeVar, Union

import torch
from typing_extensions import Self

from octoflow import logging

__all__ = [
    "LossAccumulator",
]

T = TypeVar("T", torch.Tensor, float)

logger = logging.get_logger(__name__)


class LossAccumulator(Mapping[str, T], Generic[T]):
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
    def from_loss(cls, loss: Union[Mapping[str, T], T]) -> LossAccumulator[T]:
        if not isinstance(loss, Mapping):
            loss = {"loss": loss}
        return cls(loss)

    def detach(self) -> LossAccumulator[T]:
        """Detach all tensors."""
        c = {}
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                value = value.detach()
            c[key] = value
        return LossAccumulator(c)

    def to(
        self, device: Union[torch.device, str], non_blocking: bool = False
    ) -> LossAccumulator[T]:
        """Move all tensors to device."""
        c = {}
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                value = value.to(device, non_blocking=non_blocking)
            c[key] = value
        return LossAccumulator(c)

    def cuda(self, non_blocking: bool = False) -> LossAccumulator[T]:
        """Move all tensors to cuda."""
        return self.to("cuda", non_blocking=non_blocking)

    def cpu(self) -> LossAccumulator[T]:
        """Move all tensors to cpu."""
        return self.to("cpu")

    def item(self) -> LossAccumulator[float]:
        """Convert all tensors to float."""
        c = {}
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            c[key] = value
        return LossAccumulator(c)

    def to_dict(self) -> dict:
        return dict(self)
