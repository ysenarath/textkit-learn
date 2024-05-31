from __future__ import annotations

from typing import Any, Generic, Mapping, TypeVar, Union

import torch
from typing_extensions import Self

from tklearn.utils import logging

__all__ = [
    "LossDict",
]

T = TypeVar("T", torch.Tensor, float)

logger = logging.get_logger(__name__)


class LossDict(Mapping[str, T], Generic[T]):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.components = {}
        self.loss: T
        for key, value in dict(*args, **kwargs).items():
            if value is None:
                msg = f"loss for '{key}' is None"
                logger.warn(msg)
                continue
            self.components[key] = value
            if getattr(self, "loss", None) is not None:
                self.loss += value
            else:
                self.loss = value

    def __add__(self, other: Any) -> Self:
        if isinstance(other, Mapping):
            components = {}
            for key in set(self.components.keys()).union(other.keys()):
                if key in self.components and key in other:
                    components[key] = self.components[key] + other[key]
                elif key in self.components:
                    components[key] = self.components[key]
                else:
                    components[key] = other[key]
        elif other is None:
            components = self.components
        else:
            raise NotImplementedError
        return self.__class__(components)

    def __truediv__(self, other: Any) -> Self:
        new_components = {}
        if isinstance(other, Mapping):
            raise NotImplementedError
        for key, value in self.components.items():
            new_components[key] = value / other
        return self.__class__(new_components)

    def __getitem__(self, __key: str) -> torch.Tensor:
        return self.components[__key]

    def __iter__(self):
        return iter(self.components)

    def __len__(self):
        return len(self.components)

    def __contains__(self, __key: str) -> bool:
        return __key in self.components

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.components})"

    @classmethod
    def from_loss(cls, loss: Union[Mapping[str, T], T]) -> LossDict[T]:
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
        c = {"loss": self.loss}
        for key, value in self.items():
            c[f"loss_{key}"] = value
        return c
