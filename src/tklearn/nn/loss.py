from __future__ import annotations

from typing import Any, Generic, Mapping, TypeVar, Union

import torch
from typing_extensions import Self

from tklearn.utils import logging

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
        if other is None:
            new_components = dict(self.components)
        elif isinstance(other, Mapping):
            new_components = {}
            for key in set(self.components.keys()).union(other.keys()):
                if key in self.components and key in other:
                    new_components[key] = self.components[key] + other[key]
                elif key in self.components:
                    new_components[key] = self.components[key]
                else:
                    new_components[key] = other[key]
        else:
            raise NotImplementedError
        return self.__class__(new_components)

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
    def from_loss(cls, loss: Union[dict, T]) -> LossDict[T]:
        return cls(loss if isinstance(loss, dict) else {"loss": loss})

    def item(self) -> LossDict[float]:
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
