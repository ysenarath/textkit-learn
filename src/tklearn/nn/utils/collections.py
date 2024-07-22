from __future__ import annotations

from typing import Any, Generic, Mapping, TypeVar, Union

import torch
from typing_extensions import Self

from octoflow import logging

__all__ = [
    "TensorDict",
]

T = TypeVar("T", torch.Tensor, float)

logger = logging.get_logger(__name__)


class TensorDict(Mapping[str, T], Generic[T]):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) > 1:
            raise ValueError(f"dict expected at most 1 argument, got {len(args)}")
        if len(args) > 0:
            if not isinstance(args[0], Mapping):
                args = {"arg": args[0]}
            else:
                args = args[0]
            kwargs.update(args)
        data: Mapping[str, T] = {}
        for key, value in dict(**kwargs).items():
            if value is None:
                continue
            data[key] = value
        self.data = data

    def backward(self) -> None:
        sum(self.data.values()).backward()

    def __add__(self, other: Any) -> Self:
        if isinstance(other, Mapping):
            data = {}
            for key in set(self.keys()).union(other.keys()):
                if key in self.data and key in other:
                    data[key] = self.data[key] + other[key]
                elif key in self.data:
                    data[key] = self.data[key]
                else:
                    data[key] = other[key]
        elif other is None:
            data = self.data
        else:
            raise NotImplementedError
        return self.__class__(data)

    def __truediv__(self, other: Any) -> Self:
        losses = {}
        if isinstance(other, Mapping):
            raise NotImplementedError
        for key, value in self.data.items():
            losses[key] = value / other
        return self.__class__(losses)

    def __getitem__(self, __key: str) -> torch.Tensor:
        return self.data[__key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __contains__(self, __key: str) -> bool:
        return __key in self.data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.data})"

    def detach(self) -> TensorDict[T]:
        """Detach all tensors."""
        c = {}
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                value = value.detach()
            c[key] = value
        return TensorDict(c)

    def to(
        self, device: Union[torch.device, str], non_blocking: bool = False
    ) -> TensorDict[T]:
        """Move all tensors to device."""
        c = {}
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                value = value.to(device, non_blocking=non_blocking)
            c[key] = value
        return TensorDict(c)

    def cuda(self, non_blocking: bool = False) -> TensorDict[T]:
        """Move all tensors to cuda."""
        return self.to("cuda", non_blocking=non_blocking)

    def cpu(self) -> TensorDict[T]:
        """Move all tensors to cpu."""
        return self.to("cpu")

    def item(self) -> TensorDict[float]:
        """Convert all tensors to float."""
        c = {}
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            c[key] = value
        return TensorDict(c)

    def to_dict(self) -> dict:
        return dict(self)
