from __future__ import annotations

from typing import (
    Generator,
    Generic,
    NamedTuple,
    Self,
    TypeVar,
    Union,
)

import torch
from torch.utils.data import Dataset, IterableDataset

from tklearn.utils.array import (
    MovableToDeviceMixin,
    get_index,
    length_of_first_array_like_in_nested_dict,
    move_to_device,
)

__all__ = [
    "IterableTorchDataset",
    "TorchDataset",
]

XT, YT = TypeVar("XT"), TypeVar("YT")


class TorchDataTuple(NamedTuple):
    x: XT
    y: YT = None


class TorchDataset(Dataset, MovableToDeviceMixin, Generic[XT, YT]):
    def __init__(self, x: XT, y: YT) -> None:
        super().__init__()
        self._data = TorchDataTuple(x, y)
        self._length = None
        self._device = None

    @property
    def x(self) -> XT:
        return self._data.x

    @property
    def y(self) -> YT:
        return self._data.y

    def __getitem__(self, __item: int):
        ix = get_index(self.x, index=__item)
        if self.y is None:
            return TorchDataTuple(ix)
        iy = get_index(self.y, index=__item)
        return TorchDataTuple(ix, iy)

    def __len__(self) -> int:
        if self._length is None:
            self._length = length_of_first_array_like_in_nested_dict(self.x)
        return self._length

    def to(self, device: Union[str, torch.device]) -> Self:
        x = move_to_device(self.x, device)
        if self.y is None:
            return TorchDataset(x)
        y = move_to_device(self.y, device)
        return TorchDataset(x, y)

    def __repr__(self) -> str:
        return repr(self._data)


class IterableTorchDataset(TorchDataset, IterableDataset):
    def __iter__(self) -> Generator[dict, None, None]:
        yield from (self[i] for i in range(len(self)))
