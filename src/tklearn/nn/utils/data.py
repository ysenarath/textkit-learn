from __future__ import annotations

import operator
from typing import (
    Any,
    Generator,
    Generic,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import IterableDataset as IterableTorchDataset
from typing_extensions import Self, TypedDict

from tklearn.nn.utils.array import (
    MovableToDeviceMixin,
    get_index,
    length_of_first_array_like_in_nested_dict,
    move_to_device,
    to_numpy,
)

__all__ = [
    "Dataset",
    "IterableDataset",
    "Record",
    "RecordBatch",
]

XT, YT = TypeVar("XT"), TypeVar("YT")


class Record(TypedDict, Generic[XT, YT]):
    x: XT
    y: YT
    index: int


class ILocRecordBatch:
    def __init__(self, ref: RecordBatch) -> None:
        self.ref: RecordBatch = ref

    @overload
    def __getitem__(self, index: int) -> Record[XT, YT]: ...

    @overload
    def __getitem__(self, index: slice) -> RecordBatch[XT, YT]: ...

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, int):
            return self.ref._iloc_int(index)
        if isinstance(index, slice):
            return self.ref._iloc_slice(index)
        msg = f"index must be int or slice, not {type(index).__name__}"
        raise TypeError(msg)


class RecordBatch(tuple, MovableToDeviceMixin, Generic[XT, YT]):
    _fields = ("x", "y")
    x: Sequence[XT] = property(operator.itemgetter(0))
    y: Sequence[YT] = property(operator.itemgetter(1))

    def __new__(
        cls,
        x: Sequence[XT],
        y: Optional[Sequence[YT]] = None,
        /,
        index: Optional[Sequence[int]] = None,
    ) -> Self:
        self = super().__new__(cls, (x, y))
        self._index = np.arange(len(x)) if index is None else to_numpy(index)
        return self

    @property
    def index(self) -> np.ndarray[int]:
        return self._index

    def to(self, device: Union[str, torch.device]) -> Self:
        x = move_to_device(self.x, device)
        if self.y is None:
            return RecordBatch(x, index=self.index)
        y = move_to_device(self.y, device)
        return RecordBatch(x, y, index=self.index)

    def _iloc_int(self, index: int) -> Record[XT, YT]:
        return {
            "x": get_index(self.x, index),
            "y": get_index(self.y, index) if self.y is not None else None,
            "index": self.index[index],
        }

    def _iloc_slice(self, index: slice) -> Self:
        return RecordBatch(
            get_index(self.x, index),
            get_index(self.y, index) if self.y is not None else None,
            index=self.index[index],
        )

    @property
    def iloc(self) -> ILocRecordBatch:
        return ILocRecordBatch(self)


class Dataset(TorchDataset, Generic[XT, YT]):
    def __new__(
        cls,
        x: Sequence[XT],
        y: Optional[Sequence[YT]] = None,
        /,
        iterable: bool = False,
    ) -> Self:
        if iterable:
            return IterableDataset(x, y)
        return super().__new__(cls)

    def __init__(
        self,
        x: Sequence[XT],
        y: Optional[Sequence[YT]] = None,
        /,
        **kargs,
    ) -> None:
        super().__init__()
        if isinstance(x, Tuple):
            if len(x) != 2:  # noqa: PLR2004
                msg = "x must be a tuple of length 2"
                raise ValueError(msg)
            if y is not None:
                msg = "y must be None if x is a tuple of length 2"
                raise ValueError(msg)
            x, y = x
        self._data = (x, y)
        self._length = None

    @property
    def x(self) -> Sequence[XT]:
        return self._data[0]

    @property
    def y(self) -> Sequence[YT]:
        return self._data[1]

    def __getitem__(self, __item: int) -> Record[XT, YT]:
        ix: XT = get_index(self.x, index=__item)
        if self.y is None:
            return {"x": ix, "index": __item}
        iy: YT = get_index(self.y, index=__item)
        return {"x": ix, "y": iy, "index": __item}

    def __len__(self) -> int:
        if self._length is None:
            self._length = length_of_first_array_like_in_nested_dict(self.x)
        return self._length

    def __repr__(self) -> str:
        return repr(self._data)


class IterableDataset(Dataset, IterableTorchDataset):
    def __iter__(self) -> Generator[Record, None, None]:
        yield from (self[i] for i in range(len(self)))
