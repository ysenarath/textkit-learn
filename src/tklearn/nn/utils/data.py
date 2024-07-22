from __future__ import annotations

import operator
import random
from typing import (
    Any,
    Callable,
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
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import IterableDataset as IterableBaseDataset
from torch.utils.data import default_collate as _default_collate
from typing_extensions import Self, TypedDict

from tklearn.utils.array import (
    MovableToDeviceMixin,
    get_index,
    length_of_first_array_like_in_nested_dict,
    move_to_device,
    to_numpy,
)

__all__ = [
    "TorchDataset",
    "IterableTorchDataset",
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

    @property
    def iloc(self) -> ILocRecordBatch:
        return ILocRecordBatch(self)

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

    def to(self, device: Union[str, torch.device], non_blocking: bool = False) -> Self:
        x = move_to_device(self.x, device, non_blocking=non_blocking)
        if self.y is None:
            return RecordBatch(x, index=self.index)
        y = move_to_device(self.y, device, non_blocking=non_blocking)
        return RecordBatch(x, y, index=self.index)


class TorchDataset(BaseDataset, Generic[XT, YT]):
    def __new__(
        cls,
        x: Sequence[XT],
        y: Optional[Sequence[YT]] = None,
        /,
        iterable: bool = False,
    ) -> Self:
        if isinstance(x, TorchDataset):
            if y is not None:
                msg = "y should be None when x is a Dataset"
                raise ValueError(msg)
            return x
        if iterable:
            return IterableTorchDataset(x, y)
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


class IterableTorchDataset(TorchDataset, IterableBaseDataset):
    def __iter__(self) -> Generator[Record, None, None]:
        yield from (self[i] for i in range(len(self)))


class BaseCollator:
    def __call__(self, batch: Sequence[Record[XT, YT]]) -> RecordBatch[XT, YT]:
        raise NotImplementedError


class DefaultCollator(BaseCollator):
    def __init__(self, collate_fn: Optional[Callable] = None) -> None:
        if collate_fn is None:
            collate_fn = _default_collate
        self.collate_fn = collate_fn

    def __call__(self, batch: Sequence[Record[XT, YT]]) -> RecordBatch[XT, YT]:
        """
        * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
        * NumPy Arrays -> :class:`torch.Tensor`
        * `float` -> :class:`torch.Tensor`
        * `int` -> :class:`torch.Tensor`
        * `str` -> `str` (unchanged)
        * `bytes` -> `bytes` (unchanged)
        * `Mapping[K, V_i]` -> `Mapping[K, default_collate([V_1, V_2, ...])]`
        * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[default_collate([V1_1, V1_2, ...]),
            default_collate([V2_1, V2_2, ...]), ...]`
        * `Sequence[V1_i, V2_i, ...]` -> `Sequence[default_collate([V1_1, V1_2, ...]),
            default_collate([V2_1, V2_2, ...]), ...]`
        """
        batch: dict = self.collate_fn(batch)
        return RecordBatch(
            batch["x"], batch["y"] if "y" in batch else None, index=batch.get("index")
        )


default_collate = DefaultCollator()


class AugmentedCollator(DefaultCollator):
    def __init__(self, collate_fn: Optional[Callable] = None) -> None:
        super().__init__(collate_fn)

    def __call__(self, batch: Sequence[Record[XT, YT]]) -> RecordBatch[XT, YT]:
        batch = super().__call__(batch)
        return self.generate(batch)

    def generate(self, batch: RecordBatch[XT, YT]) -> RecordBatch[XT, YT]:
        raise NotImplementedError


class PromptAugmentedCollator(AugmentedCollator):
    def __init__(self, tokenizer, id2label, sep_token=None, k=None):
        super().__init__()
        if sep_token is None:
            if hasattr(tokenizer, "sep_token"):
                sep_token = tokenizer.sep_token
            else:
                sep_token = "[SEP]"
        self.tokenizer = tokenizer
        self.sep_token = sep_token
        self.id2label = id2label
        self.ids = {cls: list(set(id2label) - {cls}) for cls in id2label}
        self.k = k

    def generate(self, batch: RecordBatch) -> RecordBatch:
        prompts = []
        labels = []
        indexes = []
        args = batch.x.pop("text"), batch.x.pop("labels", batch.y), batch.index
        for txt, true_lbl, idx in zip(*args):
            if isinstance(true_lbl, torch.Tensor):
                true_lbl = true_lbl.item()
            other_labels = self.ids[true_lbl]
            if self.k is not None:
                other_labels = random.choices(self.ids[true_lbl], k=self.k)
            aug_lbls = [true_lbl] + other_labels
            for lbl in aug_lbls:
                cls = self.id2label[true_lbl]
                prompt = f"{txt} {self.sep_token} {cls}"
                prompts.append(prompt)
                labels.append(true_lbl == lbl)
                indexes.append(idx)
        tokens = self.tokenizer(
            prompts, return_tensors="pt", padding="max_length", truncation=True
        )
        labels = torch.tensor(labels, dtype=torch.long)
        if batch.y:
            return RecordBatch(dict(tokens), labels, index=indexes)
        return RecordBatch({**tokens, "labels": labels}, index=indexes)
