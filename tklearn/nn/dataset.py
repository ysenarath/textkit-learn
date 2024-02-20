from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Generator, Generic, Literal, TypeVar

from torch.utils.data import IterableDataset

from tklearn.utils.array import (
    length_of_first_array_like_in_nested_dict,
    move_to_device,
)

__all__ = [
    "IterableTorchDataset",
    "TorchDataset",
]

KT = TypeVar("KT", bound=Literal["data", "target"])

_length_cv = ContextVar("_length_cv", default=None)


class TorchDataset(Generic[KT]):
    data: Any
    target: Any

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._data = dict(*args, **kwargs)
        # pass length context (used by .to(...) method)
        _length = _length_cv.get()
        if _length is None:
            _length = length_of_first_array_like_in_nested_dict(self._data)
        self._length = _length

    def __getattr__(self, __name: KT, /) -> Any:
        if __name not in self._data:
            msg = f"'{type(self).__name__}' object has no attribute '{__name}'"
            raise AttributeError(msg)
        return self._data[__name]

    def __getitem__(self, __item: int) -> dict:
        return {k: v[__item] for k, v in self._data.items()}

    def __len__(self) -> int:
        return self._length

    def to(self, device: str) -> TorchDataset:
        data = {k: move_to_device(v, device) for k, v in self._data.items()}
        token = _length_cv.set(self._length)
        try:
            return TorchDataset(**data)
        except Exception:
            msg = (
                "unexpected error occurred while moving data"
                f" to device '{device}'"
            )
            raise RuntimeError(msg) from None
        finally:
            _length_cv.reset(token)

    def __repr__(self) -> str:
        return repr(self._data)


class IterableTorchDataset(TorchDataset, IterableDataset):
    def __iter__(self) -> Generator[dict, None, None]:
        yield from (self[i] for i in range(self._length))
