from __future__ import annotations
from collections.abc import Iterable, Sequence
import typing

__all__ = ["Callback", "CallbackList"]


class CallbackMethod(object):
    """A callable that calls a method on a callback or a list of callbacks."""

    def __init__(self, obj: typing.Union[Callback, Iterable], name: str):
        self.obj = obj
        self.name = name

    def __call__(self, *args, **kwargs):
        if isinstance(self.obj, (Iterable, Sequence)):
            error = None
            for item in self.obj:
                try:
                    method = CallbackMethod(item, self.name)
                    method(*args, **kwargs)
                except Exception as e:
                    error = e
            if error is not None:
                raise error
        elif hasattr(self.obj, self.name):
            method = getattr(self.obj, self.name)
            method(*args, **kwargs)


class Callback(object):
    def set_params(self, params: dict):
        pass


def to_callback(callback: Callback) -> Callback:
    if isinstance(callback, type):
        callback = callback()
    if isinstance(callback, Callback):
        return callback
    raise TypeError(f"expected Callback, got {type(callback).__name__}")


def to_list_of_callbacks(*args, **kwargs) -> typing.List[Callback]:
    callbacks: typing.Union[Callback, Iterable[Callback]] = list(*args, **kwargs)
    if isinstance(callbacks, Iterable):
        return [to_callback(c) for c in callbacks if c]
    try:
        return [to_callback(callbacks)]
    except TypeError:
        pass
    raise TypeError(
        "expected Callback or Iterable[Callback], got" f" {type(callbacks).__name__}"
    )


class CallbackList(Callback, Sequence):
    def __init__(self, *args, **kwargs) -> None:
        self._callbacks: typing.List[Callback] = to_list_of_callbacks(*args, **kwargs)

    def __getitem__(self, index: int) -> Callback:
        return self._callbacks[index]

    def __getattribute__(self, name: str) -> CallbackMethod:
        if not name.startswith("_"):
            return CallbackMethod(self, name)
        return super().__getattribute__(name)

    def __iter__(self) -> typing.Iterator[Callback]:
        yield from self._callbacks

    def __len__(self) -> int:
        return len(self._callbacks)
