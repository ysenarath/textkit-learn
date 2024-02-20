from __future__ import annotations

import functools
from ast import FunctionType
from collections import UserList, deque
from collections.abc import Sequence
from typing import (
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Type,
    TypeVar,
)

__all__ = [
    "BaseCallback",
    "BaseCallbackList",
]


class BaseCallback:
    """Base class used to build new callbacks."""


T = TypeVar("T", bound=BaseCallback)


class BaseCallbackList(Sequence, Generic[T]):
    _callbacks: List[T] = None

    @classmethod
    def get_callback_type(cls) -> Type[T]:
        return BaseCallback

    def __init__(self, callbacks: Optional[List[T]] = None):
        super().__init__()
        self._callbacks: List[T] = []
        if callbacks is None:
            return
        self.extend(callbacks)

    def extend(self, callbacks: List[T]) -> None:
        deque(self.append(callback) for callback in callbacks)

    def append(self, callback: T) -> None:
        callback_type = self.get_callback_type()
        if callback_type is not None and not isinstance(
            callback, callback_type
        ):
            msg = (
                f"callback must be an instance of '{callback_type.__name__}',"
                " not '{callback.__class__.__name__}'."
            )
            raise TypeError(msg)
        self._callbacks.append(callback)

    def __getitem__(self, item) -> T:
        return self._callbacks[item]

    def __len__(self) -> int:
        return len(self._callbacks)

    def __iter__(self) -> Iterator[T]:
        return iter(self._callbacks)

    def __getattribute__(self, __name: str) -> FunctionType:
        if __name in dir(BaseCallbackList):
            return super().__getattribute__(__name)
        callback_type = self.get_callback_type()
        if __name not in dir(callback_type):
            return super().__getattribute__(__name)
        if callback_type is not None and hasattr(callback_type, __name):
            return functools.partial(self.apply, __name)
        return super().__getattribute__(__name)

    def apply(self, __name: str, /, *args, **kwargs) -> OutputList:
        outputs = OutputList()
        for i, callback in enumerate(self):
            try:
                func = getattr(callback, __name)
                output = func(*args, **kwargs)
                outputs.append(output)
            except NotImplementedError:
                outputs.append(None)
            except Exception as e:
                outputs.errors[i] = e
        return outputs


class OutputList(UserList):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._errors = {}

    @property
    def errors(self) -> Mapping[int, Exception]:
        return self._errors
