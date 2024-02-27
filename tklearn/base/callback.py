from __future__ import annotations

import functools
from collections import UserList
from collections.abc import Sequence
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Self,
    TypeVar,
)

from typing_extensions import ParamSpec

__all__ = [
    "BaseCallback",
    "BaseCallbackList",
]

P = ParamSpec("P")
T = TypeVar("T")


class BaseCallback:
    """Base class used to build new callbacks."""


class FunctionCallback(BaseCallback, Generic[P, T]):
    def __init__(
        self,
        func: Callable[P, T],
        name: Optional[str] = None,
    ) -> None:
        self.func = func
        self.name = name or func.__name__

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.func(*args, **kwargs)

    def __getattr__(self, __name: str) -> Any:
        if __name == self.name:
            return self.func
        return getattr(self.func, __name)


class BaseCallbackList(Sequence):
    callback_functions: List[str]

    def __init_subclass__(
        cls, callback_functions: Optional[List[str]] = None
    ) -> None:
        super().__init_subclass__()
        if callback_functions is None:
            callback_functions = []
        cls.callback_functions = callback_functions
        return cls

    def __init__(self, callbacks: Optional[List[BaseCallback]] = None):
        super().__init__()
        self._callbacks: List[BaseCallback] = []
        self.extend(callbacks)

    def extend(
        self,
        callbacks: Optional[List[BaseCallback]],
        /,
        inplace: bool = True,
    ) -> Self:
        if not inplace:
            self = self.copy()
        if callbacks is None:
            return
        for callback in callbacks:
            self.append(callback)
        return self

    def append(self, callback: BaseCallback) -> None:
        if not isinstance(callback, BaseCallback):
            msg = (
                "callback must be an instance of "
                f"'{BaseCallback.__name__}', not "
                f"'{callback.__class__.__name__}'."
            )
            raise TypeError(msg)
        self._callbacks.append(callback)

    def __getitem__(self, item) -> BaseCallback:
        return self._callbacks[item]

    def __len__(self) -> int:
        return len(self._callbacks)

    def __iter__(self) -> Iterator[BaseCallback]:
        return iter(self._callbacks)

    def __getattribute__(self, __name: str) -> Any:
        callback_functions = type(self).callback_functions
        if __name in callback_functions:
            return functools.partial(self.apply, __name)
        return super().__getattribute__(__name)

    def apply(self, __name: str, /, *args, **kwargs) -> ListOutput:
        outputs = ListOutput()
        for i, callback in enumerate(self):
            output = None
            try:
                func = getattr(callback, __name)
                output = func(*args, **kwargs)
            except NotImplementedError:
                pass
            except Exception as e:
                outputs.errors[i] = e
            outputs.append(output)
        return outputs

    def copy(self) -> Self:
        """Return a shallow copy of the list."""
        return type(self)(callbacks=self._callbacks)


class ListOutput(UserList):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._errors = {}

    @property
    def errors(self) -> Mapping[int, Exception]:
        return self._errors
