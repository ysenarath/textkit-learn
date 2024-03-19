from __future__ import annotations

import functools
import warnings
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
    TypeVar,
)

from typing_extensions import ParamSpec, Self

__all__ = [
    "CallbackBase",
    "CallbackListBase",
]

P = ParamSpec("P")
T = TypeVar("T")


class ListOutput(UserList):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._errors = {}

    @property
    def errors(self) -> Mapping[int, Exception]:
        return self._errors


class CallbackBase:
    """Base class used to build new callbacks."""

    def __new__(cls, *args, **kwargs) -> Self:
        self = super().__new__(cls)
        cls.reset = functools.partial(cls.__init__, self, *args, **kwargs)
        return self


class FunctionCallback(CallbackBase, Generic[P, T]):
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


class CallbackListBase(Sequence):
    callback_functions: List[str]

    def __init_subclass__(
        cls, callback_functions: Optional[List[str]] = None
    ) -> None:
        super().__init_subclass__()
        if callback_functions is None:
            callback_functions = []
        callback_functions.append("reset")
        cls.callback_functions = callback_functions
        return cls

    def __init__(self, callbacks: Optional[List[CallbackBase]] = None):
        super().__init__()
        self._callbacks: List[CallbackBase] = []
        self.extend(callbacks)

    def extend(
        self,
        callbacks: Optional[List[CallbackBase]],
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

    def append(self, callback: CallbackBase) -> None:
        if not isinstance(callback, CallbackBase):
            msg = (
                "callback must be an instance of "
                f"'{CallbackBase.__name__}', not "
                f"'{callback.__class__.__name__}'."
            )
            raise TypeError(msg)
        self._callbacks.append(callback)

    def __getitem__(self, item) -> CallbackBase:
        return self._callbacks[item]

    def __len__(self) -> int:
        return len(self._callbacks)

    def __iter__(self) -> Iterator[CallbackBase]:
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
                warnings.warn(
                    f"{e} in callback '{type(callback).__name__}' "
                    f"when calling '{__name}'",
                    RuntimeWarning,
                    stacklevel=0,
                    source=e,
                )
                outputs.errors[i] = e
            outputs.append(output)
        return outputs

    def copy(self) -> Self:
        """Return a shallow copy of the list."""
        return type(self)(callbacks=self._callbacks)


class ModelCallbackBase(CallbackBase, Generic[T]):
    def __init__(self, *args, **kwargs) -> None:
        self._model = None
        self._params = {}
        super().__init__(*args, **kwargs)

    @property
    def model(self) -> T:
        """
        Get the model associated with the callback.

        Returns
        -------
        Model
            The model associated with the callback.
        """
        return self._model

    def set_model(self, model: T):
        """
        Set the model associated with the callback.

        Parameters
        ----------
        model : Model
            The model to be set.
        """
        if (
            model is not None
            and self._model is not None
            and model is not self._model
        ):
            msg = (
                f"the callback '{type(self).__name__}' is already "
                f"associated with a model '{type(self._model).__name__}'"
            )
            raise ValueError(msg)
        self._model = model

    @property
    def params(self) -> dict:
        """
        Get the parameters of the callback.

        Returns
        -------
        dict
            The parameters of the callback.
        """
        return self._params

    def set_params(self, params):
        """
        Set the parameters of the callback.

        Parameters
        ----------
        params : dict
            The parameters to be set.
        """
        if params is None:
            params = {}
        self._params = params

    def _set_model(self, model: T) -> None:
        self.set_model(model)

    def _set_params(self, params: dict) -> None:
        self.set_params(params)


class ModelCallbackListBase(
    ModelCallbackBase[T],
    CallbackListBase,
    Generic[T],
    callback_functions=[
        "_set_model",
        "_set_params",
    ],
):
    def append(self, callback: ModelCallbackBase[T]) -> None:
        callback.set_model(self.model)
        callback.set_params(self.params)
        super().append(callback)

    def set_model(self, model: T) -> None:
        ModelCallbackBase.set_model(self, model)
        self._set_model(self.model)

    def set_params(self, params: dict) -> None:
        ModelCallbackBase.set_params(self, params)
        self._set_params(self.params)
