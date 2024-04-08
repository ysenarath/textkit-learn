from __future__ import annotations

from typing import Any, Dict, Generic, Sequence, Tuple, TypeVar, Union

from typing_extensions import ParamSpec, Self

from tklearn.base.callback import CallbackBase, CallbackListBase
from tklearn.metrics.base import Metric

__all__ = [
    "ModelBase",
    "ModelCallbackBase",
    "ModelCallbackListBase",
]

X, Y, Z = TypeVar("X"), TypeVar("Y"), TypeVar("Z")

XY = Union[X, Tuple[X, Y]]

P = ParamSpec("P")

T = TypeVar("T", bound="ModelBase")


class ModelBase(Generic[X, Y, Z]):
    def fit(self, x: XY, y: Y = None, /, **kwargs) -> Self: ...

    def predict(self, x: XY, y: Y = None, /, **kwargs) -> Z: ...

    def evaluate(
        self,
        x: XY,
        y: Y = None,
        *,
        metrics: Union[Sequence[Metric], Dict[str, Metric]],
        **kwargs: Any,
    ) -> Union[Dict[str, Any], Tuple[Any, ...]]: ...


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
        if model is not None and self._model is not None and model is not self._model:
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
