from __future__ import annotations

from typing import (
    Any,
    Generic,
    Protocol,
    Self,
    TypeVar,
    runtime_checkable,
)

from tklearn.base.callback import BaseCallback, BaseCallbackList
from tklearn.utils.func import MethodMixin

__all__ = [
    "Estimator",
    "Predictor",
    "Trainer",
]

T = TypeVar("T")


@runtime_checkable
class Estimator(Protocol):
    def fit(self, x: Any, y: Any, *args, **kwargs) -> Self: ...


@runtime_checkable
class Predictor(Protocol):
    def predict(self, x: Any, *args, **kwargs) -> Any: ...

    def predict_proba(self, x: Any, *args, **kwargs) -> Any: ...


class Trainer(Generic[T], MethodMixin):
    def __init__(self, model: T) -> None:
        self.model = model

    def fit(self, *args, **kwargs) -> Trainer[T]:
        if isinstance(self.model, Estimator):
            self.model.fit(*args, **kwargs)
        raise NotImplementedError

    def predict(self, *args, **kwargs) -> Any:
        if isinstance(self.model, Predictor):
            return self.model.predict(*args, **kwargs)
        raise NotImplementedError

    def predict_proba(self, *args, **kwargs) -> Any:
        if isinstance(self.model, Predictor):
            return self.model.predict_proba(*args, **kwargs)
        raise NotImplementedError


class TrainerCallback(BaseCallback, Generic[T]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._trainer = None
        self._params = {}

    @property
    def trainer(self) -> T:
        """
        Get the trainer associated with the callback.

        Returns
        -------
        Trainer
            The trainer associated with the callback.
        """
        return self._trainer

    def set_trainer(self, trainer: T):
        """
        Set the trainer associated with the callback.

        Parameters
        ----------
        trainer : Trainer
            The model to be set.
        """
        if self._trainer is not None:
            msg = (
                f"the callback {self.__class__.__name__} is already "
                f"associated with a trainer {self._trainer.__class__.__name__}"
            )
            raise ValueError(msg)
        self._trainer = trainer

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

    def _set_trainer(self, trainer: T) -> None:
        self.set_trainer(trainer)

    def _set_params(self, params: dict) -> None:
        self.set_params(params)


class TrainerCallbackList(
    TrainerCallback[T],
    BaseCallbackList,
    Generic[T],
    callback_functions=[
        "_set_trainer",
        "_set_params",
    ],
):
    def append(self, callback: TrainerCallback[T]) -> None:
        callback.set_trainer(self.trainer)
        callback.set_params(self.params)
        super().append(callback)

    def set_trainer(self, trainer: T) -> None:
        TrainerCallback.set_trainer(self, trainer)
        self._set_trainer(self.trainer)

    def set_params(self, params: dict) -> None:
        TrainerCallback.set_params(self, params)
        self._set_params(self.params)
