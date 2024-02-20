from __future__ import annotations

from typing import (
    Any,
    Generic,
    Protocol,
    Self,
    TypeVar,
    runtime_checkable,
)

from tklearn.utils.func import MethodMixin

__all__ = [
    "Estimator",
    "Predictor",
    "Trainer",
]

T = TypeVar("T")


@runtime_checkable
class Estimator(Protocol):
    def fit(self, data: Any, target: Any) -> Self: ...


@runtime_checkable
class Predictor(Protocol):
    def predict(self, data: Any) -> Any: ...

    def predict_proba(self, data: Any) -> Any: ...


class Trainer(Generic[T], MethodMixin):
    def __init__(self, model: T) -> None:
        self.model = model

    def fit(self, data: Any, target: Any) -> Trainer[T]:
        if isinstance(self.model, Estimator):
            self.model.fit(data, target)
        raise NotImplementedError

    def predict(self, data: Any) -> Any:
        if isinstance(self.model, Predictor):
            return self.model.predict(data)
        raise NotImplementedError

    def predict_proba(self, data: Any) -> Any:
        if isinstance(self.model, Predictor):
            return self.model.predict_proba(data)
        raise NotImplementedError
