from typing import Any, Dict, Generic, Sequence, Tuple, TypeVar, Union

from typing_extensions import Self

from tklearn.metrics.base import Metric

__all__ = [
    "ModelBase",
]

X, Y, Z = TypeVar("X"), TypeVar("Y"), TypeVar("Z")

XY = Union[X, Tuple[X, Y]]


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
