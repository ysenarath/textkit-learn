from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as nt
import torch
from numpy.typing import NDArray

from tklearn.metrics.base import MetricBase, MetricVariable

__all__ = [
    "ArrayAccum",
    "StepsCounter",
]

ArgNameType = Union[
    Literal["y_true", "y_pred", "sample_weight", "y_score"],
    str,
]
ArrayLike = Union[np.ndarray, torch.Tensor, List]


class StepsCounter(MetricBase):
    # class variable to store instances
    _instance: ClassVar[Optional[StepsCounter]] = None
    # MetricVariables i.e. properties that are updated externally
    count: MetricVariable[int] = MetricVariable()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def reset(self) -> None:
        self.count = 0

    def update(self, **kwargs: Any) -> None:
        self.count += 1

    def result(self) -> int:
        return self.count

    def __repr__(self) -> str:
        return f"StepsCounter(count={self.count})"


class ArrayAccum(MetricBase):
    # class variable to store instances
    _instances: ClassVar[Dict[Tuple[str, int], ArrayAccum]] = {}
    # instance variables
    name: str
    axis: int
    # variable properties (MetricVariables)
    num_updates: MetricVariable[int] = MetricVariable()
    array: MetricVariable[Optional[nt.NDArray]] = MetricVariable()

    def __new__(cls, name: ArgNameType, axis: int = 0) -> ArrayAccum:
        args = (name, axis)
        if args not in cls._instances:
            self = super().__new__(cls)
            self.name = name
            self.axis = axis
            cls._instances[args] = self
        return cls._instances[args]

    def reset(self) -> None:
        self.num_updates = 0
        self.array = None

    def update(
        self,
        y_true: Optional[ArrayLike] = None,
        y_pred: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> None:
        # update kwargs with y_true and y_pred
        kwargs.update({"y_true": y_true, "y_pred": y_pred})
        value = kwargs.get(self.name)
        if value is None:
            if self.array is None:
                return
            msg = f"epected a value for '{self.name}', but got None"
            raise ValueError(msg)
        elif isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        elif isinstance(value, list):
            value = np.asarray(value)
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"expected a np.ndarray, list, or torch.Tensor for '{self.name}', "
                f"but got {value.__class__.__name__}"
            )
        if self.array is None:
            num_updates = self.num_updates
            if num_updates != 0:
                raise ValueError(
                    "expected num_updates to be 0 when arrays is None, "
                    f"but got {num_updates}"
                )
            self.array = value
        else:
            self.array = np.concatenate([self.array, value], axis=self.axis)
        self.num_updates += 1

    def result(self) -> NDArray:
        return self.array

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', axis={self.axis})"
        )
