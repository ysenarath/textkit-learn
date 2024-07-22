from __future__ import annotations

from typing import (
    Any,
    List,
    Literal,
    Optional,
    Union,
)

import numpy as np
import torch
from numpy.typing import NDArray

from tklearn.metrics.base import MetricBase, MetricVariable

__all__ = [
    "ArrayAccumulator",
]

_ArgNameType = Union[Literal["y_true", "y_pred", "sample_weight", "y_score"], str]


class ArrayAccumulator(MetricBase):
    _instances = {}
    num_updates: MetricVariable[int] = MetricVariable()
    arrays: MetricVariable[Optional[List[np.ndarray]]] = MetricVariable()

    def __new__(cls, name: _ArgNameType, axis: int = 0) -> ArrayAccumulator:
        args = (name, axis)
        if args not in cls._instances:
            cls._instances[args] = super().__new__(cls)
        return cls._instances[args]

    def __init__(self, name: _ArgNameType, axis: int = 0) -> None:
        """Accumulate arrays along provided axis."""
        super().__init__()
        self.name = name
        self.axis = axis

    def reset(self) -> None:
        self.arrays = None
        self.num_updates = 0

    def update(
        self,
        y_true: Optional[torch.Tensor] = None,
        y_pred: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> None:
        # update kwargs with y_true and y_pred
        kwargs.update({"y_true": y_true, "y_pred": y_pred})
        value = kwargs.get(self.name)
        if value is None:
            if self.arrays is not None:
                msg = f"expected {self.name} to be provided"
                raise ValueError(msg)
            return
        elif isinstance(value, list):
            value = np.asarray(value)
        elif isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if not isinstance(value, np.ndarray):
            msg = "expected {}, but got {}".format(
                "np.ndarray, list, or torch.Tensor",
                value.__class__.__name__,
            )
            raise TypeError(msg)
        if self.arrays is None:
            if self.num_updates != 0:
                msg = "expected num_updates to be 0 when arrays is None"
                raise ValueError(msg)
            self.arrays = []
        self.arrays.append(value)
        self.num_updates += 1

    def result(self) -> NDArray:
        if self.arrays is None:
            return None
        return np.concatenate(self.arrays, axis=self.axis)
