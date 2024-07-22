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

from tklearn.metrics.base import MetricBase, MetricField

__all__ = ["StepsCounter"]


class StepsCounter(MetricBase):
    count = MetricField[int]()

    def reset(self) -> None:
        self.count = 0

    def update(self, **kwargs: Any) -> None:
        self.count += 1

    def result(self) -> int:
        return self.count

    def __repr__(self) -> str:
        return f"StepsCounter(count={self.count})"


count_steps = StepsCounter()


class ArrayAccumulator(MetricBase):
    steps_counter: StepsCounter = count_steps
    arrays: MetricField[Optional[List[np.ndarray]]] = MetricField()

    def __init__(
        self,
        field: Union[Literal["y_true", "y_pred", "sample_weight", "y_score"], str],
        axis: int = 0,
    ) -> None:
        super().__init__()
        self.field = field
        self.axis = axis

    def reset(self) -> None:
        self.arrays = None

    def update(
        self,
        y_true: Optional[torch.Tensor] = None,
        y_pred: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> None:
        # update kwargs with y_true and y_pred
        kwargs.update({"y_true": y_true, "y_pred": y_pred})
        value = kwargs.get(self.field, None)
        if value is not None and isinstance(value, list):
            value = np.array(value)
        elif value is not None and isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        elif value is not None and not isinstance(value, np.ndarray):
            msg = (
                f"'{self.field}' should be an instance of 'np.ndarray', "
                f"but got {value.__class__.__name__}"
            )
            raise TypeError(msg)
        num_steps = self.steps_counter.result()
        if (num_steps > 1 and self.arrays is None and value is not None) or (
            self.arrays is not None and value is None
        ):
            msg = f"no field named '{self.field}'"
            raise ValueError(msg)
        if self.arrays is None:
            self.arrays = []
        self.arrays.append(value)

    def result(self) -> NDArray:
        if self.arrays is None:
            return None
        return np.concatenate(self.arrays, axis=self.axis)


y_true_accum = ArrayAccumulator("y_true", axis=0)

y_pred_accum = ArrayAccumulator("y_pred", axis=0)

y_score_accum = ArrayAccumulator("y_score", axis=0)

sample_weight_accum = ArrayAccumulator("sample_weight", axis=0)
