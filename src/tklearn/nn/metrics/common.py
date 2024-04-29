from __future__ import annotations

from typing import (
    Any,
    Literal,
    Optional,
)

import numpy as np
import torch

from tklearn.nn.metrics.base import Metric


class StepsCounter(Metric):
    def reset(self) -> None:
        self.state = {"value": 0}

    def update(self, *args, **kwargs: Any) -> None:
        self.state["value"] += 1

    def result(self) -> int:
        return self.state["value"]


count_steps = StepsCounter()


class AccumMetric(Metric):
    steps_counter: StepsCounter = count_steps

    def __init__(
        self,
        field: Literal["y_true", "y_pred", "sample_weight", "y_score"] = "y_true",
        axis: int = 0,
    ) -> None:
        super().__init__()
        self.field = field
        self.axis = axis

    def reset(self) -> None:
        self.state = {
            "value": None,
        }

    def update(
        self,
        y_true: Optional[torch.Tensor] = None,
        y_pred: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> None:
        kwargs["y_true"] = y_true
        kwargs["y_pred"] = y_pred
        num_steps = self.steps_counter.result()
        curr_value = self.state["value"]
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
        if (num_steps > 1 and curr_value is None and value is not None) or (
            value is None and curr_value is not None
        ):
            msg = f"'{self.field}' is not provided"
            raise ValueError(msg)
        if curr_value is None:
            curr_value = value
        else:
            curr_value = np.concatenate([curr_value, value], axis=self.axis)
        self.state["value"] = curr_value

    def result(self) -> torch.Tensor:
        return self.state["value"]


y_true_getter = AccumMetric("y_true", axis=0)

y_pred_getter = AccumMetric("y_pred", axis=0)

y_score_getter = AccumMetric("y_score", axis=0)

sample_weight_getter = AccumMetric("sample_weight", axis=0)
