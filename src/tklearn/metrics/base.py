from __future__ import annotations

import abc
import copy
from contextvars import ContextVar
from dataclasses import MISSING
from functools import wraps
from typing import Any, Dict, List, Mapping, Optional, Tuple, TypeVar, Union
from weakref import WeakKeyDictionary

import cloudpickle

__all__ = [
    "Metric",
    "MetricState",
]

T = TypeVar("T")

_metric_state_cv: ContextVar[MetricState] = ContextVar("metric_state", default=MISSING)


class Metric(abc.ABC):
    @property
    def state(self) -> Dict[str, Any]:
        metric_state = _metric_state_cv.get()
        if metric_state is MISSING:
            msg = "trying to get state of a metric outside the context"
            raise ValueError(msg)
        return metric_state.states[self]

    @state.setter
    def state(self, value: Dict[str, Any]) -> None:
        metric_state = _metric_state_cv.get()
        if metric_state is MISSING:
            msg = "trying to set state of a metric outside the context"
            raise ValueError(msg)
        if not isinstance(value, dict):
            msg = "state should be a dictionary"
            raise TypeError(msg)
        metric_state.states[self] = value

    def reset(self) -> None: ...

    def update(self, **kwargs: Any) -> None: ...

    @abc.abstractmethod
    def result(self) -> Any:
        raise NotImplementedError

    def copy(self, deep: bool = True) -> Metric:
        if deep:
            return cloudpickle.loads(cloudpickle.dumps(self))
        return copy.copy(self)

    def __call__(self, **kwargs: Any) -> Any:
        state = MetricState(self)
        state.update(**kwargs)
        return state.result()[0]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(...)"


def with_metric_context(func: T) -> T:
    @wraps(func)
    def decorator(self, *args: Any, **kwargs: Any) -> Any:
        token = _metric_state_cv.set(self)
        try:
            return func(self, *args, **kwargs)
        finally:
            _metric_state_cv.reset(token)

    return decorator


class MetricState(Metric):
    def __init__(
        self,
        metrics: Union[Dict[dict, Metric], List[Metric], Metric],
        metric_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        # metrics
        if metrics is None:
            metrics = []
        elif isinstance(metrics, Metric):
            metrics = [metrics]
        elif isinstance(metrics, Mapping):
            if metric_names is not None:
                msg = "both 'metrics' and 'metric_names' are provided"
                raise ValueError(msg)
            metric_names = list(metrics.keys())
            metrics = list(metrics.values())
        self.metrics: List[Metric] = metrics
        # metric names
        if metric_names is not None:
            if len(metric_names) != len(metrics):
                msg = "length of 'metric_names' should be equal to 'metrics'"
                raise ValueError(msg)
        self.metric_names: Optional[List[str]] = metric_names
        # update states
        self.states: Dict[Metric, Dict[str, Any]] = WeakKeyDictionary()
        for metric in self.metrics:
            self.add_metric(metric)
        self.reset()

    def add_metric(self, metric: Metric) -> None:
        if not isinstance(metric, Metric):
            msg = (
                "'metric' should be an instance of 'Metric', "
                f"but got {metric.__class__.__name__}"
            )
            raise TypeError(msg)
        cls = metric.__class__
        for class_var in dir(cls):
            class_var_val = getattr(cls, class_var)
            if not isinstance(class_var_val, Metric):
                continue
            self.add_metric(class_var_val)
        if metric in self.states:
            return
        self.states[metric] = None

    @with_metric_context
    def reset(self) -> None:
        for metric in self.states:
            metric.reset()

    @with_metric_context
    def update(self, **kwargs: Any) -> None:
        for metric in self.states:
            metric.update(**kwargs)

    @with_metric_context
    def result(self) -> Union[Tuple[Any], Dict[str, Any]]:
        if self.metric_names is None:
            return tuple(metric.result() for metric in self.metrics)
        return {
            name: metric.result()
            for name, metric in zip(self.metric_names, self.metrics)
        }
