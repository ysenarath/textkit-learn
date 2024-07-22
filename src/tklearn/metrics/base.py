from __future__ import annotations

import abc
import copy
from contextvars import ContextVar
from dataclasses import MISSING
from functools import wraps
from typing import (
    Any,
    Dict,
    Generator,
    Generic,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from weakref import WeakKeyDictionary

import cloudpickle

__all__ = [
    "MetricBase",
    "MetricState",
    "MetricVariable",
]

T = TypeVar("T")

_metric_states_cv: ContextVar[MetricState] = ContextVar("metric_state", default=MISSING)


class MetricVariable(Generic[T]):
    def __set_name__(self, owner: MetricBase, name: str) -> None:
        self.name = name

    def __get__(self, instance: MetricBase, owner: Type[MetricBase]) -> T:
        if instance is None:
            return self
        metric_states = _metric_states_cv.get()
        if metric_states is MISSING:
            msg = "trying to get state of a metric outside the context"
            raise ValueError(msg)
        return metric_states[instance][self.name]

    def __set__(self, instance: MetricBase, value: T) -> None:
        metric_states = _metric_states_cv.get()
        if metric_states is MISSING:
            msg = "trying to set state of a metric outside the context"
            raise ValueError(msg)
        metric_state = metric_states[instance]
        if not isinstance(metric_state, dict):
            msg = "state should be a dictionary"
            raise TypeError(msg)
        metric_state[self.name] = value


class MetricBase(abc.ABC):
    def reset(self) -> None: ...

    def update(self, **kwargs: Any) -> None: ...

    @abc.abstractmethod
    def result(self) -> Any:
        raise NotImplementedError

    def copy(self, deep: bool = True) -> MetricBase:
        if deep:
            return cloudpickle.loads(cloudpickle.dumps(self))
        return copy.copy(self)

    def __call__(self, **kwargs: Any) -> Any:
        state = MetricState(self)
        state.update(**kwargs)
        return next(iter(state.result()))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(...)"


def with_metric_context(func: T) -> T:
    @wraps(func)
    def decorator(self, *args: Any, **kwargs: Any) -> Any:
        token = _metric_states_cv.set(self)
        try:
            return func(self, *args, **kwargs)
        finally:
            _metric_states_cv.reset(token)

    return decorator


class MetricState(MetricBase, Mapping[MetricBase, Dict[str, Any]]):
    _metric_states: Dict[MetricBase, Dict[str, Any]]

    def __init__(
        self,
        metrics: Union[
            Dict[dict, MetricBase], List[MetricBase], MetricBase, None
        ] = None,
        metric_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        # metrics
        if metrics is None:
            metrics = {}
        if isinstance(metrics, MetricBase):
            metrics = [metrics]
        elif isinstance(metrics, Mapping):
            if metric_names is not None:
                msg = "both 'metrics' and 'metric_names' are provided"
                raise ValueError(msg)
            metric_names = list(metrics.keys())
            metrics = list(metrics.values())
        # metric names
        if metric_names is not None and len(metric_names) != len(metrics):
            msg = "length of 'metric_names' should be equal to 'metrics'"
            raise ValueError(msg)
        self.metrics: List[MetricBase] = metrics
        self.metric_names: Optional[List[str]] = metric_names
        # update states
        # this should be a ordered dictionary
        self._metric_states = WeakKeyDictionary()
        for metric in self.metrics:
            self.add_metric(metric)
        self.reset()

    def __getitem__(self, metric: MetricBase) -> Dict[str, Any]:
        return self._metric_states[metric]

    def __setitem__(self, metric: MetricBase, state: Dict[str, Any]) -> None:
        self._metric_states[metric] = state

    def __delitem__(self, metric: MetricBase) -> None:
        del self._metric_states[metric]

    def __iter__(self) -> Generator[MetricBase, None, None]:
        for key in self._metric_states:
            yield key

    def __len__(self) -> int:
        return len(self._metric_states)

    def add_metric(self, metric: MetricBase) -> None:
        if not isinstance(metric, MetricBase):
            msg = (
                "'metric' should be an instance of 'Metric', "
                f"but got {metric.__class__.__name__}"
            )
            raise TypeError(msg)
        cls = metric.__class__
        for class_var in dir(cls):
            class_var_val = getattr(cls, class_var)
            if not isinstance(class_var_val, MetricBase):
                continue
            self.add_metric(class_var_val)
        if metric in self._metric_states:
            return
        self._metric_states[metric] = {}

    @with_metric_context
    def reset(self) -> None:
        for metric in self._metric_states:
            metric.reset()

    @with_metric_context
    def update(self, **kwargs: Any) -> None:
        for metric in self._metric_states:
            metric.update(**kwargs)

    @with_metric_context
    def result(self) -> Union[Tuple[Any], Dict[str, Any]]:
        if self.metric_names is None:
            return tuple(metric.result() for metric in self.metrics)
        return {
            name: metric.result()
            for name, metric in zip(self.metric_names, self.metrics)
        }

    def __call__(self, **kwargs: Any) -> Any:
        raise NotImplementedError
