from __future__ import annotations

import base64
import contextlib
import contextvars
import copy
import uuid
from collections import OrderedDict
from dataclasses import MISSING
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import cloudpickle
from typing_extensions import ParamSpec

__all__ = [
    "Evaluator",
    "Metric",
    "MetricState",
]

P = ParamSpec("P")
T = TypeVar("T")

MetricStateValue = MutableMapping[str, Any]

_current_state_cv = contextvars.ContextVar("value", default=MISSING)


@contextlib.contextmanager
def set_state_context(state: MetricState):
    current_state = _current_state_cv.get()
    if current_state is not MISSING:
        msg = "trying to set context inside another context"
        raise ValueError(msg)
    token = _current_state_cv.set(state)
    try:
        yield state
    finally:
        _current_state_cv.reset(token)


def get_state(var: Union[Metric, str]) -> MetricStateValue:
    current_state = _current_state_cv.get()
    if current_state is MISSING:
        msg = "trying to get state of a metric outside the context"
        raise ValueError(msg)
    if isinstance(var, Metric):
        var = var.id
    return current_state[var]


def set_state(var: Union[Metric, str], value: MetricStateValue) -> Any:
    if isinstance(var, Metric):
        var = var.id
    current_state: MetricState = _current_state_cv.get()
    if current_state is MISSING:
        msg = "trying to set state of a metric outside the context"
        raise ValueError(msg)
    current_state[var] = value
    _current_state_cv.set(current_state)


def update_state(__var: Union[Metric, str], **kwargs: Any) -> Any:
    if isinstance(__var, Metric):
        __var = __var.id
    current_state: MetricState = _current_state_cv.get()
    if current_state is MISSING:
        msg = "trying to set state of a metric outside the context"
        raise ValueError(msg)
    current_state[__var].update(**kwargs)
    _current_state_cv.set(current_state)


def urlsafe_b64encoded_uuid4() -> str:
    return base64.urlsafe_b64encode(uuid.uuid4().bytes).decode().rstrip("=")


class Metric:
    def __init__(self) -> None:
        self.id = urlsafe_b64encoded_uuid4()

    @property
    def state(self) -> Any:
        return get_state(self)

    @state.setter
    def state(self, value: Any) -> None:
        set_state(self, value)

    def reset(self) -> None:
        self.state = {}

    def update(self, **kwargs: Any) -> None:
        pass

    def result(self) -> Any:
        raise NotImplementedError

    def copy(self, deep: bool = True) -> Metric:
        if deep:
            return cloudpickle.loads(cloudpickle.dumps(self))
        return copy.copy(self)

    def __call__(self, **kwargs: Any) -> Any:
        evaluator = Evaluator(self)
        evaluator.update_state(**kwargs)
        return evaluator.result()[0]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(...)"


class MetricState(MutableMapping[str, MetricStateValue]):
    def __init__(self) -> None:
        super().__init__()
        self._vars_dict: MutableMapping[str, Metric] = OrderedDict()
        self._vals_dict: MutableMapping[str, MetricStateValue] = {}

    def __getitem__(self, key: Union[Metric, str]) -> MetricStateValue:
        if isinstance(key, Metric):
            key = key.id
        return self._vals_dict[key]

    def __setitem__(self, key: Union[Metric, str], value: MetricStateValue) -> None:
        if isinstance(key, Metric):
            key = key.id
        self._vals_dict[key] = value

    def __delitem__(self, key: Union[Metric, str]) -> None:
        if isinstance(key, Metric):
            key = key.id
        del self._vals_dict[key]

    def __contains__(self, key: Union[Metric, str]) -> bool:
        if isinstance(key, Metric):
            key = key.id
        return key in self._vals_dict

    def __iter__(self) -> MutableMapping[str, Any]:
        return iter(self._vals_dict)

    def __len__(self) -> int:
        return len(self._vals_dict)

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
        if metric.id not in self._vars_dict:
            self._vars_dict[metric.id] = metric
        self.reset()

    def update(self, **kwargs: Any) -> None:
        with self.ctx():
            for var in self._vars_dict.values():
                var.update(**kwargs)

    def reset(self) -> None:
        with self.ctx():
            for var in self._vars_dict.values():
                var.reset()

    def result(self, metric: Metric) -> Any:
        with self.ctx():
            return metric.result()

    def ctx(self) -> contextlib.AbstractContextManager:
        return set_state_context(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._vals_dict!r})"


EvaluatorMetricsInput = Union[Metric, Sequence[Metric], Dict[str, Metric], None]


class Evaluator:
    def __init__(
        self,
        metrics: EvaluatorMetricsInput = None,
        metric_names: Optional[List[str]] = None,
        state: Optional[MetricState] = None,
    ) -> None:
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
        self.metrics = metrics
        self.metric_names = metric_names
        if state is None:
            state = MetricState()
        self.state = state
        for metric in self.metrics:
            self.state.add_metric(metric)

    def reset(self):
        self.state.reset()

    def update_state(self, **kwargs):
        self.state.update(**kwargs)

    @overload
    def result(self, return_dict: Literal[False]) -> Tuple: ...

    @overload
    def result(self, return_dict: Literal[True]) -> Dict[str, Any]: ...

    def result(self, return_dict: bool = True) -> Any:
        if return_dict:
            return self._result_return_dict()
        return tuple(self.state.result(metric) for metric in self.metrics)

    def _result_return_dict(self) -> Dict[str, Any]:
        if len(self.metrics) == 0:
            return {}
        if self.metric_names is None:
            msg = "'metric_names' should be provided to return a dictionary"
            raise ValueError(msg)
        return {
            name: self.state.result(metric)
            for name, metric in zip(self.metric_names, self.metrics)
        }
