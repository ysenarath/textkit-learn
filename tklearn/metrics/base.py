from __future__ import annotations
import abc
import copy
from typing import Dict, Tuple, Union, List

from tklearn import utils


__all__ = [
    "Metric",
    "UnionMetric",
    "MetricOutputType",
]

# possibly nested dict of floats or ints
MetricOutputType = Dict[str, Union[float, int, "MetricOutputType"]]


class Metric(abc.ABC):
    def __init__(
        self,
        greater_is_better: bool = True,
        needs_proba: bool = False,
        needs_threshold: bool = False,
    ):
        super(Metric, self).__init__()
        self.greater_is_better: bool = greater_is_better
        self.needs_proba: bool = needs_proba
        self.needs_threshold: bool = needs_threshold

    @abc.abstractmethod
    def reset_states(self):
        raise NotImplementedError

    @abc.abstractmethod
    def update_state(self, y_true, y_pred, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def result(self) -> MetricOutputType:
        raise NotImplementedError

    def clone(self) -> Metric:
        return copy.deepcopy(self)

    def __call__(self, y_true, y_pred, **kwargs):
        obj = self.clone()
        obj.update_state(y_true, y_pred, **kwargs)
        return obj.result()

    def __repr__(self):
        return self.__class__.__name__ + "()"


class UnionMetric(Metric):
    def __init__(self, *args, **kwargs):
        self._metrics: List[Tuple[Tuple[str, ...], Metric]] = []
        self._add_metrics(args)
        self._add_metrics(kwargs)

    def _add_metrics(self, metrics: Union[Metric, Tuple, List, Dict], *key):
        if isinstance(metrics, (list, tuple)):
            for metric in metrics:
                self._add_metrics(metric, *key)
        elif isinstance(metrics, dict):
            for ckey, metric in metrics.items():
                if isinstance(ckey, str):
                    ckey = (ckey,)
                self._add_metrics(metric, *key, *ckey)
        elif isinstance(metrics, UnionMetric):
            for ckey, metric in metrics._metrics:
                self._add_metrics(metric, *key, *ckey)
        elif isinstance(metrics, Metric):
            self._metrics.append((key, metrics))
        else:
            raise TypeError(
                "expected Metric, List[Metric] or Dict[str, Metric], "
                f"got {type(metrics).__name__}"
            )

    def reset_states(self):
        for _, metric in self._metrics:
            metric.reset_states()

    def update_state(self, y_true, y_pred, **kwargs):
        for _, metric in self._metrics:
            metric.update_state(y_true, y_pred, **kwargs)

    def result(self, *args, **kwargs) -> MetricOutputType:
        output_dict = {}
        for key, metric in self._metrics:
            output = metric.result(*args, **kwargs)
            if len(key) > 0:
                store = output_dict
                for k in key:
                    if k not in store:
                        store[k] = {}
                    elif not isinstance(store[k], dict):
                        raise ValueError(f"key {key} is already used for a metric")
                    store = store[k]
                utils.merge(store, output, inplace=True)
            else:
                utils.merge(output_dict, output, inplace=True)
        return output_dict

    def __repr__(self):
        return f"{self.__class__.__name__}()"
