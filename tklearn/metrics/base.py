from __future__ import annotations
import abc
import copy
from dataclasses import dataclass, field
import inspect
from typing import Dict, Optional, Tuple, Union, List

from datasets import DownloadMode
from evaluate.utils.file_utils import DownloadConfig
from datasets.utils.version import Version
import evaluate
from evaluate import EvaluationModule


__all__ = [
    "Metric",
    "UnionMetric",
    "HuggingFaceMetric",
    "TextClassificationMetric",
    "MetricOutputType",
]

# possibly nested dict of floats or ints
MetricOutputType = Dict[str, Union[float, int, "MetricOutputType"]]


class Metric(abc.ABC):
    def __init__(self):
        super(Metric, self).__init__()

    @abc.abstractmethod
    def reset_states(self):
        raise NotImplementedError

    @abc.abstractmethod
    def update_state(self, y_true, y_pred, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def result(self) -> MetricOutputType:
        raise NotImplementedError

    def copy(self, deep=False) -> Metric:
        if deep:
            return copy.deepcopy(self)
        return copy.copy(self)

    def __call__(self, y_true, y_pred, **kwargs):
        obj = self.copy()
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
                store.update(output)
            else:
                output_dict.update(output)
        return output_dict

    def __repr__(self):
        return f"{self.__class__.__name__}()"


@dataclass
class HuggingFaceMetricLoader:
    path: str
    config_name: Optional[str] = None
    module_type: Optional[str] = None
    process_id: int = 0
    num_process: int = 1
    cache_dir: Optional[str] = None
    experiment_id: Optional[str] = None
    keep_in_memory: bool = False
    download_config: Optional[DownloadConfig] = None
    download_mode: Optional[DownloadMode] = None
    revision: Optional[Union[str, Version]] = None
    init_kwargs: dict = field(default_factory=dict)

    def __call__(self) -> EvaluationModule:
        return evaluate.load(
            self.path,
            config_name=self.config_name,
            module_type=self.module_type,
            process_id=self.process_id,
            num_process=self.num_process,
            cache_dir=self.cache_dir,
            experiment_id=self.experiment_id,
            keep_in_memory=self.keep_in_memory,
            download_config=self.download_config,
            download_mode=self.download_mode,
            revision=self.revision,
            **self.init_kwargs,
        )


class HuggingFaceMetric(Metric):
    def __init__(self, *args, **kwargs):
        super(HuggingFaceMetric, self).__init__()
        signature = inspect.signature(evaluate.load).bind(*args, **kwargs)
        signature.apply_defaults()
        load_kwargs = dict(signature.arguments)
        result_kwargs: dict = load_kwargs.pop("init_kwargs")
        init_kwargs = result_kwargs.pop("init_kwargs", None)
        if init_kwargs is None:
            init_kwargs = {}
        self._hf_metric_loader = HuggingFaceMetricLoader(
            **load_kwargs, init_kwargs=init_kwargs
        )
        self._result_kwargs = result_kwargs
        self.reset_states()

    def reset_states(self):
        self._hf_metric: EvaluationModule = self._hf_metric_loader()

    @property
    def hf_metric(self) -> EvaluationModule:
        if not hasattr(self, "_hf_metric"):
            self.reset_states()
        return getattr(self, "_hf_metric")

    def update_state(self, y_true, y_pred, **kwargs):
        self.hf_metric.add_batch(predictions=y_pred, references=y_true, **kwargs)

    def result(self, *args, **kwargs) -> MetricOutputType:
        kwargs.update(self._result_kwargs)
        avg_strategy = None
        if "average" in kwargs:
            avg_strategy = kwargs["average"].lower()
        out: dict = {}
        for key, value in (self.hf_metric.compute(*args, **kwargs) or {}).items():
            # rename key to avoid conflict with other metrics
            if avg_strategy is not None:
                out[f"{avg_strategy}_" + key] = value
            else:
                out[key] = value
        return out


class TextClassificationMetric(Metric):
    def __init__(self, num_labels=1):
        super().__init__()
        if num_labels > 1:
            self._tc_metric = UnionMetric(
                HuggingFaceMetric("f1", average="micro"),
                HuggingFaceMetric("precision", average="micro", zero_division=0),
                HuggingFaceMetric("recall", average="micro", zero_division=0),
                HuggingFaceMetric("f1", average="macro"),
                HuggingFaceMetric("precision", average="macro", zero_division=0),
                HuggingFaceMetric("recall", average="macro", zero_division=0),
                HuggingFaceMetric("f1", average="weighted"),
                HuggingFaceMetric("precision", average="weighted", zero_division=0),
                HuggingFaceMetric("recall", average="weighted", zero_division=0),
                HuggingFaceMetric("accuracy"),
            )
        else:  # binary classification
            self._tc_metric = UnionMetric(
                HuggingFaceMetric("f1"),
                HuggingFaceMetric("precision", zero_division=0),
                HuggingFaceMetric("recall", zero_division=0),
                HuggingFaceMetric("accuracy"),
            )

    def update_state(self, y_true, y_pred, **kwargs):
        # ic(y_true, y_pred)
        self._tc_metric.update_state(y_true, y_pred, **kwargs)

    def reset_states(self):
        self._tc_metric.reset_states()

    def result(self, *args, **kwargs) -> MetricOutputType:
        return self._tc_metric.result(*args, **kwargs)
