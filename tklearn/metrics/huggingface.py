from __future__ import annotations
from dataclasses import dataclass, field
import inspect
from typing import Optional, Union

from datasets import DownloadMode
from evaluate.utils.file_utils import DownloadConfig
from datasets.utils.version import Version
import evaluate
from evaluate import EvaluationModule

from tklearn.metrics.base import Metric, UnionMetric, MetricOutputType


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
        out: dict = {}
        for key, value in (self.hf_metric.compute(*args, **kwargs) or {}).items():
            # rename key to avoid conflict with other metrics
            out[key] = value
        if "average" in kwargs:
            avg_strategy: str = kwargs["average"].lower()
            return {avg_strategy: out}
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
