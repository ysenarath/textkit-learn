from __future__ import annotations
from typing import Optional, Any, Union, Callable, List, Dict

import torch
import torch.nn.functional as F
import pandas as pd

from tklearn.metrics.base import Metric, MetricOutputType
from tklearn.nn.dataset import TrainerDataset
from tklearn.nn.base import BaseTrainer
from tklearn.nn.utils import get_index

__all__ = [
    "Evaluator",
]


class Evaluator(object):
    def __init__(
        self,
        x: Any,
        y: Any = None,
        *,
        groups: Optional[Any] = None,
        group_names: Union[str, List[str], None] = None,
        metric: Optional[Metric] = None,
        postprocessor: Union[Callable, str, None] = None,
        threshold: float = 0.0,
    ):
        self.metric: Optional[Metric] = metric
        self.dataset: TrainerDataset = TrainerDataset(x=x, y=y)
        self.postprocessor = postprocessor
        if groups is not None and not isinstance(groups, pd.DataFrame):
            if group_names is None:
                raise ValueError("group_names must be specified")
            if isinstance(group_names, str):
                group_names = [group_names]
            groups = pd.DataFrame(groups, columns=group_names)
        if groups is not None:
            groups = groups.reset_index(drop=True)
        self.groups: pd.DataFrame = groups
        self.threshold = threshold

    @property
    def group_names(self) -> Optional[List[str]]:
        if self.groups is None:
            return None
        return list(self.groups.columns)

    def _postprocess(self, logits_or_probs: torch.Tensor) -> torch.Tensor:
        # check if the labels are multi class or multi label
        if self.postprocessor == "argmax":
            y_score = torch.argmax(logits_or_probs, dim=1)
        elif self.postprocessor == "binarize":
            y_score = logits_or_probs >= self.threshold
        elif self.postprocessor == "binary":
            y_score = torch.reshape(logits_or_probs, (-1,)) >= self.threshold
        elif self.postprocessor == "softmax":
            y_score = F.softmax(logits_or_probs, dim=1)  # multi class scenario
        elif self.postprocessor == "sigmoid":
            y_score = F.sigmoid(logits_or_probs)  # binary or multi label scenario
        elif callable(self.postprocessor):
            y_score = self.postprocessor(logits_or_probs)
        elif self.postprocessor is None:
            y_score = logits_or_probs
        else:
            raise ValueError(f"invalid post-processor: {self.postprocessor}")
        return y_score

    def evaluate(self, trainer: BaseTrainer) -> MetricOutputType:
        metric = None
        if self.metric is not None:
            metric = self.metric.copy()
            metric.reset_states()
        group_metrics: Dict[str, Metric] = {}
        for group_name in self.group_names or []:
            grouped_metric = self.metric.copy()
            grouped_metric.reset_states()
            group_metrics[group_name] = grouped_metric
        criterion = None
        if hasattr(trainer, "criterion"):
            criterion = getattr(trainer, "criterion")
        try:
            _predict_batch_iter: Callable = getattr(trainer, "_predict_batch_iter")
        except AttributeError as ex:
            raise ex
        val_loss_sum: float = 0.0
        batches = 0
        for batch_data, output in _predict_batch_iter(self.dataset):
            try:
                y_true: torch.Tensor = batch_data["y"]
            except KeyError:
                y_true: torch.Tensor = batch_data["x"]["labels"]
            y_true = y_true.detach().cpu()
            y_score = self._postprocess(output)
            if self.groups is not None:
                try:
                    idxs = batch_data["index"].detach().cpu()
                    groups = self.groups.iloc[idxs]
                except KeyError:
                    groups = None
            else:
                groups = None
            for group_name in group_metrics.keys():
                index = groups[group_name].reset_index(drop=True)
                y_group_true = y_true[index]
                y_group_score = y_score[index]
                group_metrics[group_name].update_state(y_group_true, y_group_score)
            # logits, target = concat(logits, output), concat(target, y_true)
            if metric is not None:
                metric.update_state(y_true, y_score)
            if criterion:
                val_loss_sum += criterion(output, y_true).item()
            batches += 1
        val_loss = (val_loss_sum / batches) if criterion else None
        result: dict = {} if metric is None else metric.result() or {}
        if val_loss is not None:
            result["val_loss"] = val_loss
        for group_name in group_metrics.keys():
            group_metric_results = group_metrics[group_name].result() or {}
            for group_metric_key, group_metric_value in group_metric_results.items():
                result[f"{group_name}_{group_metric_key}"] = group_metric_value
        return result
