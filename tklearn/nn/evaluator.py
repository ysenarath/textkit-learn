from __future__ import annotations
from typing import Optional, Any, Union, Callable, List, Dict
from collections.abc import Mapping

import torch
import torch.nn.functional as F
import pandas as pd
from transformers.utils import ModelOutput

from tklearn.metrics.base import Metric, MetricOutputType
from tklearn.nn.dataset import TrainerDataset
from tklearn.nn.base import BaseTrainer

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

    def postprocess(self, output: Union[torch.Tensor, ModelOutput]) -> torch.Tensor:
        # check if the labels are multi class or multi label
        if isinstance(output, Mapping) and "logits" in output:
            # huggingface pretrained model output
            output = output["logits"]
        if self.postprocessor == "argmax":
            y_score = torch.argmax(output, dim=1)
        elif self.postprocessor == "binarize":
            y_score = output >= self.threshold
        elif self.postprocessor == "binary":
            y_score = torch.reshape(output, (-1,)) >= self.threshold
        elif self.postprocessor == "softmax":
            y_score = F.softmax(output, dim=1)  # multi class scenario
        elif self.postprocessor == "sigmoid":
            y_score = F.sigmoid(output)  # binary or multi label scenario
        elif callable(self.postprocessor):
            y_score = self.postprocessor(output)
        elif self.postprocessor is None:
            y_score = output
        else:
            raise ValueError(f"invalid post-processor: {self.postprocessor}")
        return y_score

    def evaluate(
        self,
        trainer: BaseTrainer,
    ) -> MetricOutputType:
        criterion: Callable = None
        if hasattr(trainer, "criterion"):
            criterion = trainer.criterion
        metric = None
        group_metrics: Dict[str, Metric] = {}
        if self.metric is not None:
            metric = self.metric.clone()
            metric.reset_states()
            for group_name in self.group_names or []:
                group_metric = self.metric.clone()
                group_metric.reset_states()
                group_metrics[group_name] = group_metric
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
            y_score = self.postprocess(output)
            if self.groups is not None:
                batch_index: torch.Tensor = batch_data["index"]
                batch_index = batch_index.detach().cpu()
                group_metrics = self.update_group_states(
                    group_metrics,
                    y_true,
                    y_score,
                    batch_index,
                )
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
        group_results = self.group_results(group_metrics)
        result.update(group_results)
        return result

    def update_group_states(
        self,
        group_metrics: Dict[str, Metric],
        y_true: torch.Tensor,
        y_score: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> Dict[str, Metric]:
        group_names = set(self.groups.columns)
        for group_name in group_metrics.keys():
            if group_name not in group_names:
                continue
            group_index: pd.Series = self.groups[group_name].iloc[batch_index]
            group_index = group_index.reset_index(drop=True)
            y_group_true = y_true[group_index]
            y_group_score = y_score[group_index]
            if len(y_group_true) == 0:
                continue
            group_metrics[group_name].update_state(y_group_true, y_group_score)
        return group_metrics

    def group_results(
        self,
        group_metrics: Dict[str, Metric],
    ) -> dict:
        result = {}
        for subgroup_name in group_metrics.keys():
            try:
                group_metric_results = group_metrics[subgroup_name].result() or {}
            except ValueError as ex:
                group_metric_results = {}
            if subgroup_name not in result:
                result[subgroup_name] = {}
            for metric_name, metric_value in group_metric_results.items():
                result[subgroup_name][metric_name] = metric_value
        return result
