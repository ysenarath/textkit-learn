from __future__ import annotations
from typing import Optional, Any, Union, Callable

import torch
import torch.nn.functional as F

from tklearn.metrics.base import Metric, MetricOutputType
from tklearn.nn.dataset import TrainerDataset
from tklearn.nn.base import BaseTrainer
from tklearn.utils import logging
from tklearn.utils import concat

__all__ = [
    "Evaluator",
]


class Evaluator(object):
    def __init__(
        self,
        x: Any,
        y: Any = None,
        *,
        metric: Optional[Metric] = None,
        postprocessor: Union[Callable, str, None] = None,
        threshold: float = 0.0,
    ):
        self.metric: Optional[Metric] = metric
        self.dataset: TrainerDataset = TrainerDataset(x=x, y=y)
        self.postprocessor = postprocessor
        self.threshold = threshold

    def _postprocess(self, logits_or_probs: torch.Tensor) -> torch.Tensor:
        # check if the labels are multi class or multi label
        if self.postprocessor == "argmax":
            y_score = torch.argmax(logits_or_probs, dim=1)
        elif self.postprocessor == "binarize":
            y_score = logits_or_probs >= self.threshold
        elif self.postprocessor == "binary":
            y_score = logits_or_probs.squeeze() >= self.threshold
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
        criterion = None
        if hasattr(trainer, "criterion"):
            criterion = getattr(trainer, "criterion")
        try:
            _predict_batch_iter: Callable = getattr(trainer, "_predict_batch_iter")
            logits, target = None, None
            for batch_data, output in _predict_batch_iter(self.dataset):
                try:
                    y_true: torch.Tensor = batch_data["y"]
                except KeyError:
                    y_true: torch.Tensor = batch_data["x"]["labels"]
                y_true = y_true.detach().cpu()
                logits, target = concat(logits, output), concat(target, y_true)
                if metric is not None:
                    y_score = self._postprocess(output)
                    metric.update_state(y_true, y_score)
            val_loss = None
            if criterion:
                val_loss = criterion(logits, target).item()
        except AttributeError as ex:
            raise ex
        if metric is None:
            result: dict = {}
        else:
            result: dict = metric.result() or {}
        if val_loss is not None:
            result["val_loss"] = val_loss
        return result
