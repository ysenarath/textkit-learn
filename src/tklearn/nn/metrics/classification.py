from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

import numpy as np
import torch
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from tklearn.nn.metrics.base import Metric
from tklearn.nn.metrics.common import (
    AccumMetric,
    sample_weight_getter,
    y_pred_getter,
    y_score_getter,
    y_true_getter,
)


class Accuracy(Metric):
    y_true: AccumMetric = y_true_getter
    y_pred: AccumMetric = y_pred_getter
    sample_weight: AccumMetric = sample_weight_getter

    def __init__(
        self,
        normalize: bool = True,
        balanced: bool = False,
        adjusted: Optional[bool] = None,
    ):
        super().__init__()
        if balanced:
            adjusted = False
        self.normalize = normalize
        self.balanced = balanced
        self.adjusted = adjusted

    def result(self) -> torch.Tensor:
        y_true = self.y_true.result()
        y_pred = self.y_pred.result()
        sample_weight = self.sample_weight.result()
        if self.balanced:
            return balanced_accuracy_score(
                y_true,
                y_pred,
                sample_weight=sample_weight,
                adjusted=self.adjusted,
            )
        return accuracy_score(
            y_true,
            y_pred,
            normalize=self.normalize,
            sample_weight=sample_weight,
        )


class RocAuc(Metric):
    y_true: AccumMetric = y_true_getter
    y_score: AccumMetric = y_score_getter
    sample_weight: AccumMetric = sample_weight_getter

    def __init__(
        self,
        average: Optional[Literal["micro", "macro", "samples", "weighted"]] = "macro",
        max_fpr: Optional[float] = None,
        multi_class: Literal["raise", "ovr", "ovo"] = "raise",
        labels: Optional[ArrayLike] = None,
    ) -> None:
        super().__init__()
        self.average = average
        self.max_fpr = max_fpr
        self.multi_class = multi_class
        self.labels = labels

    def result(self) -> torch.Tensor:
        y_true = self.y_true.result()
        y_score = self.y_score.result()
        sample_weight = self.sample_weight.result()
        return roc_auc_score(
            y_true,
            y_score,
            average=self.average,
            sample_weight=sample_weight,
            max_fpr=self.max_fpr,
            multi_class=self.multi_class,
            labels=self.labels,
        )


class Precision(Metric):
    y_true: AccumMetric = y_true_getter
    y_pred: AccumMetric = y_pred_getter
    sample_weight: AccumMetric = sample_weight_getter

    def __init__(
        self,
        average: Optional[
            Literal["binary", "micro", "macro", "samples", "weighted"]
        ] = "macro",
        pos_label: Union[int, str] = 1,
        labels: Optional[ArrayLike] = None,
        zero_division: Any = 0.0,
    ) -> None:
        super().__init__()
        self.labels = labels
        self.pos_label = pos_label
        self.average = average
        self.zero_division = zero_division

    def result(self) -> torch.Tensor:
        y_true = self.y_true.result()
        y_pred = self.y_pred.result()
        sample_weight = self.sample_weight.result()
        return precision_score(
            y_true,
            y_pred,
            labels=self.labels,
            pos_label=self.pos_label,
            average=self.average,
            sample_weight=sample_weight,
            zero_division=self.zero_division,
        )


class Recall(Metric):
    y_true: AccumMetric = y_true_getter
    y_pred: AccumMetric = y_pred_getter
    sample_weight: AccumMetric = sample_weight_getter

    def __init__(
        self,
        average: Optional[
            Literal["binary", "micro", "macro", "samples", "weighted"]
        ] = "macro",
        pos_label: Union[int, str] = 1,
        labels: Optional[ArrayLike] = None,
        zero_division: Any = 0.0,
    ) -> None:
        super().__init__()
        self.labels = labels
        self.pos_label = pos_label
        self.average = average
        self.zero_division = zero_division

    def result(self) -> torch.Tensor:
        y_true = self.y_true.result()
        y_pred = self.y_pred.result()
        sample_weight = self.sample_weight.result()
        return recall_score(
            y_true,
            y_pred,
            labels=self.labels,
            pos_label=self.pos_label,
            average=self.average,
            sample_weight=sample_weight,
            zero_division=self.zero_division,
        )


class F1Score(Metric):
    y_true: AccumMetric = y_true_getter
    y_pred: AccumMetric = y_pred_getter
    sample_weight: AccumMetric = sample_weight_getter

    def __init__(
        self,
        average: Optional[
            Literal["binary", "micro", "macro", "samples", "weighted"]
        ] = "macro",
        labels: Optional[ArrayLike] = None,
        pos_label: Union[int, str] = 1,
        zero_division: Any = 0.0,
    ) -> None:
        super().__init__()
        self.labels = labels
        self.pos_label = pos_label
        self.average = average
        self.zero_division = zero_division

    def result(self) -> torch.Tensor:
        y_true = self.y_true.result()
        y_pred = self.y_pred.result()
        sample_weight = self.sample_weight.result()
        return f1_score(
            y_true,
            y_pred,
            labels=self.labels,
            pos_label=self.pos_label,
            average=self.average,
            sample_weight=sample_weight,
            zero_division=self.zero_division,
        )


class OptimalRocAucThreshold(Metric):
    y_true: AccumMetric = y_true_getter
    y_score: AccumMetric = y_score_getter
    sample_weight: AccumMetric = sample_weight_getter

    def __init__(
        self,
        pos_label: Union[int, str, None] = None,
        drop_intermediate: bool = True,
    ) -> None:
        super().__init__()
        self.pos_label = pos_label
        self.drop_intermediate = drop_intermediate

    def result(self) -> List[Dict[str, float]]:
        y_true = self.y_true.result()
        y_score = self.y_score.result()
        sample_weight = self.sample_weight.result()
        fpr, tpr, thresholds = roc_curve(
            y_true,
            y_score,
            pos_label=self.pos_label,
            sample_weight=sample_weight,
            drop_intermediate=self.drop_intermediate,
        )
        optimal_idx = np.argmax(tpr - fpr)
        return [
            {
                "fpr": f,
                "tpr": t,
                "thresholds": th,
                "optimal": i == optimal_idx,
            }
            for i, (f, t, th) in enumerate(zip(fpr, tpr, thresholds))
        ]


class OptimalPRThreshold(Metric):
    y_true: AccumMetric = y_true_getter
    y_score: AccumMetric = y_score_getter
    sample_weight: AccumMetric = sample_weight_getter

    def __init__(
        self,
        pos_label: Union[int, str, None] = None,
        drop_intermediate: bool = True,
        zero_division: Any = 0.0,
    ) -> None:
        super().__init__()
        self.pos_label = pos_label
        self.drop_intermediate = drop_intermediate
        self.zero_division = zero_division

    def result(self) -> List[Dict[str, float]]:
        y_true = self.y_true.result()
        y_score = self.y_score.result()
        sample_weight = self.sample_weight.result()
        precision, recall, thresholds = precision_recall_curve(
            y_true,
            y_score,
            pos_label=self.pos_label,
            sample_weight=sample_weight,
            drop_intermediate=self.drop_intermediate,
        )
        optimal_idx = np.argmax(2 * precision * recall / (precision + recall + 1e-12))
        return [
            {
                "precision": p,
                "recall": r,
                "f1": 2 * p * r / (p + r) if p + r > 0 else self.zero_division,
                "thresholds": t,
                "optimal": i == optimal_idx,
            }
            for i, (p, r, t) in enumerate(zip(precision, recall, thresholds))
        ]
