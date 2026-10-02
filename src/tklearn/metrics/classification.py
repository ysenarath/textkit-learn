from __future__ import annotations

from typing import Any, Literal, TypedDict, Union

import numpy as np
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

from tklearn.metrics.base import Metric

__all__ = [
    "AUC",
    "Accuracy",
    "F1",
    "OptimalAUCThreshold",
    "OptimalPRThreshold",
    "Precision",
    "Recall",
]

Average = Literal["binary", "micro", "macro", "samples", "weighted"]


class Accuracy(Metric):
    """Accuracy, or balanced accuracy when `balanced=True`.

    Reads `y_true` and `y_pred`.
    """

    inputs = ("y_true", "y_pred")

    def __init__(
        self,
        normalize: bool = True,
        balanced: bool = False,
        adjusted: bool = False,
    ) -> None:
        self.normalize = normalize
        self.balanced = balanced
        self.adjusted = adjusted

    def compute(self, y_true, y_pred, sample_weight=None) -> float:
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


class AUC(Metric):
    """Area under the ROC curve.

    Reads `y_true` and `y_score`. For multiclass scores with
    `multi_class="ovr"`, classes without positive samples are skipped
    (`average="macro"`) or reported as NaN (`average=None`) instead of
    raising.
    """

    inputs = ("y_true", "y_score")

    def __init__(
        self,
        average: Literal["micro", "macro", "samples", "weighted"]
        | None = "macro",
        max_fpr: float | None = None,
        multi_class: Literal["raise", "ovr", "ovo"] = "raise",
        labels: ArrayLike | None = None,
    ) -> None:
        self.average = average
        self.max_fpr = max_fpr
        self.multi_class = multi_class
        self.labels = labels

    def compute(self, y_true, y_score, sample_weight=None) -> Any:
        is_multiclass = y_score.ndim == 2 and y_score.shape[1] > 1
        if is_multiclass and self.multi_class == "ovr":
            if self.average == "macro":
                return float(np.nanmean(_per_class_auc(y_true, y_score)))
            if self.average is None:
                return _per_class_auc(y_true, y_score)
        return roc_auc_score(
            y_true,
            y_score,
            average=self.average,
            sample_weight=sample_weight,
            max_fpr=self.max_fpr,
            multi_class=self.multi_class,
            labels=self.labels,
        )


def _per_class_auc(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    scores = []
    for cls in range(y_score.shape[1]):
        y_true_binary = (y_true == cls).astype(int)
        if len(np.unique(y_true_binary)) != 2:
            # no positive (or no negative) samples for this class
            scores.append(np.nan)
            continue
        scores.append(roc_auc_score(y_true_binary, y_score[:, cls]))
    return np.array(scores)


class _PRFMetric(Metric):
    inputs = ("y_true", "y_pred")

    def __init__(
        self,
        average: Average | None = "binary",
        pos_label: Union[int, str] = 1,
        labels: ArrayLike | None = None,
        zero_division: Any = 0.0,
    ) -> None:
        self.average = average
        self.pos_label = pos_label
        self.labels = labels
        self.zero_division = zero_division

    def _score(self, func, y_true, y_pred, sample_weight) -> Any:
        return func(
            y_true,
            y_pred,
            labels=self.labels,
            pos_label=self.pos_label,
            average=self.average,
            sample_weight=sample_weight,
            zero_division=self.zero_division,
        )


class Precision(_PRFMetric):
    """Precision. Reads `y_true` and `y_pred`."""

    def compute(self, y_true, y_pred, sample_weight=None) -> Any:
        return self._score(precision_score, y_true, y_pred, sample_weight)


class Recall(_PRFMetric):
    """Recall. Reads `y_true` and `y_pred`."""

    def compute(self, y_true, y_pred, sample_weight=None) -> Any:
        return self._score(recall_score, y_true, y_pred, sample_weight)


class F1(_PRFMetric):
    """F1 score. Reads `y_true` and `y_pred`."""

    def compute(self, y_true, y_pred, sample_weight=None) -> Any:
        return self._score(f1_score, y_true, y_pred, sample_weight)


class ROCPoint(TypedDict):
    fpr: float
    tpr: float
    threshold: float
    optimal: bool


class ROCThreshold(TypedDict):
    threshold: float
    data: list[ROCPoint]


class OptimalAUCThreshold(Metric):
    """Decision threshold that maximizes Youden's J (TPR - FPR).

    Reads `y_true` and `y_score` (binary scores). The result also carries the
    ROC curve points for plotting.
    """

    inputs = ("y_true", "y_score")

    def __init__(
        self,
        pos_label: Union[int, str, None] = None,
        drop_intermediate: bool = True,
    ) -> None:
        self.pos_label = pos_label
        self.drop_intermediate = drop_intermediate

    def compute(self, y_true, y_score, sample_weight=None) -> ROCThreshold:
        fpr, tpr, thresholds = roc_curve(
            y_true,
            y_score,
            pos_label=self.pos_label,
            sample_weight=sample_weight,
            drop_intermediate=self.drop_intermediate,
        )
        optimal_idx = int(np.argmax(tpr - fpr))
        data: list[ROCPoint] = [
            {"fpr": f, "tpr": t, "threshold": th, "optimal": i == optimal_idx}
            for i, (f, t, th) in enumerate(
                zip(fpr.tolist(), tpr.tolist(), thresholds.tolist())
            )
        ]
        return {"threshold": thresholds[optimal_idx].item(), "data": data}


class PRPoint(TypedDict):
    precision: float
    recall: float
    f1: float
    threshold: float
    optimal: bool


class PRThreshold(TypedDict):
    threshold: float
    data: list[PRPoint]


class OptimalPRThreshold(Metric):
    """Decision threshold that maximizes F1 on the precision-recall curve.

    Reads `y_true` and `y_score` (binary scores). The result also carries the
    precision-recall curve points for plotting.
    """

    inputs = ("y_true", "y_score")

    def __init__(
        self,
        pos_label: Union[int, str, None] = None,
        drop_intermediate: bool = True,
        zero_division: Any = 0.0,
    ) -> None:
        self.pos_label = pos_label
        self.drop_intermediate = drop_intermediate
        self.zero_division = zero_division

    def compute(self, y_true, y_score, sample_weight=None) -> PRThreshold:
        precision, recall, thresholds = precision_recall_curve(
            y_true,
            y_score,
            pos_label=self.pos_label,
            sample_weight=sample_weight,
            drop_intermediate=self.drop_intermediate,
        )
        # the last precision/recall pair has no threshold
        precision, recall = precision[:-1], recall[:-1]
        denom = precision + recall
        f1 = np.where(
            denom > 0,
            2 * precision * recall / np.where(denom > 0, denom, 1),
            self.zero_division,
        )
        optimal_idx = int(np.argmax(f1))
        data: list[PRPoint] = [
            {
                "precision": p,
                "recall": r,
                "f1": f,
                "threshold": t,
                "optimal": i == optimal_idx,
            }
            for i, (p, r, f, t) in enumerate(
                zip(
                    precision.tolist(),
                    recall.tolist(),
                    f1.tolist(),
                    thresholds.tolist(),
                )
            )
        ]
        return {"threshold": thresholds[optimal_idx].item(), "data": data}
