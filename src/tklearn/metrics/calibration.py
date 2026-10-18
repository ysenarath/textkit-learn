"""Metrics of how well probabilities match outcomes: the log loss, the
Brier score and the expected calibration error.

Each reads ``y_true`` and ``y_score``, probabilities in [0, 1]: 1-D
scores of the positive class with 0/1 labels (binary), 2-D scores with
class labels (multiclass, one probability per class, summing to 1) or
2-D scores with an indicator matrix (multilabel, one probability per
label). Binary and multilabel scores count each probability as a binary
prediction of its own.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tklearn.metrics._utils import is_empty, sum_states, to_numpy
from tklearn.metrics.base import Metric
from tklearn.metrics.ranking import _binarize, _same_task

__all__ = [
    "BrierScore",
    "ExpectedCalibrationError",
    "LogLoss",
]


def _probabilities(
    y_true: Any, y_score: Any
) -> tuple[str, np.ndarray, np.ndarray]:
    """The task, and (n_samples, n_classes) 0/1 targets and probabilities."""
    task, targets, scores = _binarize(to_numpy(y_true), to_numpy(y_score))
    if not np.all((scores >= 0) & (scores <= 1)):
        msg = "y_score must hold probabilities in [0, 1]"
        raise ValueError(msg)
    return task, targets, scores


class _ProbabilityMetric(Metric):
    """Mean of a loss over the predictions: one per sample for multiclass
    scores, one per probability for binary and multilabel ones."""

    def __init__(self) -> None:
        super().__init__()
        self.add_state("task", None, reduce=_same_task)
        self.add_state("total", 0.0)
        self.add_state("n", 0)

    @staticmethod
    def _losses(
        task: str, targets: np.ndarray, scores: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    def update(self, y_true: Any, y_score: Any) -> None:
        if is_empty(y_true, y_score):
            return
        task, targets, scores = _probabilities(y_true, y_score)
        self.task = _same_task(self.task, task)
        losses = self._losses(task, targets, scores)
        self.total += float(losses.sum())
        self.n += losses.size

    def compute(self) -> float:
        if self.n == 0:
            msg = f"{type(self).__name__} has no samples; call update first"
            raise ValueError(msg)
        return self.total / self.n


class LogLoss(_ProbabilityMetric):
    """Negative log-likelihood of the labels under the probabilities.

    The mean over the samples of ``-log p`` of the true class for
    multiclass scores, and the mean binary cross-entropy over the
    probabilities for binary and multilabel ones, as scikit-learn's
    `log_loss` for each probability. Probabilities are clipped to
    ``[eps, 1 - eps]`` with the float64 machine epsilon, so a confident
    mistake costs about 36 rather than infinity.
    """

    @staticmethod
    def _losses(
        task: str, targets: np.ndarray, scores: np.ndarray
    ) -> np.ndarray:
        eps = np.finfo(np.float64).eps
        scores = np.clip(scores, eps, 1 - eps)
        if task == "multiclass":
            return -np.log(scores[targets])
        return -np.where(targets, np.log(scores), np.log1p(-scores))


class BrierScore(_ProbabilityMetric):
    """Mean squared difference between the probabilities and the labels.

    For multiclass scores, the sum over the classes of each sample's
    squared differences, averaged over the samples (Brier's original
    definition, from 0 to 2). For binary and multilabel scores, the mean
    over the probabilities, from 0 to 1, as scikit-learn's
    `brier_score_loss`.
    """

    @staticmethod
    def _losses(
        task: str, targets: np.ndarray, scores: np.ndarray
    ) -> np.ndarray:
        squares = (scores - targets) ** 2
        return squares.sum(axis=1) if task == "multiclass" else squares


class ExpectedCalibrationError(Metric):
    """Expected calibration error of the predicted labels' confidence.

    The confidence of a prediction is the probability of the label it
    predicts: the highest class probability for multiclass scores, and
    ``max(p, 1 - p)`` for each binary or multilabel probability ``p``,
    predicting the positive label from 0.5. The predictions are grouped
    into `n_bins` equal-width bins of confidence; the error is the mean
    over the bins, weighted by their sizes, of the gap between the mean
    confidence and the accuracy (Guo et al., 2017).

    Parameters
    ----------
    n_bins : int, default=15
        Number of confidence bins between 0 and 1.
    """

    def __init__(self, n_bins: int = 15) -> None:
        if (
            isinstance(n_bins, bool)
            or not isinstance(n_bins, int)
            or n_bins < 1
        ):
            msg = f"n_bins must be a positive int, got {n_bins!r}"
            raise ValueError(msg)
        super().__init__()
        self.n_bins = n_bins
        self.add_state("task", None, reduce=_same_task)
        # per bin: predictions, summed confidence, correct predictions
        self.add_state("bins", np.zeros((3, n_bins)))

    def update(self, y_true: Any, y_score: Any) -> None:
        if is_empty(y_true, y_score):
            return
        task, targets, scores = _probabilities(y_true, y_score)
        self.task = _same_task(self.task, task)
        if task == "multiclass":
            confidence = scores.max(axis=1)
            correct = targets[np.arange(len(scores)), scores.argmax(axis=1)]
        else:
            confidence = np.maximum(scores, 1 - scores).ravel()
            correct = ((scores >= 0.5) == targets).ravel()
        # bin i holds confidences in [i / n_bins, (i + 1) / n_bins), and
        # the last one 1 too
        index = np.minimum(
            (confidence * self.n_bins).astype(int), self.n_bins - 1
        )
        bins = np.stack([
            np.bincount(index, minlength=self.n_bins),
            np.bincount(index, weights=confidence, minlength=self.n_bins),
            np.bincount(index, weights=correct, minlength=self.n_bins),
        ])
        self.bins = sum_states(self.bins, bins)

    def compute(self) -> float:
        counts, confidence, correct = self.bins
        n = counts.sum()
        if n == 0:
            msg = f"{type(self).__name__} has no samples; call update first"
            raise ValueError(msg)
        return float(np.abs(confidence - correct).sum() / n)

    def _check_merge(self, other: Metric) -> None:
        if other.n_bins != self.n_bins:
            msg = f"cannot merge {other.n_bins} bins into {self.n_bins}"
            raise ValueError(msg)
