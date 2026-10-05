from __future__ import annotations

from typing import Any, Literal, NamedTuple, Union

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from tklearn.metrics._utils import (
    check_option,
    divide,
    is_empty,
    nanaverage,
    require_columns,
    sum_states,
    to_numpy,
)
from tklearn.metrics.base import Metric

__all__ = [
    "AUROC",
    "AveragePrecision",
    "OptimalThreshold",
    "PRPoints",
    "PrecisionRecallCurve",
    "ROCCurve",
    "ROCPoints",
]

Average = Literal["micro", "macro", "weighted"]

#: Scores of one class: exact (targets, scores) arrays, or a binned
#: ``(n_thresholds, 2)`` histogram of negative and positive counts.
_ClassData = Union[tuple[np.ndarray, np.ndarray], np.ndarray]


class ROCPoints(NamedTuple):
    """Points of a ROC curve, by decreasing threshold."""

    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray


class PRPoints(NamedTuple):
    """Points of a precision-recall curve, by increasing threshold.

    As in scikit-learn, the last point (precision 1, recall 0) has no
    threshold.
    """

    precision: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray


def _same_task(a: str | None, b: str | None) -> str | None:
    if a is None or a == b:
        return b
    if b is None:
        return a
    msg = f"inputs changed from {a} to {b} scores between updates"
    raise ValueError(msg)


def _binarize(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[str, np.ndarray, np.ndarray]:
    """Return the task, and (n_samples, n_classes) targets and scores."""
    require_columns(y_score, "class")
    y_score = y_score.astype(float)
    if y_score.ndim == 1:
        task, targets, y_score = "binary", y_true[:, None], y_score[:, None]
    elif y_score.ndim == 2 and y_true.ndim == 1:
        if not np.isin(y_true, np.arange(y_score.shape[1])).all():
            msg = (
                "y_true must contain class labels in "
                f"[0, {y_score.shape[1]}) for multiclass scores"
            )
            raise ValueError(msg)
        task, targets = (
            "multiclass",
            y_true[:, None] == np.arange(y_score.shape[1]),
        )
    elif y_score.ndim == 2:
        task, targets = "multilabel", y_true
    else:
        msg = f"y_score must be 1-D or 2-D, got {y_score.ndim}-D"
        raise ValueError(msg)
    if targets.shape != y_score.shape:
        msg = (
            f"y_true of shape {y_true.shape} does not match y_score of "
            f"shape {y_score.shape}"
        )
        raise ValueError(msg)
    if not np.isin(targets, (0, 1)).all():
        msg = "y_true must contain 0/1 labels for binary and multilabel scores"
        raise ValueError(msg)
    return task, targets.astype(bool), y_score


def _histogram(
    targets: np.ndarray, scores: np.ndarray, n_thresholds: int
) -> np.ndarray:
    if not np.all((scores >= 0) & (scores <= 1)):
        msg = (
            "binned metrics need scores in [0, 1]; pass probabilities or "
            "use thresholds=None"
        )
        raise ValueError(msg)
    grid = np.linspace(0, 1, n_thresholds)
    n_classes = scores.shape[1]
    # bin i holds scores in [grid[i], grid[i + 1])
    bins = np.searchsorted(grid, scores, side="right") - 1
    index = (np.arange(n_classes) * n_thresholds + bins) * 2 + targets
    counts = np.bincount(index.ravel(), minlength=n_classes * n_thresholds * 2)
    return counts.reshape(n_classes, n_thresholds, 2)


def _class_counts(data: _ClassData) -> tuple[int, int]:
    """Number of positives and negatives."""
    if isinstance(data, tuple):
        n_pos = int(data[0].sum())
        return n_pos, len(data[0]) - n_pos
    return int(data[:, 1].sum()), int(data[:, 0].sum())


def _cumulative_counts(hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """False and true positives at each threshold, highest first."""
    return np.cumsum(hist[::-1, 0]), np.cumsum(hist[::-1, 1])


def _roc(data: _ClassData) -> ROCPoints:
    if isinstance(data, tuple):
        return ROCPoints(*roc_curve(*data))
    fps, tps = _cumulative_counts(data)
    grid = np.linspace(0, 1, len(data))
    return ROCPoints(
        fpr=np.r_[0.0, divide(fps, fps[-1], np.nan)],
        tpr=np.r_[0.0, divide(tps, tps[-1], np.nan)],
        thresholds=np.r_[np.inf, grid[::-1]],
    )


def _precision_recall(data: _ClassData) -> PRPoints:
    if isinstance(data, tuple):
        return PRPoints(*precision_recall_curve(*data))
    fps, tps = _cumulative_counts(data)
    fps, tps = fps[::-1], tps[::-1]
    # thresholds above every score predict nothing positive: precision 1
    precision = divide(tps, tps + fps, 1.0)
    recall = divide(tps, tps[0], np.nan)
    return PRPoints(
        precision=np.r_[precision, 1.0],
        recall=np.r_[recall, 0.0],
        thresholds=np.linspace(0, 1, len(data)),
    )


def _auroc(data: _ClassData) -> float:
    n_pos, n_neg = _class_counts(data)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    fpr, tpr, _ = _roc(data)
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def _average_precision(data: _ClassData) -> float:
    n_pos, _ = _class_counts(data)
    if n_pos == 0:
        return float("nan")
    precision, recall, _ = _precision_recall(data)
    return float(-np.sum(np.diff(recall) * precision[:-1]))


class _ScoreMetric(Metric):
    """Accumulates scores, exactly or in a histogram.

    ``y_score`` is 1-D for binary targets (0/1 ``y_true``), 2-D with class
    labels ``y_true`` for multiclass targets (one-vs-rest) and 2-D with an
    indicator matrix ``y_true`` for multilabel targets.

    With ``thresholds=None`` every score is kept and results match
    scikit-learn. With an integer, scores (which must be in [0, 1]) are
    counted in that many bins between 0 and 1, so memory stays fixed and
    results are approximate.
    """

    def __init__(self, *, thresholds: int | None = None) -> None:
        if thresholds is not None and thresholds < 2:
            msg = f"thresholds must be None or at least 2, got {thresholds}"
            raise ValueError(msg)
        super().__init__()
        self.thresholds = thresholds
        self.add_state("task", None, reduce=_same_task)
        if thresholds is None:
            self.add_state("targets", [], reduce="cat")
            self.add_state("scores", [], reduce="cat")
        else:
            self.add_state("histogram", np.zeros((0, thresholds, 2), np.int64))

    @property
    def _n_classes(self) -> int:
        if self.thresholds is not None:
            return self.histogram.shape[0]
        return self.scores[0].shape[1] if self.scores else 0

    def update(self, y_true: Any, y_score: Any) -> None:
        if is_empty(y_true, y_score):
            return
        task, targets, scores = _binarize(to_numpy(y_true), to_numpy(y_score))
        if self._n_classes not in (0, scores.shape[1]):
            msg = (
                f"expected scores for {self._n_classes} classes, got "
                f"{scores.shape[1]}"
            )
            raise ValueError(msg)
        self.task = _same_task(self.task, task)
        if self.thresholds is None:
            self.targets.append(targets)
            self.scores.append(scores)
        else:
            self.histogram = sum_states(
                self.histogram,
                _histogram(targets, scores, self.thresholds),
            )

    def _class_data(self) -> list[_ClassData]:
        if self.task is None:
            msg = f"{type(self).__name__} has no samples; call update first"
            raise ValueError(msg)
        if self.thresholds is not None:
            return list(self.histogram)
        targets = np.concatenate(self.targets)
        scores = np.concatenate(self.scores)
        return [(targets[:, i], scores[:, i]) for i in range(self._n_classes)]


class _BinaryScoreMetric(_ScoreMetric):
    def update(self, y_true: Any, y_score: Any) -> None:
        if not is_empty(y_true, y_score) and np.ndim(y_score) != 1:
            msg = (
                f"{type(self).__name__} supports binary scores only; "
                "pass 1-D scores for the positive class, e.g. y_score[:, 1]"
            )
            raise ValueError(msg)
        super().update(y_true, y_score)

    def _binary_data(self) -> _ClassData:
        if self.task not in (None, "binary"):
            msg = f"{type(self).__name__} supports binary scores only"
            raise ValueError(msg)
        data = self._class_data()
        return data[0]


class _AveragedScoreMetric(_ScoreMetric):
    def __init__(
        self,
        *,
        average: Average | None = "macro",
        thresholds: int | None = None,
    ) -> None:
        check_option("average", average, ("micro", "macro", "weighted", None))
        super().__init__(thresholds=thresholds)
        self.average = average

    @staticmethod
    def _score(data: _ClassData) -> float:
        raise NotImplementedError

    def _pooled_data(self) -> _ClassData:
        """All classes as one binary problem (micro average)."""
        data = self._class_data()
        if self.thresholds is not None:
            return np.sum(data, axis=0)
        return (
            np.concatenate([t for t, _ in data]),
            np.concatenate([s for _, s in data]),
        )

    def compute(self) -> float | np.ndarray:
        data = self._class_data()
        if self.task == "binary":
            return self._score(data[0])
        if self.average == "micro":
            return self._score(self._pooled_data())
        scores = np.array([self._score(d) for d in data])
        if self.average is None:
            return scores
        weights = None
        if self.average == "weighted":
            weights = np.array([_class_counts(d)[0] for d in data], float)
        return nanaverage(scores, weights)


class AUROC(_AveragedScoreMetric):
    """Area under the ROC curve.

    Reads ``y_true`` and ``y_score``: 1-D scores with 0/1 labels (binary),
    2-D scores with class labels (multiclass, one-vs-rest) or 2-D scores
    with an indicator matrix (multilabel; one column is one label, so
    ``average=None`` returns one score in an array). A class without both
    positives and negatives scores NaN and is left out of the average.

    Parameters
    ----------
    average : {"micro", "macro", "weighted"} or None, default="macro"
        For multiclass and multilabel scores: ``"micro"`` pools all classes
        into one binary problem; ``"macro"`` and ``"weighted"`` average the
        per-class scores, unweighted or weighted by the number of
        positives. None returns one score per class. Ignored for binary
        scores.
    thresholds : int, optional
        Number of bins for approximate, fixed-memory accumulation of
        scores in [0, 1]. By default every score is kept (exact).
    """

    _score = staticmethod(_auroc)


class AveragePrecision(_AveragedScoreMetric):
    """Average precision: the area under the precision-recall curve.

    Inputs and parameters are as in `AUROC`. A class without positives
    scores NaN and is left out of the average.
    """

    _score = staticmethod(_average_precision)


class ROCCurve(_BinaryScoreMetric):
    """ROC curve of binary scores, as `ROCPoints`.

    Reads ``y_true`` (0/1) and 1-D ``y_score``.

    Parameters
    ----------
    thresholds : int, optional
        Number of bins for approximate, fixed-memory accumulation of
        scores in [0, 1]. By default every score is kept (exact).
    """

    def compute(self) -> ROCPoints:
        return _roc(self._binary_data())


class PrecisionRecallCurve(_BinaryScoreMetric):
    """Precision-recall curve of binary scores, as `PRPoints`.

    Reads ``y_true`` (0/1) and 1-D ``y_score``.

    Parameters
    ----------
    thresholds : int, optional
        Number of bins for approximate, fixed-memory accumulation of
        scores in [0, 1]. By default every score is kept (exact).
    """

    def compute(self) -> PRPoints:
        return _precision_recall(self._binary_data())


class OptimalThreshold(_BinaryScoreMetric):
    """Decision threshold that maximizes a criterion on binary scores.

    Reads ``y_true`` (0/1) and 1-D ``y_score``.

    Parameters
    ----------
    criterion : {"youden", "f1"}, default="youden"
        ``"youden"`` maximizes TPR - FPR on the ROC curve; ``"f1"``
        maximizes F1 on the precision-recall curve.
    thresholds : int, optional
        Number of bins for approximate, fixed-memory accumulation of
        scores in [0, 1]. By default every score is kept (exact).
    """

    def __init__(
        self,
        *,
        criterion: Literal["youden", "f1"] = "youden",
        thresholds: int | None = None,
    ) -> None:
        check_option("criterion", criterion, ("youden", "f1"))
        super().__init__(thresholds=thresholds)
        self.criterion = criterion

    def compute(self) -> float:
        data = self._binary_data()
        if self.criterion == "youden":
            fpr, tpr, thresholds = _roc(data)
            return float(thresholds[np.nanargmax(tpr - fpr)])
        precision, recall, thresholds = _precision_recall(data)
        precision, recall = precision[:-1], recall[:-1]
        f1 = divide(2 * precision * recall, precision + recall, 0.0)
        return float(thresholds[np.nanargmax(f1)])
