from __future__ import annotations

from typing import Any, ClassVar, Literal

import numpy as np

from tklearn.metrics._utils import (
    add_padded,
    average_scores,
    check_option,
    prf_score,
    to_numpy,
)
from tklearn.metrics.base import Metric

__all__ = [
    "Accuracy",
    "BalancedAccuracy",
    "ConfusionMatrix",
    "F1",
    "FBeta",
    "Precision",
    "Recall",
]

Average = Literal["binary", "micro", "macro", "weighted"]

_AVERAGES = ("binary", "micro", "macro", "weighted", None)


def _as_labels(y: np.ndarray, name: str) -> np.ndarray:
    if y.dtype != bool and not np.issubdtype(y.dtype, np.integer):
        if not np.issubdtype(y.dtype, np.number) or not np.all(y % 1 == 0):
            msg = f"{name} must contain integer class labels"
            raise ValueError(msg)
    y = y.astype(np.int64)
    if y.size and y.min() < 0:
        msg = f"{name} must contain non-negative class labels"
        raise ValueError(msg)
    return y


def _as_indicator(y: np.ndarray, name: str) -> np.ndarray:
    if not np.isin(y, (0, 1)).all():
        msg = f"{name} must be a binary indicator matrix (0/1 values)"
        raise ValueError(msg)
    return y.astype(bool)


class _ConfusionMatrixMetric(Metric):
    """Accumulates a confusion matrix.

    Class labels (1-D, integers in ``[0, n_classes)``) fill an
    ``(n_classes, n_classes)`` matrix of true (rows) by predicted (columns)
    counts, which grows as new labels appear unless `num_classes` is set.
    Indicator matrices (2-D, multilabel) fill one ``[[tn, fp], [fn, tp]]``
    matrix per label.
    """

    def __init__(self, num_classes: int | None = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        n = num_classes or 0
        self.add_state("confmat", np.zeros((n, n), np.int64))
        self.add_state("label_confmat", np.zeros((0, 2, 2), np.int64))

    @property
    def _is_multilabel(self) -> bool:
        return self.label_confmat.shape[0] > 0

    def update(self, y_true: Any, y_pred: Any) -> None:
        y_true, y_pred = to_numpy(y_true), to_numpy(y_pred)
        if y_true.shape != y_pred.shape:
            msg = (
                f"y_true and y_pred have different shapes: {y_true.shape} "
                f"and {y_pred.shape}"
            )
            raise ValueError(msg)
        if y_true.ndim == 1:
            if self._is_multilabel:
                msg = "got class labels after multilabel targets"
                raise ValueError(msg)
            self._update_labels(
                _as_labels(y_true, "y_true"), _as_labels(y_pred, "y_pred")
            )
        elif y_true.ndim == 2:
            if self.confmat.any():
                msg = "got multilabel targets after class labels"
                raise ValueError(msg)
            self._update_indicators(
                _as_indicator(y_true, "y_true"),
                _as_indicator(y_pred, "y_pred"),
            )
        else:
            msg = (
                "expected class labels (1-D) or a multilabel indicator "
                f"matrix (2-D), got {y_true.ndim}-D inputs"
            )
            raise ValueError(msg)

    def _update_labels(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        n = max(
            self.confmat.shape[0],
            int(max(y_true.max(initial=-1), y_pred.max(initial=-1))) + 1,
        )
        if self.num_classes is not None and n > self.num_classes:
            msg = f"class labels must be less than {self.num_classes}"
            raise ValueError(msg)
        batch = np.bincount(y_true * n + y_pred, minlength=n * n)
        self.confmat = add_padded(self.confmat, batch.reshape(n, n))

    def _update_indicators(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> None:
        n_labels = y_true.shape[1]
        if self._is_multilabel and self.label_confmat.shape[0] != n_labels:
            msg = (
                f"expected {self.label_confmat.shape[0]} labels, got "
                f"{n_labels}"
            )
            raise ValueError(msg)
        if self.num_classes is not None and n_labels != self.num_classes:
            msg = f"expected {self.num_classes} labels, got {n_labels}"
            raise ValueError(msg)
        tp = (y_true & y_pred).sum(axis=0)
        fp = (~y_true & y_pred).sum(axis=0)
        fn = (y_true & ~y_pred).sum(axis=0)
        tn = len(y_true) - tp - fp - fn
        batch = np.stack([tn, fp, fn, tp], axis=1).reshape(-1, 2, 2)
        self.label_confmat = add_padded(self.label_confmat, batch)

    def _check_updated(self) -> None:
        if not self._is_multilabel and not self.confmat.any():
            msg = f"{type(self).__name__} has no samples; call update first"
            raise ValueError(msg)

    def _counts(self) -> tuple[np.ndarray, ...]:
        """Per-class tp, fp and fn, and which classes to average over.

        Without `num_classes`, only classes that occur in ``y_true`` or
        ``y_pred`` are averaged over, as in scikit-learn.
        """
        self._check_updated()
        if self._is_multilabel:
            cm = self.label_confmat
            tp, fp, fn = cm[:, 1, 1], cm[:, 0, 1], cm[:, 1, 0]
            return tp, fp, fn, np.ones(len(tp), bool)
        cm = self.confmat
        tp = np.diag(cm)
        fp, fn = cm.sum(axis=0) - tp, cm.sum(axis=1) - tp
        if self.num_classes is None:
            mask = (cm.sum(axis=0) + cm.sum(axis=1)) > 0
        else:
            mask = np.ones(len(tp), bool)
        return tp, fp, fn, mask


class Accuracy(_ConfusionMatrixMetric):
    """Fraction of correct predictions.

    For multilabel targets this is subset accuracy: the fraction of samples
    whose labels are all predicted correctly.

    Reads ``y_true`` and ``y_pred``: class labels, or indicator matrices for
    multilabel targets.
    """

    def __init__(self) -> None:
        super().__init__()
        self.add_state("n_exact", 0)

    def update(self, y_true: Any, y_pred: Any) -> None:
        y_true, y_pred = to_numpy(y_true), to_numpy(y_pred)
        super().update(y_true, y_pred)
        if y_true.ndim == 2:
            self.n_exact += int((y_true == y_pred).all(axis=1).sum())

    def compute(self) -> float:
        self._check_updated()
        if self._is_multilabel:
            n_samples = int(self.label_confmat[0].sum())
            return self.n_exact / n_samples if n_samples else 0.0
        return float(np.trace(self.confmat) / self.confmat.sum())


class BalancedAccuracy(_ConfusionMatrixMetric):
    """Mean recall over the classes that occur in ``y_true``.

    Reads ``y_true`` and ``y_pred`` class labels.

    Parameters
    ----------
    adjusted : bool, default=False
        Rescale so that random guessing scores 0 and a perfect score is 1.
    """

    def __init__(self, *, adjusted: bool = False) -> None:
        super().__init__()
        self.adjusted = adjusted

    def compute(self) -> float:
        self._check_updated()
        if self._is_multilabel:
            msg = "BalancedAccuracy does not support multilabel targets"
            raise ValueError(msg)
        support = self.confmat.sum(axis=1)
        present = support > 0
        score = float(
            np.mean(np.diag(self.confmat)[present] / support[present])
        )
        if self.adjusted:
            chance = 1 / present.sum()
            score = (score - chance) / (1 - chance)
        return score


class ConfusionMatrix(_ConfusionMatrixMetric):
    """Counts of true (rows) by predicted (columns) class.

    Reads ``y_true`` and ``y_pred``. Rows and columns are class labels
    ``0..n_classes-1``. For multilabel targets the result has one
    ``[[tn, fp], [fn, tp]]`` matrix per label.

    Parameters
    ----------
    num_classes : int, optional
        Fixes the size of the matrix; by default it fits the largest label.
    normalize : {"true", "pred", "all"}, optional
        Normalize over true classes (rows), predicted classes (columns) or
        all samples. Not supported for multilabel targets.
    """

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        normalize: Literal["true", "pred", "all"] | None = None,
    ) -> None:
        check_option("normalize", normalize, ("true", "pred", "all", None))
        super().__init__(num_classes)
        self.normalize = normalize

    def compute(self) -> np.ndarray:
        self._check_updated()
        if self._is_multilabel:
            if self.normalize is not None:
                msg = "normalize is not supported for multilabel targets"
                raise ValueError(msg)
            return self.label_confmat.copy()
        cm = self.confmat
        if self.normalize is None:
            return cm.copy()
        with np.errstate(all="ignore"):
            if self.normalize == "true":
                cm = cm / cm.sum(axis=1, keepdims=True)
            elif self.normalize == "pred":
                cm = cm / cm.sum(axis=0, keepdims=True)
            else:
                cm = cm / cm.sum()
        return np.nan_to_num(cm)


class _PRFMetric(_ConfusionMatrixMetric):
    _kind: ClassVar[str]
    beta: float = 1.0

    def __init__(
        self,
        *,
        average: Average | None = "binary",
        pos_label: int = 1,
        num_classes: int | None = None,
        zero_division: float = 0.0,
    ) -> None:
        check_option("average", average, _AVERAGES)
        super().__init__(num_classes)
        self.average = average
        self.pos_label = pos_label
        self.zero_division = zero_division

    def compute(self) -> float | np.ndarray:
        tp, fp, fn, mask = self._counts()
        if self.average == "binary":
            if self._is_multilabel or mask.sum() > 2:
                msg = (
                    "average='binary' needs binary targets; choose "
                    "'micro', 'macro', 'weighted' or None"
                )
                raise ValueError(msg)
            i = self.pos_label
            counts = (tp[i], fp[i], fn[i]) if i < len(tp) else (0, 0, 0)
            return float(
                prf_score(self._kind, *counts, self.zero_division, self.beta)
            )
        if self.average is None:
            return prf_score(
                self._kind, tp, fp, fn, self.zero_division, self.beta
            )
        return average_scores(
            self._kind,
            tp[mask],
            fp[mask],
            fn[mask],
            self.average,
            self.zero_division,
            self.beta,
        )


class Precision(_PRFMetric):
    """Precision: tp / (tp + fp).

    Reads ``y_true`` and ``y_pred``: class labels, or indicator matrices for
    multilabel targets.

    Parameters
    ----------
    average : {"binary", "micro", "macro", "weighted"} or None
        ``"binary"`` scores only `pos_label`; ``"micro"`` pools the counts of
        all classes; ``"macro"`` and ``"weighted"`` average the per-class
        scores, unweighted or weighted by support. None returns one score
        per class label (index).
    pos_label : int, default=1
        The positive class for ``average="binary"``.
    num_classes : int, optional
        Number of classes. If set, ``"macro"`` and ``"weighted"`` include
        classes that never occur; otherwise only classes found in ``y_true``
        or ``y_pred`` are averaged, as in scikit-learn.
    zero_division : float, default=0.0
        Score of a class whose denominator is 0; ``np.nan`` excludes it
        from the average.
    """

    _kind = "precision"


class Recall(_PRFMetric):
    """Recall: tp / (tp + fn). Parameters are as in `Precision`."""

    _kind = "recall"


class F1(_PRFMetric):
    """F1 score: 2 tp / (2 tp + fp + fn). Parameters are as in `Precision`."""

    _kind = "fbeta"


class FBeta(_PRFMetric):
    """F-beta score, with recall `beta` times as important as precision.

    Parameters
    ----------
    beta : float
        Weight of recall relative to precision.
    average, pos_label, num_classes, zero_division
        As in `Precision`.
    """

    _kind = "fbeta"

    def __init__(
        self,
        beta: float,
        *,
        average: Average | None = "binary",
        pos_label: int = 1,
        num_classes: int | None = None,
        zero_division: float = 0.0,
    ) -> None:
        super().__init__(
            average=average,
            pos_label=pos_label,
            num_classes=num_classes,
            zero_division=zero_division,
        )
        self.beta = beta
