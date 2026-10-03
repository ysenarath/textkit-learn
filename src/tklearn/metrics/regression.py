from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.stats import rankdata

from tklearn.metrics._utils import (
    add_padded,
    check_option,
    is_empty,
    to_numpy,
)
from tklearn.metrics.base import Metric

__all__ = [
    "MeanAbsoluteError",
    "MeanSquaredError",
    "PearsonCorrelation",
    "R2Score",
    "RootMeanSquaredError",
    "SpearmanCorrelation",
]

MultiOutput = Literal["uniform_average", "raw_values"]


def _as_2d(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    y_true = to_numpy(y_true).astype(float, copy=False)
    y_pred = to_numpy(y_pred).astype(float, copy=False)
    if y_true.shape != y_pred.shape:
        msg = (
            f"y_true and y_pred have different shapes: {y_true.shape} and "
            f"{y_pred.shape}"
        )
        raise ValueError(msg)
    if y_true.ndim == 1:
        return y_true[:, None], y_pred[:, None]
    if y_true.ndim != 2:
        msg = f"expected 1-D or 2-D inputs, got {y_true.ndim}-D"
        raise ValueError(msg)
    return y_true, y_pred


def _moments(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Count, means, sums of squared deviations, co-deviation and ranges.

    Rows are ``n, mean_true, mean_pred, m2_true, m2_pred, c, min_true,
    max_true, min_pred, max_pred``, one column per output.
    """
    n = np.full(y_true.shape[1], float(len(y_true)))
    mean_true, mean_pred = y_true.mean(axis=0), y_pred.mean(axis=0)
    d_true, d_pred = y_true - mean_true, y_pred - mean_pred
    return np.stack([
        n,
        mean_true,
        mean_pred,
        (d_true**2).sum(axis=0),
        (d_pred**2).sum(axis=0),
        (d_true * d_pred).sum(axis=0),
        y_true.min(axis=0),
        y_true.max(axis=0),
        y_pred.min(axis=0),
        y_pred.max(axis=0),
    ])


def _merge_moments(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Chan et al.'s pairwise update, stable where sums of squares are not
    if a.size == 0:
        return b
    if b.size == 0:
        return a
    n_a, n_b = a[0], b[0]
    n = n_a + n_b
    d_true, d_pred = b[1] - a[1], b[2] - a[2]
    scale = n_a * n_b / n
    return np.stack([
        n,
        a[1] + d_true * n_b / n,
        a[2] + d_pred * n_b / n,
        a[3] + b[3] + d_true**2 * scale,
        a[4] + b[4] + d_pred**2 * scale,
        a[5] + b[5] + d_true * d_pred * scale,
        np.minimum(a[6], b[6]),
        np.maximum(a[7], b[7]),
        np.minimum(a[8], b[8]),
        np.maximum(a[9], b[9]),
    ])


def _is_constant(moments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Whether each output of y_true and of y_pred is constant.

    Rounding in the means can leave a constant input with a tiny nonzero
    sum of squares, so this compares the range instead.
    """
    return moments[6] == moments[7], moments[8] == moments[9]


def _pearson(moments: np.ndarray) -> np.ndarray:
    m2_true, m2_pred, c = moments[3:6]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.clip(c / np.sqrt(m2_true * m2_pred), -1.0, 1.0)
    const_true, const_pred = _is_constant(moments)
    r[const_true | const_pred] = np.nan
    return r


class _RegressionMetric(Metric):
    """Base for metrics of 1-D targets or 2-D (one column per output)."""

    def __init__(self, *, multioutput: MultiOutput = "uniform_average"):
        check_option(
            "multioutput", multioutput, ("uniform_average", "raw_values")
        )
        super().__init__()
        self.multioutput = multioutput

    def _check_outputs(self, state: np.ndarray, n_outputs: int) -> None:
        if state.shape[-1] not in (0, n_outputs):
            msg = f"expected {state.shape[-1]} outputs, got {n_outputs}"
            raise ValueError(msg)

    def _check_updated(self, n: int) -> None:
        if n == 0:
            msg = f"{type(self).__name__} has no samples; call update first"
            raise ValueError(msg)

    def _average(self, values: np.ndarray) -> float | np.ndarray:
        if self.multioutput == "raw_values":
            return values
        return float(np.mean(values))


class _ErrorSumMetric(_RegressionMetric):
    def __init__(self, *, multioutput: MultiOutput = "uniform_average"):
        super().__init__(multioutput=multioutput)
        self.add_state("error_sum", np.zeros(0))
        self.add_state("n", 0)

    @staticmethod
    def _errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(self, y_true: Any, y_pred: Any) -> None:
        if is_empty(y_true, y_pred):
            return
        y_true, y_pred = _as_2d(y_true, y_pred)
        self._check_outputs(self.error_sum, y_true.shape[1])
        errors = self._errors(y_true, y_pred).sum(axis=0)
        self.error_sum = add_padded(self.error_sum, errors)
        self.n += len(y_true)

    def _mean_error(self) -> np.ndarray:
        self._check_updated(self.n)
        return self.error_sum / self.n


class MeanSquaredError(_ErrorSumMetric):
    """Mean squared error.

    Reads ``y_true`` and ``y_pred``: 1-D, or 2-D with one column per output.

    Parameters
    ----------
    multioutput : {"uniform_average", "raw_values"}, default="uniform_average"
        For 2-D targets, average the per-output errors or return them.
    """

    @staticmethod
    def _errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return (y_true - y_pred) ** 2

    def compute(self) -> float | np.ndarray:
        return self._average(self._mean_error())


class RootMeanSquaredError(MeanSquaredError):
    """Square root of the mean squared error, per output.

    Inputs and parameters are as in `MeanSquaredError`.
    """

    def compute(self) -> float | np.ndarray:
        return self._average(np.sqrt(self._mean_error()))


class MeanAbsoluteError(_ErrorSumMetric):
    """Mean absolute error.

    Inputs and parameters are as in `MeanSquaredError`.
    """

    @staticmethod
    def _errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.abs(y_true - y_pred)

    def compute(self) -> float | np.ndarray:
        return self._average(self._mean_error())


class _MomentMetric(_RegressionMetric):
    def __init__(self, *, multioutput: MultiOutput = "uniform_average"):
        super().__init__(multioutput=multioutput)
        self.add_state("moments", np.zeros((10, 0)), reduce=_merge_moments)

    def update(self, y_true: Any, y_pred: Any) -> None:
        if is_empty(y_true, y_pred):
            return
        y_true, y_pred = _as_2d(y_true, y_pred)
        self._check_outputs(self.moments, y_true.shape[1])
        self.moments = _merge_moments(self.moments, _moments(y_true, y_pred))

    def _n(self) -> int:
        return int(self.moments[0, 0]) if self.moments.size else 0


class R2Score(_MomentMetric):
    """Coefficient of determination (R²).

    Inputs and parameters are as in `MeanSquaredError`. As in
    scikit-learn, a constant ``y_true`` scores 1.0 when predicted exactly
    and 0.0 otherwise, and fewer than two samples score NaN.
    """

    def __init__(self, *, multioutput: MultiOutput = "uniform_average"):
        super().__init__(multioutput=multioutput)
        self.add_state("squared_error_sum", np.zeros(0))

    def update(self, y_true: Any, y_pred: Any) -> None:
        if is_empty(y_true, y_pred):
            return
        y_true, y_pred = _as_2d(y_true, y_pred)
        super().update(y_true, y_pred)
        self.squared_error_sum = add_padded(
            self.squared_error_sum, ((y_true - y_pred) ** 2).sum(axis=0)
        )

    def compute(self) -> float | np.ndarray:
        n = self._n()
        self._check_updated(n)
        if n < 2:
            return self._average(np.full(self.moments.shape[1], np.nan))
        residual = self.squared_error_sum
        const_true, _ = _is_constant(self.moments)
        total = np.where(const_true, 0.0, self.moments[3])
        scores = np.ones_like(total)
        valid = (residual != 0) & (total != 0)
        scores[valid] = 1 - residual[valid] / total[valid]
        scores[(residual != 0) & (total == 0)] = 0.0
        return self._average(scores)


class PearsonCorrelation(_MomentMetric):
    """Pearson correlation between ``y_true`` and ``y_pred``.

    Inputs and parameters are as in `MeanSquaredError`. A constant input
    scores NaN.
    """

    def compute(self) -> float | np.ndarray:
        self._check_updated(self._n())
        return self._average(_pearson(self.moments))


class SpearmanCorrelation(_RegressionMetric):
    """Spearman rank correlation between ``y_true`` and ``y_pred``.

    Inputs and parameters are as in `MeanSquaredError`. Ties get their
    average rank. Ranks need every value, so all inputs are kept.
    """

    def __init__(self, *, multioutput: MultiOutput = "uniform_average"):
        super().__init__(multioutput=multioutput)
        self.add_state("y_true", [], reduce="cat")
        self.add_state("y_pred", [], reduce="cat")

    def update(self, y_true: Any, y_pred: Any) -> None:
        if is_empty(y_true, y_pred):
            return
        y_true, y_pred = _as_2d(y_true, y_pred)
        if self.y_true:
            self._check_outputs(self.y_true[0], y_true.shape[1])
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def compute(self) -> float | np.ndarray:
        self._check_updated(sum(len(y) for y in self.y_true))
        rank_true = rankdata(np.concatenate(self.y_true), axis=0)
        rank_pred = rankdata(np.concatenate(self.y_pred), axis=0)
        return self._average(_pearson(_moments(rank_true, rank_pred)))
