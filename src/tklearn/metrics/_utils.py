from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:
    from tklearn.metrics.base import Metric


def to_numpy(value: Any) -> np.ndarray:
    """Convert an array-like, including a torch tensor, to a numpy array."""
    if hasattr(value, "detach"):  # torch tensor, without importing torch
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def is_empty(*inputs: Any) -> bool:
    """Whether every input is an empty batch.

    Metrics skip empty batches: an empty list cannot tell a 2-D batch from
    a 1-D one, so its shape would not match the other inputs.
    """
    try:
        return all(len(x) == 0 for x in inputs)
    except TypeError:  # 0-d inputs are rejected by the shape checks
        return False


def require_columns(y: np.ndarray, name: str) -> None:
    """Reject a 2-D input without columns.

    No metric is defined without labels or outputs, and such a batch would
    count its rows while adding nothing per column. Empty batches (no rows)
    are skipped instead, see `is_empty`.
    """
    if y.ndim == 2 and y.shape[1] == 0:
        msg = f"expected at least one {name} column, got shape {y.shape}"
        raise ValueError(msg)


def to_python(value: Any) -> Any:
    # numpy scalars become python numbers so results can be logged and
    # serialized without special handling
    if isinstance(value, np.generic):
        return value.item()
    return value


def get_required_and_optional_params(
    metric: Metric,
) -> tuple[set[str], set[str]]:
    """Names of the required and optional inputs of ``metric.update``."""
    required, optional = set(), set()
    for param in inspect.signature(metric.update).parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            msg = (
                f"{type(metric).__name__}.update must name its inputs to be "
                "used in a MetricCollection"
            )
            raise TypeError(msg)
        if param.default is param.empty:
            required.add(param.name)
        else:
            optional.add(param.name)
    return required, optional


def sum_states(a: Any, b: Any) -> Any:
    """Add two states of the same shape.

    An empty array is the state before the first batch, when the number of
    columns is not known yet, so it takes the shape of the other.
    """
    if not isinstance(a, np.ndarray) or a.shape == b.shape:
        return a + b
    if a.size == 0:
        return b.copy()
    if b.size == 0:
        return a.copy()
    msg = f"states have different shapes: {a.shape} and {b.shape}"
    raise ValueError(msg)


def cat_states(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> list:
    """Concatenate two lists of batches with the same columns."""
    if a and b and a[0].shape[1:] != b[0].shape[1:]:
        msg = f"batches have different shapes: {a[0].shape} and {b[0].shape}"
        raise ValueError(msg)
    return a + b


def divide(num: Any, den: Any, zero_division: float) -> np.ndarray:
    """``num / den``, with `zero_division` where ``den`` is 0."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    out = np.full(np.broadcast(num, den).shape, float(zero_division))
    np.divide(num, den, out=out, where=den != 0)
    return out


def nanaverage(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Average ignoring NaNs; equal weights when all weights are 0."""
    mask = ~np.isnan(values)
    if not mask.any():
        return float("nan")
    values = values[mask]
    if weights is None or weights[mask].sum() == 0:
        return float(values.mean())
    return float(np.average(values, weights=weights[mask]))


def prf_score(
    kind: str,
    tp: Any,
    fp: Any,
    fn: Any,
    zero_division: float,
    beta: float = 1.0,
) -> np.ndarray:
    """Precision, recall or F-beta from counts, as scikit-learn defines them."""
    if kind == "precision":
        return divide(tp, np.add(tp, fp), zero_division)
    if kind == "recall":
        return divide(tp, np.add(tp, fn), zero_division)
    if kind not in ("f", "f1", "fbeta"):
        err = "expected one of 'precision', 'recall', 'f', 'f1', 'fbeta'"
        raise ValueError(f"unknown kind {kind!r}; {err}")
    if kind == "f1" and beta != 1.0:
        raise ValueError("f1 score must have beta=1.0")
    beta2 = beta**2
    num = np.multiply(1 + beta2, tp)
    return divide(num, num + fp + np.multiply(beta2, fn), zero_division)


def average_scores(
    kind: str,
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    average: str | None,
    zero_division: float,
    beta: float = 1.0,
) -> float | np.ndarray:
    """Average per-class precision, recall or F-beta.

    ``"micro"`` pools the counts,
    ``"macro"`` averages the per-class scores (ignoring NaN),
    ``"weighted"`` weights them by support and None returns
    them unaveraged.
    """
    if average not in (None, "micro", "macro", "weighted"):
        err = "expected one of None, 'micro', 'macro', 'weighted'"
        raise ValueError(f"unknown average {average!r}; {err}")
    if average == "micro":
        return float(
            prf_score(
                kind,
                tp.sum(),
                fp.sum(),
                fn.sum(),
                zero_division,
                beta,
            )
        )
    scores = prf_score(
        kind,
        tp,
        fp,
        fn,
        zero_division,
        beta,
    )
    if average is None:
        return scores
    weights = tp + fn if average == "weighted" else None
    return nanaverage(scores, weights)


def check_option(name: str, value: Any, options: tuple) -> None:
    if value not in options:
        expected = ", ".join(map(repr, options))
        msg = f"{name} must be one of {expected}, got {value!r}"
        raise ValueError(msg)
