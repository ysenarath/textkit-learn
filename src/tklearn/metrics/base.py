from __future__ import annotations

import abc
import copy
from collections.abc import Iterator, Mapping
from typing import Any, Callable, Literal, Union

from tklearn.metrics._utils import (
    cat_states,
    get_required_and_optional_params,
    sum_states,
    to_numpy,
    to_python,
)

__all__ = [
    "Metric",
    "MetricCollection",
]

#: How two values of a state are combined:
# 1. added (``"sum"``),
# 2. concatenated (``"cat"``), or by a
# 3. function of the two values.
Reduction = Union[Literal["sum", "cat"], Callable[[Any, Any], Any]]


class Metric(abc.ABC):
    """A metric accumulated batch by batch.

    `update` adds a batch and `compute` returns the value over every batch
    since the last `reset`. The state holds only what the metric needs (a
    confusion matrix, running sums, ...), so `merge` can combine metrics
    updated on different shards or processes.

    Calling a metric computes it for the given inputs alone and leaves the
    accumulated state untouched. Empty batches are ignored.

    Subclasses register their state with `add_state` in ``__init__`` and
    implement `update` and `compute`.

    Examples
    --------
    >>> f1 = F1(average="macro")
    >>> for y_true, y_pred in batches:
    ...     f1.update(y_true, y_pred)
    >>> f1.compute()
    0.83
    >>> F1()([0, 1, 1], [0, 1, 0])  # one-off
    0.6666666666666666
    """

    def __init__(self) -> None:
        self._defaults: dict[str, Any] = {}
        self._reductions: dict[str, Reduction] = {}

    def add_state(
        self, name: str, default: Any, reduce: Reduction = "sum"
    ) -> None:
        """Register a state attribute, reset to a copy of `default`."""
        self._defaults[name] = default
        self._reductions[name] = reduce
        setattr(self, name, copy.deepcopy(default))

    def reset(self) -> None:
        """Discard the accumulated state."""
        for name, default in self._defaults.items():
            setattr(self, name, copy.deepcopy(default))

    @abc.abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        """Add a batch to the state."""

    @abc.abstractmethod
    def compute(self) -> Any:
        """Compute the metric over everything added since the last reset."""

    def merge(self, other: Metric) -> None:
        """Add the state of another metric of the same type to this one.

        Raises ValueError, leaving this metric unchanged, if the states do
        not fit together, e.g. different numbers of labels or outputs.
        """
        if type(other) is not type(self):
            msg = (
                f"cannot merge {type(other).__name__} into "
                f"{type(self).__name__}"
            )
            raise TypeError(msg)
        self._check_merge(other)
        values = {}
        for name, reduce in self._reductions.items():
            a, b = getattr(self, name), getattr(other, name)
            try:
                if reduce == "sum":
                    values[name] = sum_states(a, b)
                elif reduce == "cat":
                    values[name] = cat_states(a, b)
                else:
                    values[name] = reduce(a, b)
            except ValueError as e:
                msg = f"cannot merge {type(self).__name__}: {e}"
                raise ValueError(msg) from e
        # set the states only once all of them were combined
        for name, value in values.items():
            setattr(self, name, value)

    def _check_merge(self, other: Metric) -> None:
        """Raise ValueError if `other` cannot be merged into this metric."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Compute the metric for these inputs only."""
        metric = copy.copy(self)
        metric.reset()
        metric.update(*args, **kwargs)
        return metric.compute()

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={v!r}"
            for k, v in vars(self).items()
            if not k.startswith("_") and k not in self._defaults
        )
        return f"{type(self).__name__}({params})"


class MetricCollection(Mapping[str, Metric]):
    """Named metrics updated from the same batches.

    `update` takes keyword inputs and passes each metric only the ones its
    `update` accepts, so classification metrics can read ``y_pred`` while
    ranking metrics read ``y_score``. Inputs no metric accepts are ignored.

    Parameters
    ----------
    metrics : Mapping[str, Metric]
        Metrics keyed by the name used in `compute`.

    Examples
    --------
    >>> metrics = MetricCollection({"f1": F1(average="macro"), "auc": AUROC()})
    >>> for batch in batches:
    ...     metrics.update(y_true=..., y_pred=..., y_score=...)
    >>> metrics.compute()
    {'f1': 0.83, 'auc': 0.91}
    """

    def __init__(self, metrics: Mapping[str, Metric] | None = None) -> None:
        self._metrics: dict[str, Metric] = {}
        self._params: dict[str, tuple[set[str], set[str]]] = {}
        for name, metric in (metrics or {}).items():
            if not isinstance(metric, Metric):
                msg = (
                    f"expected a Metric for '{name}', got "
                    f"{type(metric).__name__}"
                )
                raise TypeError(msg)
            self._metrics[name] = metric
            self._params[name] = get_required_and_optional_params(metric)

    def __getitem__(self, name: str) -> Metric:
        return self._metrics[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._metrics)

    def __len__(self) -> int:
        return len(self._metrics)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._metrics!r})"

    @property
    def input_names(self) -> frozenset[str]:
        """Names of the inputs that any metric accepts."""
        return frozenset(
            name
            for required, optional in self._params.values()
            for name in required | optional
        )

    def update(self, **inputs: Any) -> None:
        """Update every metric with the inputs it accepts.

        Raises
        ------
        KeyError
            If an input that a metric requires is missing.
        """
        missing = sorted({
            param
            for required, _ in self._params.values()
            for param in required
            if inputs.get(param) is None
        })
        if missing:
            msg = f"missing metric inputs: {', '.join(missing)}"
            raise KeyError(msg)
        # convert tensors once, not once per metric
        inputs = {
            k: to_numpy(v) if hasattr(v, "detach") else v
            for k, v in inputs.items()
        }
        for name, metric in self._metrics.items():
            required, optional = self._params[name]
            kwargs = {k: inputs[k] for k in required}
            kwargs.update(
                (k, inputs[k]) for k in optional if inputs.get(k) is not None
            )
            metric.update(**kwargs)

    def compute(self) -> dict[str, Any]:
        """Compute every metric."""
        return {
            name: to_python(metric.compute())
            for name, metric in self._metrics.items()
        }

    def reset(self) -> None:
        """Reset every metric."""
        for metric in self._metrics.values():
            metric.reset()

    def merge(self, other: MetricCollection) -> None:
        """Merge the metrics of another collection with the same names."""
        if set(other) != set(self):
            msg = "cannot merge collections with different metrics"
            raise ValueError(msg)
        for name, metric in self._metrics.items():
            metric.merge(other[name])
