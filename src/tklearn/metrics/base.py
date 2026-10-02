from __future__ import annotations

import abc
from collections.abc import Iterator, Mapping
from typing import Any, ClassVar

import numpy as np
import torch

__all__ = [
    "Metric",
    "MetricCollection",
]


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_python(value: Any) -> Any:
    # numpy scalars and 0-d arrays become plain python numbers so that results
    # can be compared, logged and serialized without special handling
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


class Metric(abc.ABC):
    """A metric computed from arrays accumulated over a whole dataset.

    Subclasses declare the arrays they read in `inputs` (required) and
    `optional_inputs`, and implement `compute`. A metric holds configuration
    only; accumulation across batches is handled by `MetricCollection`.

    The input names follow scikit-learn: `y_true`, `y_pred`, `y_score` and
    `sample_weight`. Models provide them from `Module.compute_metric_inputs`.

    Examples
    --------
    >>> f1 = F1(average="macro")
    >>> f1(y_true=[0, 1, 1], y_pred=[0, 1, 0])  # one-off computation
    0.6666666666666666
    """

    inputs: ClassVar[tuple[str, ...]] = ()
    optional_inputs: ClassVar[tuple[str, ...]] = ("sample_weight",)

    @abc.abstractmethod
    def compute(self, **arrays: np.ndarray | None) -> Any:
        """Compute the metric from complete (already concatenated) arrays.

        Parameters
        ----------
        **arrays : np.ndarray or None
            One array per name in `inputs`, and one per name in
            `optional_inputs` (None when it was not provided).

        Returns
        -------
        Any
            The metric value.
        """
        raise NotImplementedError

    def __call__(self, **inputs: Any) -> Any:
        """Compute the metric directly from complete inputs."""
        collection = MetricCollection({"value": self})
        collection.update(**inputs)
        return collection.result()["value"]

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={v!r}"
            for k, v in vars(self).items()
            if not k.startswith("_")
        )
        return f"{type(self).__name__}({params})"


class _ArrayAccumulator:
    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []

    def append(self, value: Any) -> None:
        self._chunks.append(_to_numpy(value))

    def result(self) -> np.ndarray | None:
        if not self._chunks:
            return None
        if len(self._chunks) > 1:
            self._chunks = [np.concatenate(self._chunks, axis=0)]
        return self._chunks[0]


class MetricCollection(Mapping[str, Metric]):
    """A named group of metrics evaluated over the same stream of batches.

    Each input array is accumulated once, however many metrics read it.

    Parameters
    ----------
    metrics : Mapping[str, Metric]
        Metrics keyed by the name used in `result`.

    Examples
    --------
    >>> metrics = MetricCollection({"acc": Accuracy(), "f1": F1()})
    >>> for batch in batches:
    ...     metrics.update(y_true=batch_true, y_pred=batch_pred)
    >>> metrics.result()
    {'acc': 0.9, 'f1': 0.88}
    """

    def __init__(self, metrics: Mapping[str, Metric] | None = None) -> None:
        metrics = dict(metrics or {})
        for name, metric in metrics.items():
            if not isinstance(metric, Metric):
                msg = (
                    f"expected a Metric for '{name}', got "
                    f"{type(metric).__name__}"
                )
                raise TypeError(msg)
        self._metrics = metrics
        self._required = {n for m in metrics.values() for n in m.inputs}
        self._optional = {
            n for m in metrics.values() for n in m.optional_inputs
        } - self._required
        self.reset()

    def __getitem__(self, name: str) -> Metric:
        return self._metrics[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._metrics)

    def __len__(self) -> int:
        return len(self._metrics)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._metrics!r})"

    def reset(self) -> None:
        """Discard all accumulated inputs."""
        self._arrays = {
            name: _ArrayAccumulator()
            for name in self._required | self._optional
        }

    def update(self, **inputs: Any) -> None:
        """Accumulate one batch of inputs.

        Inputs that no metric reads are ignored, so a model can return more
        than any single metric needs.

        Raises
        ------
        KeyError
            If an input required by one of the metrics is missing.
        """
        missing = sorted(n for n in self._required if inputs.get(n) is None)
        if missing:
            msg = f"missing metric inputs: {', '.join(missing)}"
            raise KeyError(msg)
        for name, accumulator in self._arrays.items():
            value = inputs.get(name)
            if value is not None:
                accumulator.append(value)

    def result(self) -> dict[str, Any]:
        """Compute every metric over the inputs accumulated so far."""
        arrays = {name: acc.result() for name, acc in self._arrays.items()}
        results = {}
        for name, metric in self._metrics.items():
            names = metric.inputs + metric.optional_inputs
            value = metric.compute(**{n: arrays.get(n) for n in names})
            results[name] = _to_python(value)
        return results
