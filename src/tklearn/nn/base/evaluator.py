from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Callable, Generic, TypeVar

from tklearn.metrics import Metric, MetricCollection
from tklearn.nn.base.module import Module
from tklearn.nn.base.predictor import iter_batch_outputs
from tklearn.nn.callbacks.base import Callback, CallbackList, CallbacksMixin
from tklearn.nn.loss import LossDict, LossFunction

__all__ = [
    "Evaluator",
]

BatchT = TypeVar("BatchT")
OutputT = TypeVar("OutputT")


class Evaluator(CallbacksMixin, Generic[BatchT, OutputT]):
    """Compute the loss and metrics of a model over a dataloader.

    Parameters
    ----------
    model : Module
        The model to evaluate.
    metrics : Mapping[str, Metric], optional
        Metrics keyed by result name. Their inputs come from
        `model.compute_metric_inputs` (or `compute_metric_inputs`).
    loss : callable, optional
        ``loss(batch, output)``; defaults to `model.compute_loss`.
    include_loss : bool, default=True
        Whether to report the mean loss. Set to False for models without a
        loss.
    compute_metric_inputs : callable, optional
        ``compute_metric_inputs(batch, output)`` -> dict, overriding
        `model.compute_metric_inputs`.
    callbacks : Callback or iterable of Callback, optional
        Callbacks receiving the ``on_test_*`` hooks.

    Examples
    --------
    >>> evaluator = Evaluator(model, metrics={"f1": F1(average="macro")})
    >>> evaluator.evaluate(valid_loader, prefix="valid_")
    {'valid_loss': 0.41, 'valid_f1': 0.83}
    """

    def __init__(
        self,
        model: Module[BatchT, OutputT],
        metrics: Mapping[str, Metric] | None = None,
        *,
        loss: LossFunction[BatchT, OutputT] | None = None,
        include_loss: bool = True,
        compute_metric_inputs: Callable[[BatchT, OutputT], dict[str, Any]]
        | None = None,
        callbacks: CallbackList | Iterable[Callback] | Callback | None = None,
    ) -> None:
        self.model = model
        self.metrics = MetricCollection(metrics)
        self.loss = loss
        self.include_loss = include_loss
        self.compute_metric_inputs = compute_metric_inputs
        self.callbacks = callbacks

    def _compute_loss(self, batch: BatchT, output: OutputT) -> LossDict:
        if self.loss is not None:
            return LossDict(self.loss(batch, output))
        return LossDict(self.model.compute_loss(batch, output))

    def _metric_inputs(self, batch: BatchT, output: OutputT) -> dict[str, Any]:
        if self.compute_metric_inputs is not None:
            return self.compute_metric_inputs(batch, output)
        return self.model.compute_metric_inputs(batch, output)

    def evaluate(
        self, dataloader: Iterable[BatchT], prefix: str = ""
    ) -> dict[str, Any]:
        """Evaluate the model over a dataloader.

        Parameters
        ----------
        dataloader : iterable
            Batches to evaluate on.
        prefix : str, default=""
            Prepended to every result key, e.g. ``"valid_"``.

        Returns
        -------
        dict
            Mean loss terms (``loss`` or the keys of a `LossDict`) followed by
            one entry per metric.
        """
        self.metrics.reset()
        total_loss, n_batches = None, 0
        for batch, output in iter_batch_outputs(
            self.model, dataloader, self.callbacks, stage="test"
        ):
            n_batches += 1
            if self.include_loss:
                total_loss = self._compute_loss(batch, output) + total_loss
            if self.metrics:
                self.metrics.update(**self._metric_inputs(batch, output))
        results: dict[str, Any] = {}
        if total_loss is not None:
            results.update((total_loss / n_batches).item())
        if self.metrics and n_batches:
            results.update(self.metrics.result())
        results = {f"{prefix}{k}": v for k, v in results.items()}
        self.callbacks.on_test_end(results)
        return results
