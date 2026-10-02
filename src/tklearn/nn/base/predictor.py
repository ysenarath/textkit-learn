from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Generic, Literal, TypeVar

import torch

from tklearn.nn.base.module import Module
from tklearn.nn.callbacks.base import Callback, CallbackList, CallbacksMixin
from tklearn.utils.array import concat, detach, move_to_device

__all__ = [
    "Predictor",
]

BatchT = TypeVar("BatchT")
OutputT = TypeVar("OutputT")


def _len_or_none(dataloader: Iterable) -> int | None:
    try:
        return len(dataloader)
    except TypeError:
        return None


def iter_batch_outputs(
    model: Module[BatchT, OutputT],
    dataloader: Iterable[BatchT],
    callbacks: CallbackList,
    stage: Literal["test", "predict"],
) -> Iterator[tuple[BatchT, OutputT]]:
    """Run `model.predict_step` over a dataloader without gradients.

    Puts the model in eval mode, moves each batch to the model's device and
    fires the ``on_{stage}_*`` callback hooks around the loop and each batch.
    Yields ``(batch, output)`` pairs on the model's device.
    """
    model.eval()
    callbacks.set_model(model)
    callbacks.set_params({
        **callbacks.params,
        f"{stage}_steps": _len_or_none(dataloader),
    })
    getattr(callbacks, f"on_{stage}_begin")({})
    batch_begin = getattr(callbacks, f"on_{stage}_batch_begin")
    batch_end = getattr(callbacks, f"on_{stage}_batch_end")
    device = model.device
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = move_to_device(batch, device, non_blocking=True)
            batch_begin(batch_idx, {})
            output = model.predict_step(batch)
            yield batch, output
            batch_end(batch_idx, {})
    # 'on_test_end' is fired by the Evaluator once results are available
    if stage == "predict":
        callbacks.on_predict_end({})


class Predictor(CallbacksMixin, Generic[BatchT, OutputT]):
    """Run a model over a dataloader and collect its outputs.

    Parameters
    ----------
    model : Module
        The model; its `predict_step` is called on every batch.
    callbacks : Callback or iterable of Callback, optional
        Callbacks receiving the ``on_predict_*`` hooks.

    Examples
    --------
    >>> predictor = Predictor(model)
    >>> logits = predictor.predict(test_loader, output_key="logits")
    """

    def __init__(
        self,
        model: Module[BatchT, OutputT],
        callbacks: CallbackList | Iterable[Callback] | Callback | None = None,
    ) -> None:
        self.model = model
        self.callbacks = callbacks

    def iter_outputs(
        self, dataloader: Iterable[BatchT]
    ) -> Iterator[tuple[BatchT, OutputT]]:
        """Yield ``(batch, output)`` for each batch, on the model's device."""
        return iter_batch_outputs(
            self.model, dataloader, self.callbacks, stage="predict"
        )

    def predict(
        self, dataloader: Iterable[BatchT], output_key: str | None = None
    ) -> Any:
        """Predict over a dataloader and concatenate the outputs on the CPU.

        Parameters
        ----------
        dataloader : iterable
            Batches to run the model on.
        output_key : str, optional
            Return only this field of each output (e.g. ``"logits"``).
            By default the whole output is concatenated, keeping its type.

        Returns
        -------
        Any
            The concatenated outputs (or output field) for the whole
            dataloader.
        """
        outputs = []
        for _, output in self.iter_outputs(dataloader):
            if output_key is not None:
                output = output[output_key]
            outputs.append(move_to_device(detach(output), "cpu"))
        if not outputs:
            msg = "cannot predict on an empty dataloader"
            raise ValueError(msg)
        return concat(outputs)
