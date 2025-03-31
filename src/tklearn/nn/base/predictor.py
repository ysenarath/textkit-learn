from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Callable, Generator, Generic, TypeVar

import torch
from torch.utils.data import DataLoader

from tklearn.nn.base.module import Module
from tklearn.nn.callbacks.base import Callback, CallbackList, CallbacksMixin
from tklearn.nn.loss import LossDict
from tklearn.utils.array import concat, move_to_device

K = TypeVar("K")
V = TypeVar("V")
L = torch.Tensor | Mapping[str, torch.Tensor] | LossDict | None


class Predictor(CallbacksMixin, Generic[K, V]):
    def __init__(
        self,
        model: Module[K, V],
        dataloader: DataLoader,
        loss: Callable[[K, V], L] | None = None,
        callbacks: CallbackList | Iterable[Callback] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.callbacks = callbacks
        self.loss = loss

    def compute_loss(self, batch: K, output: V) -> LossDict:
        if self.loss is None:
            try:
                batch_loss = self.model.compute_loss(batch, output)
            except NotImplementedError:
                batch_loss = None
        else:
            batch_loss = self.loss(batch, output)
        return LossDict(batch_loss)

    @torch.no_grad()
    def iter_batches(
        self,
    ) -> Generator[tuple[int, K, V, LossDict], None, None]:
        if self.model.training:
            self.model.eval()

        # set the callback params
        callback_params = {}
        if self.callbacks.params is not None:
            callback_params.update(self.callbacks.params)
        callback_params.update({"pred_steps": len(self.dataloader)})

        self.callbacks.set_params(callback_params)
        self.callbacks.set_model(self.model)

        # start the prediction
        self.callbacks.on_predict_begin()

        dataloader_idx = None
        for batch_idx, batch in enumerate(self.dataloader):
            batch = move_to_device(batch, self.model.device, non_blocking=True)
            self.callbacks.on_predict_batch_begin(batch_idx)
            output = self.model.predict_step(
                batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx
            )
            batch_loss = self.compute_loss(batch, output)
            self.callbacks.on_predict_batch_end(batch_idx, logs={})
            yield batch_idx, batch, output, batch_loss

        self.callbacks.on_predict_end()

    def predict(self) -> torch.Tensor:
        self.model.eval()
        logits = None
        for batch_idx, batch, output, batch_loss in self.iter_batches():
            logits = concat((logits, output["logits"]))
        if logits is None:
            msg = "empty prediction results"
            raise ValueError(msg)
        return logits
