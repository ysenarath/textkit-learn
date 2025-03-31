from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Callable, Generic, TypeVar

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader

from tklearn.nn.base.evaluator import Evaluator
from tklearn.nn.base.module import Module
from tklearn.nn.callbacks.base import Callback, CallbackList, CallbacksMixin
from tklearn.nn.callbacks.history import History
from tklearn.nn.loss import LossDict
from tklearn.nn.optim import LRSchedulerConfig, get_scheduler
from tklearn.utils.array import move_to_device

K = TypeVar("K")
V = TypeVar("V")
L = torch.Tensor | Mapping[str, torch.Tensor] | LossDict | None


class Trainer(CallbacksMixin, Generic[K, V]):
    def __init__(
        self,
        model: Module[K, V],
        dataloader: DataLoader,
        optimizer: Optimizer,
        loss: Callable[[K, V], L] | None = None,
        epochs: int = 1,
        lr_scheduler: LRScheduler | LRSchedulerConfig | str | None = None,
        lr_scheduler_kwargs: Mapping[str, Any] | None = None,
        clip_grad_norm: (
            int
            | float
            | bool
            | dict[str, Any]
            | Callable[[torch.Tensor | Iterable[torch.Tensor]], None]
        ) = None,
        evaluator: Evaluator | None = None,
        callbacks: CallbackList | Iterable[Callback] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.epochs = epochs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.clip_grad_norm = clip_grad_norm
        self.loss = loss
        self.evaluator = evaluator
        self.callbacks = callbacks

    def _training_step_grad(
        self,
        batch: K,
        batch_idx: int | None = None,
        dataloader_idx: int | None = None,
    ) -> LossDict:
        if self.loss is None:
            # if loss is not defined, try to use the training_step method
            # if that is not implemented, try to use the predict_step method
            # with compute_loss
            try:
                batch_loss = self.model.training_step(
                    batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx
                )
            except NotImplementedError:
                try:
                    batch_output = self.model.predict_step(
                        batch,
                        batch_idx=batch_idx,
                        dataloader_idx=dataloader_idx,
                    )
                    batch_loss = self.model.compute_loss(batch, batch_output)
                except NotImplementedError:
                    batch_loss = None
        else:
            # if the loss is provided externally, use that instead with the
            # predict_step method
            batch_output = self.model.predict_step(
                batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx
            )
            batch_loss = self.loss(batch, batch_output)
        if not isinstance(batch_loss, LossDict):
            batch_loss = LossDict(batch_loss)

        self.callbacks.on_before_zero_grad(self.optimizer)

        self.optimizer.zero_grad()

        self.callbacks.on_before_backward()

        batch_loss.backward()

        if self.clip_grad_norm:
            if isinstance(self.clip_grad_norm, (int, float, bool)):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=float(self.clip_grad_norm),
                )
            elif isinstance(self.clip_grad_norm, Mapping):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), **self.clip_grad_norm
                )
            else:
                self.clip_grad_norm(self.model.parameters())

        self.callbacks.on_after_backward()

        return batch_loss

    def _training_step(
        self,
        batch: K,
        batch_idx: int | None = None,
        dataloader_idx: int | None = None,
        device: torch.device | str | None = None,
    ) -> LossDict[torch.Tensor]:
        if device is None:
            device = self.model.device

        batch = move_to_device(batch, device, non_blocking=True)

        if not self.model.training:
            self.model.train()

        # pre grad calculation here
        self.callbacks.on_train_batch_begin(batch_idx)

        batch_loss = self._training_step_grad(
            batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx
        )

        # post grad calculation here
        self.callbacks.on_before_optimizer_step(self.optimizer)

        self.optimizer.step()

        if getattr(self, "_lr_scheduler", None) is not None:
            # `_batch_lr_scheduler` is set during self.train() method
            self._lr_scheduler.step()

        self.callbacks.on_train_batch_end(batch_idx, logs={})

        return batch_loss.detach()

    def train(self) -> History:
        # Move the model to device
        device = self.model.device
        move_to_device(self.model, device, non_blocking=True)

        # get the number of steps per epoch
        steps_per_epoch = len(self.dataloader)

        # create the learning rate scheduler
        if isinstance(self.lr_scheduler, str):
            self._lr_scheduler = get_scheduler(
                name=self.lr_scheduler,
                optimizer=self.optimizer,
                epochs=self.epochs,
                steps_per_epoch=steps_per_epoch,
                **(self.lr_scheduler_kwargs or {}),
            )
        else:
            self._lr_scheduler = self.lr_scheduler

        # change to train mode
        self.model.train()

        try:
            history = self.callbacks[History]
        except KeyError:
            # add the history callback
            history = History()
            self.callbacks.append(history)

        # set the callback params
        params = {
            "batch_size": self.dataloader.batch_size,
            "epochs": self.epochs,
            "steps": steps_per_epoch,
        }
        self.callbacks.set_params(params)
        self.callbacks.set_model(self.model)
        self.callbacks.on_train_begin()

        epoch_logs = {}
        for epoch_idx in range(self.epochs):
            self.callbacks.on_epoch_begin(epoch_idx)

            total_loss, batch_idx = None, 0
            dataloader_idx = None
            for batch_idx, batch in enumerate(self.dataloader):
                batch_loss = self._training_step(
                    batch,
                    batch_idx=batch_idx,
                    dataloader_idx=dataloader_idx,
                    device=device,
                )
                total_loss = batch_loss + total_loss
            epoch_logs = {}
            if total_loss is not None:
                # average the loss (dict)
                epoch_logs = (total_loss / (batch_idx + 1)).item().to_dict()

            # Run evaluation if configured
            if self.evaluator:
                eval_results = self.evaluator.evaluate()
                epoch_logs.update(eval_results)

            # End the epoch and update callbacks
            self.callbacks.on_epoch_end(epoch_idx, logs=epoch_logs)

            if getattr(self.model, "stop_training", False):
                break
        self.callbacks.on_train_end(epoch_logs)
        return history
