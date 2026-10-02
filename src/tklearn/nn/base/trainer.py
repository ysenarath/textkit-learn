from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Generic, TypeVar

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from tklearn.nn.base.evaluator import Evaluator
from tklearn.nn.base.module import Module
from tklearn.nn.callbacks.base import Callback, CallbackList, CallbacksMixin
from tklearn.nn.callbacks.history import History
from tklearn.nn.loss import LossDict, LossFunction
from tklearn.nn.optim import get_scheduler
from tklearn.utils.array import move_to_device

__all__ = [
    "Trainer",
]

BatchT = TypeVar("BatchT")
OutputT = TypeVar("OutputT")


class Trainer(CallbacksMixin, Generic[BatchT, OutputT]):
    """Train a model with an optimizer, optional scheduler and callbacks.

    Parameters
    ----------
    model : Module
        The model to train. The loss comes from `model.training_step`, or
        from ``loss(batch, model.predict_step(batch))`` when `loss` is given.
    optimizer : Optimizer
        Optimizer over the model's parameters.
    loss : callable, optional
        ``loss(batch, output)``, overriding the model's own loss.
    lr_scheduler : LRScheduler or str, optional
        A scheduler, or the name of a `transformers` schedule (e.g.
        ``"linear"``) built for the run's total number of steps. It is
        stepped after every batch.
    lr_scheduler_kwargs : Mapping, optional
        Extra arguments for a named scheduler, e.g. ``num_warmup_steps``.
    clip_grad_norm : float, optional
        Clip the global gradient norm to this value before each step.
    evaluator : Evaluator, optional
        Evaluates the model after each epoch when `fit` gets an
        ``eval_dataloader``.
    callbacks : Callback or iterable of Callback, optional
        Callbacks receiving the training hooks. A callback can stop training
        after the current epoch by setting ``trainer.stop_training = True``.

    Examples
    --------
    >>> trainer = Trainer(
    ...     model,
    ...     torch.optim.AdamW(model.parameters(), lr=2e-5),
    ...     lr_scheduler="linear",
    ...     evaluator=Evaluator(model, metrics={"f1": F1(average="macro")}),
    ...     callbacks=[EarlyStopping(monitor="valid_loss", patience=2)],
    ... )
    >>> history = trainer.fit(train_loader, epochs=10, eval_dataloader=valid_loader)
    >>> history.to_pandas()
    """

    def __init__(
        self,
        model: Module[BatchT, OutputT],
        optimizer: Optimizer,
        *,
        loss: LossFunction[BatchT, OutputT] | None = None,
        lr_scheduler: LRScheduler | str | None = None,
        lr_scheduler_kwargs: Mapping[str, Any] | None = None,
        clip_grad_norm: float | None = None,
        evaluator: Evaluator[BatchT, OutputT] | None = None,
        callbacks: CallbackList | Iterable[Callback] | Callback | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.clip_grad_norm = clip_grad_norm
        self.evaluator = evaluator
        self.callbacks = callbacks
        self.stop_training = False

    def _build_scheduler(
        self, epochs: int, steps_per_epoch: int
    ) -> LRScheduler | None:
        if isinstance(self.lr_scheduler, str):
            return get_scheduler(
                name=self.lr_scheduler,
                optimizer=self.optimizer,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                **(self.lr_scheduler_kwargs or {}),
            )
        return self.lr_scheduler

    def _compute_training_loss(self, batch: BatchT) -> LossDict:
        if self.loss is None:
            return LossDict(self.model.training_step(batch))
        return LossDict(self.loss(batch, self.model.predict_step(batch)))

    def _train_batch(
        self,
        batch: BatchT,
        batch_idx: int,
        callbacks: CallbackList,
        scheduler: LRScheduler | None,
    ) -> LossDict:
        callbacks.on_train_batch_begin(batch_idx, {})
        loss = self._compute_training_loss(batch)
        callbacks.on_before_zero_grad(self.optimizer)
        self.optimizer.zero_grad()
        callbacks.on_before_backward({})
        loss.backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=float(self.clip_grad_norm)
            )
        callbacks.on_after_backward({})
        callbacks.on_before_optimizer_step(self.optimizer)
        self.optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss = loss.detach()
        callbacks.on_train_batch_end(batch_idx, loss.item().to_dict())
        return loss

    def fit(
        self,
        dataloader: Iterable[BatchT],
        epochs: int = 1,
        *,
        eval_dataloader: Iterable[BatchT] | None = None,
        eval_prefix: str = "valid_",
    ) -> History:
        """Train the model.

        Parameters
        ----------
        dataloader : iterable
            Training batches; iterated once per epoch.
        epochs : int, default=1
            Maximum number of epochs.
        eval_dataloader : iterable, optional
            Evaluated with `evaluator` after each epoch.
        eval_prefix : str, default="valid_"
            Prefix for evaluation results in the epoch logs, so that
            ``loss`` (training) and ``valid_loss`` do not collide.

        Returns
        -------
        History
            Per-epoch logs; ``history.to_pandas()`` gives a DataFrame.
        """
        if eval_dataloader is not None and self.evaluator is None:
            msg = (
                "'eval_dataloader' was given but the trainer has no evaluator"
            )
            raise ValueError(msg)
        steps_per_epoch = len(dataloader)
        scheduler = self._build_scheduler(epochs, steps_per_epoch)
        history = History()
        callbacks = CallbackList([*self.callbacks, history])
        callbacks.set_model(self.model)
        callbacks.set_trainer(self)
        callbacks.set_params({
            "epochs": epochs,
            "steps": steps_per_epoch,
            "batch_size": getattr(dataloader, "batch_size", None),
        })
        self.stop_training = False
        device = self.model.device
        callbacks.on_train_begin({})
        epoch_logs: dict[str, Any] = {}
        for epoch in range(epochs):
            callbacks.on_epoch_begin(epoch, {})
            self.model.train()
            total_loss, n_batches = None, 0
            for batch_idx, batch in enumerate(dataloader):
                batch = move_to_device(batch, device, non_blocking=True)
                loss = self._train_batch(
                    batch, batch_idx, callbacks, scheduler
                )
                total_loss = loss + total_loss
                n_batches += 1
            epoch_logs = {}
            if total_loss is not None:
                epoch_logs.update((total_loss / n_batches).item())
            if eval_dataloader is not None:
                epoch_logs.update(
                    self.evaluator.evaluate(
                        eval_dataloader, prefix=eval_prefix
                    )
                )
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break
        callbacks.on_train_end(epoch_logs)
        callbacks.set_trainer(None)
        return history
