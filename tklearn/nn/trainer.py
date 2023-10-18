"""Model trainer for PyTorch models.

Notes
-----
This trainer is designed to be compatible with HuggingFace
datasets and torch style datasets.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Union, Callable, Any
import functools

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from transformers import get_scheduler, PreTrainedModel
from transformers.utils import ModelOutput
from accelerate import Accelerator

from tklearn.nn.base import BaseTrainer
from tklearn.nn.losses import AutoLoss
from tklearn.nn.dataset import TrainerDataset
from tklearn.nn.evaluator import Evaluator
from tklearn.exceptions import EarlyStoppingException
from tklearn.utils import _utils, logging
from tklearn.config import configurable

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

__all__ = [
    "Trainer",
]

logger = logging.get_logger(__name__)


def move_to_device(data: Any, device, detach=False, numpy=False) -> Any:
    """Move data to device.

    This function is a recursive function that will
    move all tensors in a mapping or sequence to
    the specified device.

    Notes
    -----
    If the data cannot be cannot be moved to the
    specified device, then the data is returned
    as is.

    Parameters
    ----------
    data : torch.Tensor or Mapping or Sequence or typing.Any
        The data to move to device.
    device : torch.device or str
        The device to move the data to.
    detach : bool, optional
        Whether to detach the data from the computation graph.
        Defaults to False.
    numpy : bool, optional
        Whether to convert the data to a numpy array.

    Returns
    -------
    torch.Tensor or Mapping or Sequence or typing.Any
        The data moved to device.
    """
    if torch.is_tensor(data):
        if detach:
            data = data.detach()
        data = data.to(device)
        if numpy:
            data = data.numpy()
        return data
    elif isinstance(data, Mapping):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, Sequence) and not isinstance(data, str):
        # try moving each element to device
        # assuming that they are already tensors
        return [move_to_device(value, device) for value in data]
    # if we cannot move the data to device
    # then just return the data as is
    return data


def build_criterion(
    model: torch.nn.Module,
    loss: Union[str, Callable] = None,
) -> Callable:
    if isinstance(loss, str):
        criterion_cls = getattr(torch.nn, loss)
        criterion = criterion_cls()
    elif loss is None and isinstance(model, PreTrainedModel):
        criterion = AutoLoss.from_pretrained(model)
    else:
        raise ValueError(f"unsupported loss: {loss}")
    return criterion


def build_optimizer(
    model: torch.nn.Module,
    optimizer: Optimizer,
) -> Optimizer:
    params = model.parameters()
    if isinstance(optimizer, str):
        optimizer_cls = getattr(torch.optim, optimizer)
        optimizer_args = {}
        optimizer_args.setdefault("lr", 1e-5)
        optimizer = optimizer_cls(params, **optimizer_args)
    elif isinstance(optimizer, functools.partial):
        optimizer = optimizer(params)
    return optimizer


def build_lr_scheduler(
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    num_training_steps: int,
) -> LRScheduler:
    if isinstance(lr_scheduler, str):
        lr_scheduler = get_scheduler(
            name=lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
    elif isinstance(lr_scheduler, functools.partial):
        lr_scheduler = lr_scheduler(optimizer)
    return lr_scheduler


def train_step(
    model: torch.nn.Module,
    x: dict,
    y,
    criterion: Union[Callable, None] = None,
    optimizer: Union[Optimizer, None] = None,
    lr_scheduler: Union[LRScheduler, None] = None,
    accelerator: Union[Accelerator, None] = None,
    device: Union[torch.device, str] = "cpu",
    clip_grad_strategy: str = "norm",
    clip_grad_value: float = 1.0,
):
    if not model.training:
        # change the model if needed
        model.train()
    # move the data to device
    if accelerator is None:
        x = move_to_device(x, device)
    # pop the labels from x
    if y is None:
        # pop label from x
        y = x["labels"]
        # del x["labels"]
    # move to device (if needed)
    if accelerator is None:
        y = move_to_device(y, device)
    # forward pass
    outputs = model(**x)
    # HuggingFase transformers output
    hft_output = isinstance(outputs, ModelOutput)
    # compute the prediction error
    if hft_output and hasattr(outputs, "logits"):
        # huggingface pretrained model output
        logits_or_proba = getattr(outputs, "logits")
    else:
        logits_or_proba = outputs
    loss_val: torch.Tensor = criterion(logits_or_proba, y)
    if hasattr(outputs, "loss") and outputs.loss:
        assert (
            outputs.loss == loss_val
        ), f"loss value mismatch, found {outputs.loss}, expected {loss_val}"
    # backpropagation
    if accelerator is None:
        loss_val.backward()
    else:
        accelerator.backward(loss_val)
    clip_grad_type = clip_grad_strategy
    if clip_grad_type == "norm":
        clip_grad_value = clip_grad_value
        # Clips gradient norm of an iterable of parameters.
        torch.nn.utils.clip_grad.clip_grad_norm_(
            model.parameters(),
            clip_grad_value,
        )
    elif clip_grad_type == "value":
        clip_grad_value = clip_grad_value
        # Clips gradient of an iterable of parameters at specified value.
        torch.nn.utils.clip_grad.clip_grad_value_(
            model.parameters(),
            clip_grad_value,
        )
    optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()
    return loss_val


@configurable
class Trainer(BaseTrainer):
    """Model trainer for PyTorch models."""

    model: torch.nn.Module

    def __init__(
        self,
        model: torch.nn.Module,
        loss: Union[str, Callable] = None,
        optimizer: Union[str, Optimizer, functools.partial] = "AdamW",
        lr_scheduler: Union[str, LRScheduler, None] = "linear",
        clip_grad_strategy: str = "norm",
        clip_grad_value: float = 1.0,
        epochs: int = 3,
        batch_size: int = 8,
        shuffle: bool = False,
        accelerator: Union[bool, Accelerator] = True,
        device: Union[torch.device, str, None] = None,
        callbacks=None,
        verbose=True,
    ):
        """Initialize the trainer."""
        super(Trainer, self).__init__(model=model, callbacks=callbacks, verbose=verbose)
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.clip_grad_strategy = clip_grad_strategy
        self.clip_grad_value = clip_grad_value
        self.accelerator = Accelerator() if accelerator is True else accelerator
        self.epochs = epochs
        self.device = device

    @property
    def device(self) -> Union[torch.device, str]:
        """The device to use for training."""
        return self._device

    @device.setter
    def device(self, value):
        """Set the device to use for training."""
        if value is None:
            value = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self._device = value

    def fit(
        self,
        x,
        y=None,
        evaluator: Union[Evaluator, None] = None,
    ):
        dataset = TrainerDataset(x=x, y=y)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )
        self.callbacks.set_trainer(self)
        num_batches = len(dataloader)
        num_training_steps = num_batches * self.epochs
        criterion = build_criterion(self.model, self.loss)
        optimizer = build_optimizer(self.model, self.optimizer)
        if self.accelerator:
            dataloader, model, optimizer = self.accelerator.prepare(
                dataloader, self.model, optimizer
            )
        else:
            model = self.model.to(self.device)
        self.model.train()
        lr_scheduler = build_lr_scheduler(
            self.lr_scheduler, optimizer, num_training_steps
        )  # set the lr scheduler
        callback_params = {
            "epochs": self.epochs,
            "steps": num_training_steps,
            "verbose": self.verbose,
        }
        self.callbacks.set_params(callback_params)
        self.callbacks.on_train_begin()
        for epoch in range(self.epochs):
            try:
                epoch_logs = {}
                self.callbacks.on_epoch_begin(epoch + 1, logs=epoch_logs)
                running_loss = 0.0
                for batch_idx, batch in enumerate(dataloader):
                    batch_log = {}
                    self.callbacks.on_train_batch_begin(batch_idx + 1, logs=batch_log)
                    x_batch, y_batch = batch["x"], batch.get("y", None)
                    # train the model
                    loss = train_step(
                        model,
                        x_batch,
                        y_batch,
                        criterion=criterion,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        accelerator=self.accelerator,
                    )
                    batch_loss = loss.item()
                    running_loss += batch_loss
                    batch_log.update(dict(loss=batch_loss))
                    self.callbacks.on_train_batch_end(batch_idx + 1, logs=batch_log)
                # epoch loss is average loss per batch
                epoch_logs.update(dict(loss=running_loss / num_batches))
                if evaluator is not None:
                    eval_results = evaluator.evaluate(self)
                    epoch_logs.update(eval_results)
                self.callbacks.on_epoch_end(epoch + 1, logs=epoch_logs)
            except EarlyStoppingException:
                break
        final_logs = {}
        self.callbacks.on_train_end(final_logs)
        return self

    def _predict_batch_iter(self, x):
        # convert x to dataset if needed
        dataset = x if isinstance(x, TrainerDataset) else TrainerDataset(x=x)
        del x
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        if self.accelerator is None:
            model = self.model.to(self.device)
        else:
            dataloader, model = self.accelerator.prepare(dataloader, self.model)
        if model.training:
            # change the model if needed
            model.eval()
        callback_params = {
            "predict_steps": len(dataloader),
            "verbose": self.verbose,
        }
        self.callbacks.set_params(callback_params)
        self.callbacks.on_predict_begin(logs={})
        for batch, batch_data in enumerate(dataloader):
            self.callbacks.on_predict_batch_begin(batch, logs={})
            x_batch = batch_data["x"]
            # if "labels" in x_batch:
            #     del x_batch["labels"]
            if self.accelerator is None:
                x_batch = move_to_device(x_batch, self.device)
            with torch.no_grad():
                output = model(**x_batch)
            if hasattr(output, "logits"):
                # huggingface pretrained model output
                output = getattr(output, "logits")
            yield batch_data, move_to_device(
                output, device="cpu", detach=True, numpy=False
            )
            self.callbacks.on_predict_batch_end(batch, logs={})
        self.callbacks.on_predict_end(logs={})

    def predict_logits(self, x) -> Any:
        """this will output the logits -- need to convert to probabilities"""
        y_pred = None
        for _, y_pred_batch in self._predict_batch_iter(x):
            y_pred = _utils.concat(y_pred, y_pred_batch)
        return y_pred

    def predict_proba(self, x) -> Any:
        """this will output the logits -- need to convert to probabilities"""
        y_pred = None
        for _, y_pred_batch in self._predict_batch_iter(x):
            y_pred = _utils.concat(y_pred, y_pred_batch)
        return y_pred

    def predict(self, x):
        return self.predict_proba(x)
