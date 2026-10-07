from __future__ import annotations

import contextlib
import copy
import math
from collections.abc import Iterable, Mapping
from typing import Any, Callable, Generator, Union, cast
from weakref import WeakKeyDictionary

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.utils import gather_object, send_to_device
from datasets import IterableDataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from tklearn.metrics import Metric, MetricCollection
from tklearn.nn.callbacks import Callback
from tklearn.nn.module import Loss, Module
from tklearn.nn.schedules import get_scheduler

__all__ = [
    "Trainer",
]

#: A scheduler, the name of a `get_scheduler` schedule, or
#: ``f(optimizer, num_training_steps)`` returning a scheduler.
SchedulerLike = Union[
    LRScheduler, str, Callable[[Optimizer, int], LRScheduler]
]


class _StepRunner(torch.nn.Module):
    # Calls the step hooks through ``forward``, so that what `Accelerator`
    # wraps around forward (DDP gradient sync, autocast) applies to them.

    def __init__(self, module: Module) -> None:
        super().__init__()
        self.module = module
        # training_step
        # predict_step

    def forward(self, step: str, batch: Any) -> Any:
        return getattr(self.module, step)(batch)


class Trainer:
    """Fit, evaluate and run a `Module` with Hugging Face Accelerate.

    The same code runs on CPU, a GPU or Apple MPS (picked in that order of
    preference by Accelerate), with mixed precision, and on several GPUs
    when started with ``accelerate launch``.

    Parameters
    ----------
    model : Module
        The model; see `Module` for the hooks it implements.
    optimizer : Optimizer, optional
        Optimizer over the model's parameters. Required by `fit`.
    lr_scheduler : LRScheduler, str or callable, optional
        Stepped after every optimizer step. Either a scheduler, the name of
        a `get_scheduler` schedule (e.g. ``"linear"``), or
        ``f(optimizer, num_training_steps)`` returning a scheduler. Names
        and callables are built at the start of each `fit` call.
    warmup : int or float, default=0
        Warmup of a named schedule, in optimizer steps (int) or as a
        fraction of all steps (float).
    metrics : Mapping[str, Metric], optional
        Metrics computed by `evaluate`, keyed by result name. Each reads
        its inputs (``y_true``, ``y_pred``, ``y_score``, ...) from the
        `predict_step` outputs.
    callbacks : iterable of Callback, optional
        Callbacks receiving the hooks of `fit`, `evaluate` and `predict`.
        Each hook goes to them in the order given, except for wrapper
        callbacks (`Callback.wrapper`), which get ``*_begin`` hooks before
        and ``*_end`` hooks after the others. Kept as the `callbacks`
        list; a change to it during a hook applies from the next hook.
    max_grad_norm : float, optional
        Clip the gradient norm to this value before each optimizer step.
    mixed_precision : {"no", "fp16", "bf16"}, optional
        Mixed precision mode of the accelerator created by the trainer.
        Defaults to Accelerate's configuration (``"no"`` unless set by
        ``accelerate config`` or ``accelerate launch``).
    gradient_accumulation_steps : int, optional
        Batches whose gradients are accumulated before each optimizer step;
        the losses are averaged over them. Defaults to 1.
    accelerator : Accelerator, optional
        An accelerator to use instead of creating one. Configure mixed
        precision and gradient accumulation on it, not on the trainer.

    Attributes
    ----------
    history : list of dict
        Logs of each epoch of the last `fit` call.
    epochs : int
        Maximum number of epochs of the last `fit` call.
    epoch : int
        Current (zero-based) epoch.
    global_step : int
        Optimizer steps taken in the current `fit` call.
    num_batches : int or None
        Batches in the dataloader of the running loop (a training epoch,
        `evaluate` or `predict`), or None when it has no length.
    grad_norm : Tensor or None
        Total gradient norm before clipping at the last optimizer step,
        when `max_grad_norm` is set; a tensor, so that reading it is the
        only thing that waits for the device.
    should_stop : bool
        Set to True (e.g. by a callback) to stop `fit` after the current
        batch. On several processes, setting it on one stops them all.

    Examples
    --------
    >>> trainer = Trainer(
    ...     model,
    ...     torch.optim.AdamW(model.parameters(), lr=2e-5),
    ...     lr_scheduler="linear",
    ...     warmup=0.1,
    ...     metrics={"f1": F1(average="macro")},
    ... )
    >>> history = trainer.fit(train_loader, valid_loader, epochs=3)
    >>> pd.DataFrame(history)  # loss, valid_loss, valid_f1 per epoch
    >>> trainer.evaluate(test_loader)
    {'loss': 0.41, 'f1': 0.83}
    >>> trainer.predict(test_loader)["y_score"]
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer | None = None,
        *,
        lr_scheduler: SchedulerLike | None = None,
        warmup: int | float = 0,
        metrics: Mapping[str, Metric] | None = None,
        callbacks: Iterable[Callback] | None = None,
        max_grad_norm: float | None = None,
        mixed_precision: str | None = None,
        gradient_accumulation_steps: int | None = None,
        accelerator: Accelerator | None = None,
    ) -> None:
        if not isinstance(model, Module):
            msg = f"expected a tklearn.nn.Module, got {type(model).__name__}"
            raise TypeError(msg)
        if lr_scheduler is not None and optimizer is None:
            msg = "an lr_scheduler needs an optimizer"
            raise ValueError(msg)
        if warmup and not isinstance(lr_scheduler, str):
            msg = (
                "warmup applies to named schedules; pass the warmup to the "
                "scheduler you build instead"
            )
            raise ValueError(msg)
        callbacks = list(callbacks or [])
        for callback in callbacks:
            if not isinstance(callback, Callback):
                msg = f"expected a Callback, got {type(callback).__name__}"
                raise TypeError(msg)
        if accelerator is None:
            accelerator = Accelerator(
                mixed_precision=mixed_precision,
                gradient_accumulation_steps=gradient_accumulation_steps or 1,
            )
        elif mixed_precision is not None or gradient_accumulation_steps:
            msg = (
                "set mixed_precision and gradient_accumulation_steps on the "
                "accelerator that is passed in"
            )
            raise ValueError(msg)
        self._model = model
        self._optimizer = optimizer
        self._accelerator = accelerator
        self.lr_scheduler = lr_scheduler
        self.warmup = warmup
        self.metrics = metrics
        self.callbacks = callbacks
        self.max_grad_norm = max_grad_norm
        self.history: list[dict[str, Any]] = []
        self.epochs = 0
        self.epoch = 0
        self.global_step = 0
        self.num_batches: int | None = None
        self.grad_norm: torch.Tensor | None = None
        self.should_stop = False
        # prepared lazily, on the first fit, evaluate or predict
        self._prepared_runner: torch.nn.Module | None = None
        self._prepared_optimizer: Optimizer | None = None
        self._dataloaders: WeakKeyDictionary[DataLoader, DataLoader] = (
            WeakKeyDictionary()
        )

    # The accelerator wraps the model and optimizer on first use and the
    # trainer keeps the wrapped versions, so all three are fixed.

    @property
    def model(self) -> Module:
        """The model, fixed when the trainer is created."""
        return self._model

    @property
    def optimizer(self) -> Optimizer | None:
        """The optimizer, fixed when the trainer is created."""
        return self._optimizer

    @property
    def accelerator(self) -> Accelerator:
        """The accelerator, fixed when the trainer is created."""
        return self._accelerator

    @property
    def metrics(self) -> MetricCollection:
        """Metrics computed by `evaluate`.

        Each evaluation updates a copy, so these keep their own state.
        """
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: Mapping[str, Metric] | None) -> None:
        self._metrics = MetricCollection(metrics)

    # --- fit ---------------------------------------------------------------

    def fit(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None = None,
        *,
        epochs: int = 1,
    ) -> list[dict[str, Any]]:
        """Train the model.

        Each call is a new run: `history`, `epoch` and `global_step` start
        over, while the model and optimizer keep their state.

        Parameters
        ----------
        train_dataloader : DataLoader
            Training batches, iterated once per epoch.
        eval_dataloader : DataLoader, optional
            Evaluated after every epoch; its results are logged with a
            ``valid_`` prefix (``valid_loss``, ``valid_f1``, ...).
        epochs : int, default=1
            Maximum number of epochs.

        Returns
        -------
        list of dict
            The logs of each epoch (also kept as `history`): the mean
            training ``loss`` and other logged terms, then the evaluation
            results.
        """
        if self.optimizer is None:
            msg = "fit needs an optimizer; pass one to the Trainer"
            raise ValueError(msg)
        if epochs < 1:
            msg = f"epochs must be at least 1, got {epochs}"
            raise ValueError(msg)
        runner, optimizer = self._prepare()
        dataloader = self._prepare_dataloader(train_dataloader)
        scheduler = self._build_scheduler(dataloader, epochs)
        self.history = []
        self.epochs = epochs
        self.epoch = self.global_step = 0
        self.grad_norm = None
        self.should_stop = False
        # a stopped fit can end inside a gradient accumulation window, where
        # Accelerate skips zero_grad; start a new window without gradients
        self.accelerator.step = 0
        self.accelerator.sync_gradients = True
        optimizer.zero_grad()
        self.num_batches = _num_batches(dataloader)
        self._callback("on_train_begin")
        for epoch in range(epochs):
            self.epoch = epoch
            if hasattr(dataloader, "set_epoch"):  # reshuffle each epoch
                # required for synchronized shuffling across epochs
                dataloader = cast(IterableDataset, dataloader)
                dataloader.set_epoch(epoch)
            self.num_batches = _num_batches(dataloader)
            self._callback("on_epoch_begin")
            logs = self._train_epoch(runner, optimizer, scheduler, dataloader)
            if eval_dataloader is not None:
                logs.update(self.evaluate(eval_dataloader, prefix="valid_"))
            self.history.append(logs)
            self._callback("on_epoch_end", logs)
            if self._sync_should_stop():
                break
        self._callback("on_train_end")
        return self.history

    def _train_epoch(
        self,
        runner: _StepRunner,
        optimizer: AcceleratedOptimizer,
        scheduler: LRScheduler | None,
        dataloader: DataLoader,
    ) -> dict[str, float]:
        runner.train()
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for batch in dataloader:
            self._callback("on_train_batch_begin", batch)
            with self.accelerator.accumulate(runner):
                loss, logs = _split_loss(runner("training_step", batch))
                self.accelerator.backward(loss)
                # true at the end of accumulation
                if self.accelerator.sync_gradients:
                    if self.max_grad_norm is not None:
                        self.grad_norm = self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                    self._callback("on_before_optimizer_step")
                    optimizer.step()
                    optimizer.zero_grad()
            if self.accelerator.sync_gradients:
                self.global_step += 1
                # fp16 skips steps whose gradients overflowed
                if scheduler is not None and not optimizer.step_was_skipped:
                    scheduler.step()
            for key, value in logs.items():
                totals[key] = totals.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1
            self._callback("on_train_batch_end", batch, logs)
            if self._sync_should_stop():
                break
        return self._mean(totals, counts)

    def _build_scheduler(
        self, dataloader: DataLoader, epochs: int
    ) -> LRScheduler | None:
        lr_scheduler = self.lr_scheduler
        if lr_scheduler is None or isinstance(lr_scheduler, LRScheduler):
            return lr_scheduler
        try:
            num_batches = len(dataloader)
        except TypeError:
            msg = (
                "a named or callable lr_scheduler needs a dataloader with a "
                "length to count the training steps"
            )
            raise ValueError(msg) from None
        num_steps = epochs * math.ceil(
            num_batches / self.accelerator.gradient_accumulation_steps
        )
        if isinstance(self.lr_scheduler, str):
            return get_scheduler(
                self.lr_scheduler,
                self.optimizer,
                num_steps,
                warmup=self.warmup,
            )
        return self.lr_scheduler(self.optimizer, num_steps)

    # --- evaluate ----------------------------------------------------------

    def evaluate(
        self,
        dataloader: DataLoader,
        *,
        prefix: str = "",
        metrics: Mapping[str, Metric] | None = None,
    ) -> dict[str, Any]:
        """Compute the mean loss and the metrics over a dataloader.

        Parameters
        ----------
        dataloader : DataLoader
            Batches to evaluate on.
        prefix : str, default=""
            Prepended to every result key.
        metrics : Mapping[str, Metric], optional
            Metrics for this call instead of the trainer's `metrics`.

        Returns
        -------
        dict
            ``loss``, the mean of the batch losses (when `predict_step`
            returns one), followed by one entry per metric.
        """
        runner, _ = self._prepare()
        dataloader = self._prepare_dataloader(dataloader)
        # update a copy, so that an evaluation started meanwhile (from a
        # callback, or by another trainer sharing the metrics) cannot reset it
        metrics = copy.deepcopy(
            self.metrics if metrics is None else MetricCollection(metrics)
        )
        metrics.reset()
        input_names = metrics.input_names
        loss_sum, num_losses, num_batches = 0.0, 0, 0
        # restored at the end, as fit evaluates in the middle of its epoch
        previous, self.num_batches = self.num_batches, _num_batches(dataloader)
        self._callback("on_test_begin")
        with self._eval_mode(runner):
            for batch in dataloader:
                self._callback("on_test_batch_begin", batch)
                outputs = runner("predict_step", batch)
                if not isinstance(outputs, Mapping):
                    msg = (
                        "predict_step must return a mapping of metric "
                        f"inputs to be evaluated, got {type(outputs).__name__}"
                    )
                    raise TypeError(msg)
                if outputs.get("loss") is not None:
                    loss_sum += float(outputs["loss"])
                    num_losses += 1
                if input_names:
                    inputs = {
                        k: v for k, v in outputs.items() if k in input_names
                    }
                    metrics.update(**self._gather(inputs))
                num_batches += 1
                self._callback("on_test_batch_end", batch, outputs)
        if num_batches == 0:
            msg = "cannot evaluate on an empty dataloader"
            raise ValueError(msg)
        results: dict[str, Any] = self._mean(
            {"loss": loss_sum}, {"loss": num_losses}
        )
        results.update(metrics.compute())
        results = {f"{prefix}{k}": v for k, v in results.items()}
        self._callback("on_test_end", results)
        self.num_batches = previous
        return results

    # --- predict -----------------------------------------------------------

    def predict(self, dataloader: DataLoader) -> Any:
        """Run `predict_step` over a dataloader and concatenate the outputs.

        Tensors and arrays are concatenated along the first dimension and
        lists are joined; tuples and mappings are concatenated item by item,
        leaving out the ``"loss"`` of a mapping. The result is on the CPU.

        Parameters
        ----------
        dataloader : DataLoader
            Batches to run the model on.

        Returns
        -------
        Any
            The outputs for the whole dataloader, structured like the
            output of one batch.
        """
        runner, _ = self._prepare()
        dataloader = self._prepare_dataloader(dataloader)
        chunks = []
        previous, self.num_batches = self.num_batches, _num_batches(dataloader)
        self._callback("on_predict_begin")
        with self._eval_mode(runner):
            for batch in dataloader:
                self._callback("on_predict_batch_begin", batch)
                outputs = runner("predict_step", batch)
                self._callback("on_predict_batch_end", batch, outputs)
                if isinstance(outputs, Mapping):
                    outputs = {k: v for k, v in outputs.items() if k != "loss"}
                chunks.append(send_to_device(self._gather(outputs), "cpu"))
        self._callback("on_predict_end")
        self.num_batches = previous
        if not chunks:
            msg = "cannot predict on an empty dataloader"
            raise ValueError(msg)
        return _concat(chunks)

    # --- helpers -----------------------------------------------------------

    def _callback(self, hook: str, *args: Any) -> None:
        for callback in _hook_order(self.callbacks, hook):
            getattr(callback, hook)(self, *args)

    def _sync_should_stop(self) -> bool:
        # a callback may stop training on one process only (e.g. on rank 0
        # metrics); stop all of them, or the others hang in the next
        # collective
        if self.accelerator.num_processes > 1:
            if self.should_stop:
                self.accelerator.set_trigger()
            if self.accelerator.check_trigger():
                self.should_stop = True
        return self.should_stop

    def _prepare(self) -> tuple[_StepRunner, AcceleratedOptimizer | None]:
        if self._prepared_runner is None:
            runner = _StepRunner(self.model)
            if self.optimizer is None:
                self._prepared_runner = self.accelerator.prepare(runner)
            else:
                self._prepared_runner, self._prepared_optimizer = (
                    self.accelerator.prepare(runner, self.optimizer)
                )
        return self._prepared_runner, self._prepared_optimizer

    def _prepare_dataloader(self, dataloader: DataLoader) -> DataLoader:
        if not isinstance(dataloader, DataLoader):
            msg = (
                f"expected a torch DataLoader, got {type(dataloader).__name__}"
                "; wrap ready-made batches with "
                "DataLoader(batches, batch_size=None)"
            )
            raise TypeError(msg)
        # prepare each dataloader once, so persistent workers are reused
        prepared = self._dataloaders.get(dataloader)
        if prepared is None:
            prepared = self.accelerator.prepare(dataloader)
            self._dataloaders[dataloader] = prepared
        return prepared

    @contextlib.contextmanager
    def _eval_mode(self, runner: torch.nn.Module) -> Generator[None]:
        """Context manager to temporarily set the model to evaluation mode."""
        training = self.model.training
        runner.eval()
        try:
            with torch.no_grad():
                yield
        finally:
            runner.train(training)

    def _gather(self, data: Any) -> Any:
        # collect the outputs of every process, without the duplicates
        # that pad the last batch
        if self.accelerator.num_processes == 1:
            return data
        if isinstance(data, Mapping):
            return {k: self._gather(v) for k, v in data.items()}
        return self.accelerator.gather_for_metrics(data)

    def _mean(
        self, totals: dict[str, float], counts: dict[str, int]
    ) -> dict[str, float]:
        """Mean of each value over the batches that logged it."""
        if self.accelerator.num_processes > 1:
            # processes can log different keys, but must reduce tensors of
            # the same keys in the same order; gather_object orders the keys
            # by rank, then by first appearance
            gathered = gather_object([list(totals)])
            keys = list(dict.fromkeys(k for ks in gathered for k in ks))
            values = torch.tensor(
                [totals.get(k, 0.0) for k in keys]
                + [counts.get(k, 0) for k in keys],
                dtype=torch.float64,
                device=self.accelerator.device,
            )
            values = self.accelerator.reduce(values, reduction="sum").tolist()
            totals = dict(zip(keys, values[: len(keys)]))
            counts = dict(zip(keys, values[len(keys) :]))
        return {k: v / counts[k] for k, v in totals.items() if counts[k]}


def _hook_order(callbacks: list[Callback], hook: str) -> tuple[Callback, ...]:
    """The callbacks in the order they get `hook`: wrappers first on
    ``*_begin`` hooks and last, in reverse, on ``*_end`` hooks.

    A copy, so that a callback changing `Trainer.callbacks` during the
    hook changes who gets the next hook, not this one.
    """
    wrappers = tuple(c for c in callbacks if c.wrapper)
    if wrappers and hook.endswith("_begin"):
        others = tuple(c for c in callbacks if not c.wrapper)
        return (*wrappers, *others)
    if wrappers and hook.endswith("_end"):
        others = tuple(c for c in callbacks if not c.wrapper)
        return (*others, *reversed(wrappers))
    return tuple(callbacks)


def _num_batches(dataloader: DataLoader) -> int | None:
    """Length of a dataloader; None for an iterable dataset without one."""
    try:
        return len(dataloader)
    except TypeError:
        return None


def _split_loss(loss: Loss) -> tuple[torch.Tensor, dict[str, float]]:
    """The loss to minimize and the values to log."""
    if isinstance(loss, Mapping):
        if loss.get("loss") is None:
            msg = "training_step returned a mapping without a 'loss'"
            raise ValueError(msg)
        terms = dict(loss)
    else:
        terms = {"loss": loss}
    total = terms["loss"]
    if not isinstance(total, torch.Tensor) or total.numel() != 1:
        msg = f"the loss must be a scalar tensor, got {total!r}"
        raise ValueError(msg)
    logs = {}
    for key, value in terms.items():
        if isinstance(value, torch.Tensor):
            value = value.detach()
        try:
            logs[key] = float(value)
        except (TypeError, ValueError, RuntimeError):
            msg = f"training_step returned {key!r}, which is not a scalar"
            raise ValueError(msg) from None
    return total, logs


def _concat(chunks: list[Any]) -> Any:
    """Concatenate per-batch outputs of the same structure."""
    first = chunks[0]
    if first is None:
        return None
    if isinstance(first, torch.Tensor):
        if first.ndim == 0:
            msg = (
                "cannot concatenate scalar outputs; predict_step must return "
                "one value per example"
            )
            raise ValueError(msg)
        return torch.cat(chunks)
    if isinstance(first, np.ndarray):
        return np.concatenate(chunks)
    if isinstance(first, Mapping):
        return {k: _concat([chunk[k] for chunk in chunks]) for k in first}
    if isinstance(first, tuple):
        values = [_concat(list(items)) for items in zip(*chunks)]
        if hasattr(first, "_fields"):  # namedtuple
            return type(first)(*values)
        return tuple(values)
    if isinstance(first, list):
        return [item for chunk in chunks for item in chunk]
    msg = f"cannot concatenate outputs of type {type(first).__name__}"
    raise TypeError(msg)
