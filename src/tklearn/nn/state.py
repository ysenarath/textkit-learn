"""The state of one `Trainer.fit`, `evaluate` or `predict` call, and the
loops that run it."""

from __future__ import annotations

import contextlib
import copy
import math
from collections.abc import Generator, Mapping
from typing import TYPE_CHECKING, Any, cast

import torch
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.utils import send_to_device
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from tklearn.metrics import Metric, MetricCollection
from tklearn.nn._outputs import (
    BatchLogs,
    RunningMean,
    concat,
    example_losses,
    split_loss,
)
from tklearn.nn.callbacks import Callback, Dispatcher
from tklearn.nn.module import Module
from tklearn.nn.schedules import get_scheduler

if TYPE_CHECKING:
    from tklearn.nn.trainer import Trainer

__all__ = [
    "ModuleWrapper",
    "TrainerState",
]


class ModuleWrapper(torch.nn.Module):
    # Calls the step hooks through ``forward``, so that what `Accelerator`
    # wraps around forward (DDP gradient sync, autocast) applies to them.

    def __init__(self, module: Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, step: str, batch: Any) -> Any:
        return getattr(self.module, step)(batch)


class TrainerState:
    """The state of one `Trainer.fit`, `evaluate` or `predict` call.

    The trainer creates one when the call starts, and drops it when the
    call returns; nothing of it is written back to the trainer, so the
    next call starts over. Callbacks read it as `Callback.state`.

    Attributes
    ----------
    trainer : Trainer
        The configuration of the run.
    model : Module
        The trainer's model.
    callbacks : list of Callback
        The callbacks of the run, a copy of the trainer's; a change to it
        during a hook applies from the next hook of this run.
    optimizer : Optimizer or None
        The optimizer this `fit` built from the trainer's factory; None
        for `evaluate` and `predict`.
    scheduler : LRScheduler or None
        The learning rate scheduler this `fit` built, if any.
    history : list of dict
        Logs of each epoch so far.
    epochs : int
        Maximum number of epochs; 0 outside `fit`.
    epoch : int
        Current (zero-based) epoch.
    global_step : int
        Optimizer steps taken; steps that fp16 skipped for overflowing
        gradients do not count.
    num_batches : int or None
        Batches in the dataloader of the running loop (a training epoch,
        an evaluation or a prediction), or None when it has no length.
    grad_norm : Tensor or None
        Total gradient norm before clipping at the last optimizer step,
        when `Trainer.max_grad_norm` is set; a tensor, so that reading it
        is the only thing that waits for the device.
    should_stop : bool
        Set to True (e.g. by a callback) to stop `fit` after the current
        batch. On several processes, setting it on one stops them all.
    """

    def __init__(self, trainer: Trainer) -> None:
        self.trainer = trainer
        self.model: Module = trainer.model
        self.callbacks: list[Callback] = list(trainer.callbacks)
        self.optimizer: Optimizer | None = None
        self.scheduler: LRScheduler | None = None
        self.history: list[dict[str, Any]] = []
        self.epochs = 0
        self.epoch = 0
        self.global_step = 0
        self.num_batches: int | None = None
        self.grad_norm: torch.Tensor | None = None
        self.should_stop = False
        # what the accelerator prepared, while the run lasts; see _run
        self._accelerator: Accelerator | None = None
        self._dispatcher: Dispatcher | None = None
        self._runner: ModuleWrapper | None = None
        self._prepared_optimizer: AcceleratedOptimizer | None = None
        # prepared dataloaders, so that the evaluation after each epoch
        # reuses the persistent workers of the one before
        self._dataloaders: dict[DataLoader, DataLoader] = {}

    @property
    def accelerator(self) -> Accelerator:
        """The accelerator of the run; only available while it lasts."""
        if self._accelerator is None:
            msg = (
                "the accelerator exists only during a fit, evaluate or predict"
            )
            raise RuntimeError(msg)
        return self._accelerator

    # --- for callbacks -----------------------------------------------------

    def evaluate(
        self,
        dataloader: DataLoader,
        *,
        prefix: str = "",
        metrics: Mapping[str, Metric] | None = None,
    ) -> dict[str, Any]:
        """Evaluate within this run, as `Trainer.evaluate` does.

        For callbacks: it reuses the run's accelerator and prepared model,
        and gives the callbacks its ``on_test_*`` hooks. `num_batches` is
        restored when it returns.
        """
        previous = self.num_batches
        try:
            return self._evaluate(dataloader, prefix, metrics)
        finally:
            self.num_batches = previous

    def predict(self, dataloader: DataLoader) -> Any:
        """Predict within this run, as `Trainer.predict` does.

        For callbacks, as `evaluate`.
        """
        previous = self.num_batches
        try:
            return self._predict(dataloader)
        finally:
            self.num_batches = previous

    # --- the run -----------------------------------------------------------

    @contextlib.contextmanager
    def _run(self, training: bool) -> Generator[None]:
        """Prepare the run, and release what it prepared when it ends.

        Builds the optimizer when training, creates the accelerator and
        prepares the model (and optimizer) with it, and gives the callbacks
        this state until the run ends. When the run raises, the callbacks
        get `on_exception`.
        """
        trainer = self.trainer
        # gives the callbacks this state, or raises when one is in another
        # run
        dispatcher = Dispatcher(self)
        try:
            # Accelerate fixes cpu and mixed_precision for the process at
            # the first accelerator, and raises when a later one asks
            # otherwise
            accelerator = Accelerator(**trainer.accelerator_args)
            runner = ModuleWrapper(self.model)
            if training:
                factory = cast(Any, trainer.optimizer)
                self.optimizer = factory(self.model.parameters())
                runner, optimizer = accelerator.prepare(runner, self.optimizer)
            else:
                # device placement and autocast, without the distributed
                # wrapping (and its parameter broadcast) only training needs
                runner = accelerator.prepare_model(
                    runner, evaluation_mode=True
                )
                optimizer = None
        except BaseException:
            dispatcher.close()
            raise
        self._dispatcher = dispatcher
        self._accelerator = accelerator
        self._runner, self._prepared_optimizer = runner, optimizer
        try:
            yield
        except BaseException as e:
            dispatcher.on_exception(e)
            raise
        finally:
            dispatcher.close()
            self._dispatcher = self._accelerator = None
            self._runner = self._prepared_optimizer = None
            self._dataloaders = {}

    def _fit(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None,
        epochs: int,
    ) -> list[dict[str, Any]]:
        accelerator = self.accelerator
        optimizer = cast(AcceleratedOptimizer, self._prepared_optimizer)
        dataloader = self._prepare_dataloader(train_dataloader)
        num_batches = _num_batches(dataloader)
        if num_batches == 0:
            msg = "cannot fit on an empty dataloader"
            raise ValueError(msg)
        self.scheduler = self._build_scheduler(num_batches, epochs)
        # Accelerate keeps the accumulation steps process-wide, and an
        # accelerator of another run (e.g. one evaluating from a callback)
        # sets its own; read ours before any callback runs, and set them
        # again before each batch
        accumulation_steps = accelerator.gradient_accumulation_steps
        self.epochs = epochs
        # a stopped fit can end inside a gradient accumulation window,
        # where Accelerate skips zero_grad; the flag is process-wide state,
        # so start a new window without gradients
        accelerator.sync_gradients = True
        optimizer.zero_grad()
        # also the length at each epoch: a nested evaluate restores it
        self.num_batches = num_batches
        self._callback("on_train_begin")
        for epoch in range(epochs):
            self.epoch = epoch
            # the dataloaders Accelerate prepares reshuffle by epoch, in
            # sync across processes
            if hasattr(dataloader, "set_epoch"):
                dataloader.set_epoch(epoch)
            self._callback("on_epoch_begin")
            logs = self._train_epoch(dataloader, accumulation_steps)
            if eval_dataloader is not None:
                logs.update(self.evaluate(eval_dataloader, prefix="valid_"))
            self.history.append(logs)
            self._callback("on_epoch_end", logs)
            if self._sync_should_stop():
                break
        self._callback("on_train_end")
        return self.history

    def _train_epoch(
        self, dataloader: DataLoader, accumulation_steps: int
    ) -> dict[str, float]:
        accelerator = self.accelerator
        runner = cast(ModuleWrapper, self._runner)
        optimizer = cast(AcceleratedOptimizer, self._prepared_optimizer)
        max_grad_norm = self.trainer.max_grad_norm
        runner.train()
        # summed on the device, so that only the end of the epoch waits for
        # it
        means = RunningMean(accelerator)
        for batch in dataloader:
            self._callback("on_train_batch_begin", batch)
            accelerator.gradient_accumulation_steps = accumulation_steps
            with accelerator.accumulate(runner):
                loss, logs = split_loss(runner("training_step", batch))
                accelerator.backward(loss)
                # true at the end of accumulation
                if accelerator.sync_gradients:
                    if max_grad_norm is not None:
                        self.grad_norm = accelerator.clip_grad_norm_(
                            runner.parameters(), max_grad_norm
                        )
                    self._callback("on_before_optimizer_step")
                    optimizer.step()
                    optimizer.zero_grad()
            # fp16 skips steps whose gradients overflowed
            if accelerator.sync_gradients and not optimizer.step_was_skipped:
                self.global_step += 1
                if self.scheduler is not None:
                    self.scheduler.step()
            means.update(logs)
            self._callback("on_train_batch_end", batch, BatchLogs(logs))
            if self._sync_should_stop():
                break
        return means.compute()

    def _build_scheduler(
        self, num_batches: int | None, epochs: int
    ) -> LRScheduler | None:
        lr_scheduler = self.trainer.lr_scheduler
        if lr_scheduler is None:
            return None
        if num_batches is None:
            msg = (
                "an lr_scheduler needs a dataloader with a length to count "
                "the training steps"
            )
            raise ValueError(msg)
        num_steps = epochs * math.ceil(
            num_batches / self.accelerator.gradient_accumulation_steps
        )
        optimizer = cast(Optimizer, self.optimizer)
        if isinstance(lr_scheduler, str):
            return get_scheduler(
                lr_scheduler, optimizer, num_steps, warmup=self.trainer.warmup
            )
        return lr_scheduler(optimizer, num_steps)

    def _evaluate(
        self,
        dataloader: DataLoader,
        prefix: str,
        metrics: Mapping[str, Metric] | None,
    ) -> dict[str, Any]:
        runner = cast(ModuleWrapper, self._runner)
        dataloader = self._prepare_dataloader(dataloader)
        # update a copy, so that an evaluation started meanwhile (from a
        # callback, or by another trainer sharing the metrics) cannot reset
        # it
        metrics = copy.deepcopy(
            self.trainer.metrics
            if metrics is None
            else MetricCollection(metrics)
        )
        metrics.reset()
        input_names = metrics.input_names
        loss_sum, num_examples, num_batches = 0.0, 0, 0
        self.num_batches = _num_batches(dataloader)
        self._callback("on_test_begin")
        with _eval_mode(self.model):
            for batch in dataloader:
                self._callback("on_test_batch_begin", batch)
                outputs = runner("predict_step", batch)
                if not isinstance(outputs, Mapping):
                    msg = (
                        "predict_step must return a mapping of metric inputs "
                        f"to be evaluated, got {type(outputs).__name__}"
                    )
                    raise TypeError(msg)
                if outputs.get("loss") is not None:
                    # one loss per example, without the duplicates that pad
                    # the last batch on several processes
                    losses = self._gather(
                        example_losses(outputs, batch, self.accelerator.device)
                    )
                    loss_sum += float(losses.float().sum())
                    num_examples += losses.numel()
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
        results: dict[str, Any] = {}
        if num_examples:
            results["loss"] = loss_sum / num_examples
        results.update(metrics.compute())
        results = {f"{prefix}{k}": v for k, v in results.items()}
        self._callback("on_test_end", results)
        return results

    def _predict(self, dataloader: DataLoader) -> Any:
        runner = cast(ModuleWrapper, self._runner)
        dataloader = self._prepare_dataloader(dataloader)
        self.num_batches = _num_batches(dataloader)
        chunks = []
        self._callback("on_predict_begin")
        with _eval_mode(self.model):
            for batch in dataloader:
                self._callback("on_predict_batch_begin", batch)
                outputs = runner("predict_step", batch)
                self._callback("on_predict_batch_end", batch, outputs)
                if isinstance(outputs, Mapping):
                    outputs = {k: v for k, v in outputs.items() if k != "loss"}
                chunks.append(send_to_device(self._gather(outputs), "cpu"))
        self._callback("on_predict_end")
        if not chunks:
            msg = "cannot predict on an empty dataloader"
            raise ValueError(msg)
        return concat(chunks)

    # --- helpers -----------------------------------------------------------

    def _callback(self, hook: str, *args: Any) -> None:
        # only called during the run, which creates the dispatcher
        cast(Dispatcher, self._dispatcher)(hook, *args)

    def _sync_should_stop(self) -> bool:
        # a callback may stop training on one process only (e.g. on rank 0
        # metrics); stop all of them, or the others hang in the next
        # collective
        accelerator = self.accelerator
        if accelerator.num_processes > 1:
            if self.should_stop:
                accelerator.set_trigger()
            if accelerator.check_trigger():
                self.should_stop = True
        return self.should_stop

    def _prepare_dataloader(self, dataloader: DataLoader) -> DataLoader:
        if not isinstance(dataloader, DataLoader):
            msg = (
                f"expected a torch DataLoader, got {type(dataloader).__name__}"
                "; wrap ready-made batches with "
                "DataLoader(batches, batch_size=None)"
            )
            raise TypeError(msg)
        # prepare each dataloader once per run, so that persistent workers
        # are reused across epochs
        prepared = self._dataloaders.get(dataloader)
        if prepared is None:
            prepared = self.accelerator.prepare(dataloader)
            self._dataloaders[dataloader] = prepared
        return prepared

    def _gather(self, data: Any) -> Any:
        # collect the outputs of every process, without the duplicates
        # that pad the last batch
        if self.accelerator.num_processes == 1:
            return data
        if isinstance(data, Mapping):
            return {k: self._gather(v) for k, v in data.items()}
        return self.accelerator.gather_for_metrics(data)


@contextlib.contextmanager
def _eval_mode(model: Module) -> Generator[None]:
    """Put the model in evaluation mode, without gradients, and restore its
    mode afterwards."""
    training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        model.train(training)


def _num_batches(dataloader: DataLoader) -> int | None:
    """Length of a dataloader; None for an iterable dataset without one."""
    try:
        return len(dataloader)
    except TypeError:
        return None
