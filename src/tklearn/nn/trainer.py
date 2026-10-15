from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Callable, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from tklearn.metrics import Metric, MetricCollection
from tklearn.nn.callbacks import Callback
from tklearn.nn.module import Module
from tklearn.nn.state import TrainerState

__all__ = [
    "OptimizerFactory",
    "SchedulerFactory",
    "Trainer",
]

#: Builds an optimizer from the model's parameters, e.g.
#: ``functools.partial(torch.optim.AdamW, lr=2e-5)``.
OptimizerFactory = Callable[[Iterator[torch.nn.Parameter]], Optimizer]

#: The name of a `get_scheduler` schedule, or
#: ``f(optimizer, num_training_steps)`` returning a scheduler.
SchedulerFactory = Union[str, Callable[[Optimizer, int], LRScheduler]]


class Trainer:
    """Fit, evaluate and run a `Module` with Hugging Face Accelerate.

    The same code runs on CPU, a GPU or Apple MPS (picked in that order of
    preference by Accelerate), with mixed precision, and on several GPUs
    when started with ``accelerate launch``.

    A trainer holds configuration only. Each `fit`, `evaluate` or
    `predict` call runs with a new `TrainerState`, which it drops when it
    returns, and builds a new optimizer and scheduler from the factories;
    a call writes nothing back to the trainer or to what it was given,
    except the model, which it trains.

    Parameters
    ----------
    model : Module
        The model; see `Module` for the hooks it implements.
    optimizer : callable, optional
        Builds the optimizer from the model's parameters at the start of
        each `fit`, e.g. ``functools.partial(torch.optim.AdamW, lr=2e-5)``.
        Required by `fit`.
    lr_scheduler : str or callable, optional
        Stepped after every optimizer step. Either the name of a
        `get_scheduler` schedule (e.g. ``"linear"``), or
        ``f(optimizer, num_training_steps)`` returning a scheduler; built
        at the start of each `fit`.
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
        and ``*_end`` hooks after the others. Each run gets a copy of the
        list, as `TrainerState.callbacks`.
    max_grad_norm : float, optional
        Clip the gradient norm to this value before each optimizer step.
    mixed_precision : {"no", "fp16", "bf16"}, optional
        Mixed precision mode. Defaults to Accelerate's configuration
        (``"no"`` unless set by ``accelerate config`` or ``accelerate
        launch``).
    gradient_accumulation_steps : int, default=1
        Batches whose gradients are accumulated before each optimizer step;
        the losses are averaged over them.
    cpu : bool, default=False
        Run on the CPU even when a GPU or MPS device is available.

    Each call creates an `Accelerator` from these; anything else (plugins,
    distributed settings) comes from ``accelerate config`` or
    ``accelerate launch``.

    Examples
    --------
    >>> trainer = Trainer(
    ...     model,
    ...     functools.partial(torch.optim.AdamW, lr=2e-5),
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
        optimizer: OptimizerFactory | None = None,
        *,
        lr_scheduler: SchedulerFactory | None = None,
        warmup: int | float = 0,
        metrics: Mapping[str, Metric] | None = None,
        callbacks: Iterable[Callback] | None = None,
        max_grad_norm: float | None = None,
        mixed_precision: str | None = None,
        gradient_accumulation_steps: int = 1,
        cpu: bool = False,
    ) -> None:
        if not isinstance(model, Module):
            msg = f"expected a tklearn.nn.Module, got {type(model).__name__}"
            raise TypeError(msg)
        if isinstance(optimizer, Optimizer) or not (
            optimizer is None or callable(optimizer)
        ):
            msg = (
                "optimizer must build the optimizer from the model's "
                "parameters, e.g. functools.partial(torch.optim.AdamW, "
                f"lr=2e-5), got {type(optimizer).__name__}"
            )
            raise TypeError(msg)
        if isinstance(lr_scheduler, LRScheduler) or not (
            lr_scheduler is None
            or isinstance(lr_scheduler, str)
            or callable(lr_scheduler)
        ):
            msg = (
                "lr_scheduler must be the name of a schedule or build the "
                "scheduler as f(optimizer, num_training_steps), got "
                f"{type(lr_scheduler).__name__}"
            )
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
        if (
            not isinstance(gradient_accumulation_steps, int)
            or gradient_accumulation_steps < 1
        ):
            msg = (
                "gradient_accumulation_steps must be an int of at least 1, "
                f"got {gradient_accumulation_steps!r}"
            )
            raise ValueError(msg)
        self._model = model
        self._optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.warmup = warmup
        self.metrics = metrics
        self.callbacks = callbacks
        self.max_grad_norm = max_grad_norm
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.cpu = cpu

    @property
    def model(self) -> Module:
        """The model, fixed when the trainer is created."""
        return self._model

    @property
    def optimizer(self) -> OptimizerFactory | None:
        """The optimizer factory, fixed when the trainer is created."""
        return self._optimizer

    @property
    def metrics(self) -> MetricCollection:
        """Metrics computed by `evaluate`.

        Each evaluation updates a copy, so these keep their own state.
        """
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: Mapping[str, Metric] | None) -> None:
        self._metrics = MetricCollection(metrics)

    @property
    def accelerator_args(self) -> dict[str, Any]:
        """The arguments of the `Accelerator` of each run."""
        return {
            "cpu": self.cpu,
            "mixed_precision": self.mixed_precision,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }

    def fit(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None = None,
        *,
        epochs: int = 1,
    ) -> list[dict[str, Any]]:
        """Train the model.

        Each call is a new run, with a new optimizer and scheduler built
        from the factories; the model keeps what it learned.

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
            The logs of each epoch: the mean training ``loss`` and other
            logged terms, then the evaluation results.
        """
        if self.optimizer is None:
            msg = "fit needs an optimizer; pass one to the Trainer"
            raise ValueError(msg)
        if epochs < 1:
            msg = f"epochs must be at least 1, got {epochs}"
            raise ValueError(msg)
        state = TrainerState(self)
        with state._run(training=True):
            return state._fit(train_dataloader, eval_dataloader, epochs)

    def evaluate(
        self,
        dataloader: DataLoader,
        *,
        prefix: str = "",
        metrics: Mapping[str, Metric] | None = None,
    ) -> dict[str, Any]:
        """Compute the mean loss and the metrics over a dataloader.

        A callback evaluating during a run uses `TrainerState.evaluate`
        instead, which reuses the run.

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
            ``loss``, the mean loss over the examples (when `predict_step`
            returns a loss), followed by one entry per metric. A loss per
            example is averaged as it is; a scalar one, the mean over the
            batch, counts once per example of its batch. On several
            processes, the duplicates that pad the last batch are left
            out, which is exact with a loss per example.
        """
        state = TrainerState(self)
        with state._run(training=False):
            return state.evaluate(dataloader, prefix=prefix, metrics=metrics)

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
        state = TrainerState(self)
        with state._run(training=False):
            return state.predict(dataloader)
