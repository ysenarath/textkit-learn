from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Callable, Union

import torch
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from tklearn.metrics import Metric, MetricCollection
from tklearn.nn.callbacks import Callback
from tklearn.nn.module import Module
from tklearn.nn.schedules import get_scheduler
from tklearn.nn.session import Session

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

    A trainer holds configuration only, fixed when it is created: its
    attributes are read-only, so for other settings create another
    trainer, e.g. with the same model. Each `fit`, `evaluate` or
    `predict` call opens a new `Session`, which it closes when it
    returns, and each `fit` builds a new optimizer and scheduler with
    `create_optimizer` and `create_scheduler`; a call writes nothing back
    to the trainer or to what it was given, except the model, which it
    trains.

    Parameters
    ----------
    model : Module
        The model; see `Module` for the hooks it implements.
    optimizer : callable, optional
        Builds the optimizer from the model's parameters at the start of
        each `fit`, e.g. ``functools.partial(torch.optim.AdamW, lr=2e-5)``.
        Required by `fit`, unless a subclass overrides `create_optimizer`.
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
        and ``*_end`` hooks after the others. Each session gets them as
        a list, `Session.callbacks`.
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

    Each call creates an `Accelerator` from these with
    `create_accelerator`; anything else (plugins, distributed settings)
    comes from ``accelerate config`` or ``accelerate launch``, or from a
    subclass that overrides `create_accelerator`.

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
        self._lr_scheduler = lr_scheduler
        self._warmup = warmup
        self._metrics = MetricCollection(metrics)
        self._callbacks = tuple(callbacks)
        self._max_grad_norm = max_grad_norm
        self._mixed_precision = mixed_precision
        self._gradient_accumulation_steps = gradient_accumulation_steps
        self._cpu = cpu

    @property
    def model(self) -> Module:
        """The model."""
        return self._model

    @property
    def metrics(self) -> MetricCollection:
        """Metrics computed by `evaluate`.

        Each evaluation updates a copy, so these keep their own state.
        """
        return self._metrics

    @property
    def callbacks(self) -> tuple[Callback, ...]:
        """The callbacks, of which each session gets a copy."""
        return self._callbacks

    @property
    def max_grad_norm(self) -> float | None:
        """The gradient norm to clip to before each optimizer step."""
        return self._max_grad_norm

    @property
    def mixed_precision(self) -> str | None:
        """The mixed precision mode."""
        return self._mixed_precision

    @property
    def gradient_accumulation_steps(self) -> int:
        """Batches whose gradients are accumulated before each step."""
        return self._gradient_accumulation_steps

    @property
    def cpu(self) -> bool:
        """Whether to run on the CPU."""
        return self._cpu

    def create_accelerator(self) -> Accelerator:
        """Create the `Accelerator` of a session.

        Each `fit`, `evaluate` or `predict` calls it when it starts.
        Override it to give the accelerator more arguments, e.g. plugins
        or kwargs handlers.

        Accelerate fixes ``cpu`` and ``mixed_precision`` for the process
        at the first accelerator, and raises when a later one asks
        otherwise, so calling it outside a session fixes them for every
        later session too.
        """
        return Accelerator(
            cpu=self.cpu,
            mixed_precision=self.mixed_precision,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )

    def create_optimizer(self) -> Optimizer:
        """Create the optimizer of a `fit`, from the `optimizer` factory.

        Each `fit` calls it when it starts. Override it to build the
        optimizer otherwise, e.g. with parameter groups::

            def create_optimizer(self):
                params = dict(self.model.named_parameters())
                decay = [p for n, p in params.items() if "bias" not in n]
                rest = [p for n, p in params.items() if "bias" in n]
                return self._optimizer([
                    {"params": decay},
                    {"params": rest, "weight_decay": 0.0},
                ])
        """
        if self._optimizer is None:
            msg = "fit needs an optimizer; pass one to the Trainer"
            raise ValueError(msg)
        return self._optimizer(self.model.parameters())

    def create_scheduler(
        self, optimizer: Optimizer, num_training_steps: int | None
    ) -> LRScheduler | None:
        """Create the learning rate scheduler of a `fit`, from
        `lr_scheduler` and `warmup`; None without one.

        Each `fit` calls it after `create_optimizer`, with the optimizer
        it created. Override it to build the scheduler otherwise.

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer of the `fit`.
        num_training_steps : int or None
            Optimizer steps of the `fit`; None when the training
            dataloader has no length.
        """
        lr_scheduler = self._lr_scheduler
        if lr_scheduler is None:
            return None
        if num_training_steps is None:
            msg = (
                "an lr_scheduler needs a dataloader with a length to count "
                "the training steps"
            )
            raise ValueError(msg)
        if isinstance(lr_scheduler, str):
            return get_scheduler(
                lr_scheduler,
                optimizer,
                num_training_steps,
                warmup=self._warmup,
            )
        return lr_scheduler(optimizer, num_training_steps)

    def fit(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None = None,
        *,
        epochs: int = 1,
    ) -> list[dict[str, Any]]:
        """Train the model.

        Each call is a new session, with a new optimizer and scheduler from
        `create_optimizer` and `create_scheduler`; the model keeps what it
        learned.

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
        if epochs < 1:
            msg = f"epochs must be at least 1, got {epochs}"
            raise ValueError(msg)
        session = Session(self)
        with session._open(training=True):
            return session._fit(train_dataloader, eval_dataloader, epochs)

    def evaluate(
        self,
        dataloader: DataLoader,
        *,
        prefix: str = "",
        metrics: Mapping[str, Metric] | None = None,
    ) -> dict[str, Any]:
        """Compute the mean loss and the metrics over a dataloader.

        A callback evaluating during a session uses `Session.evaluate`
        instead, which reuses the session.

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
        session = Session(self)
        with session._open(training=False):
            return session.evaluate(dataloader, prefix=prefix, metrics=metrics)

    def predict(self, dataloader: DataLoader) -> Any:
        """Run `predict_step` over a dataloader and concatenate the outputs.

        Tensors and arrays are concatenated along the first dimension and
        lists are joined; tuples and mappings are concatenated item by item,
        leaving out the ``"loss"`` of a mapping. The result is on the CPU.

        Tensors and arrays whose other dimensions differ between batches
        (or processes), such as per-token outputs padded to each batch's
        longest sequence, are padded at the end to the largest: signed
        integers with -100, the label index losses and metrics ignore,
        and the rest, e.g. floats and bools, with 0 (False). Return masks
        as bools, so that padding reads as masked out.

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
        session = Session(self)
        with session._open(training=False):
            return session.predict(dataloader)
