from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from tklearn.nn.trainer import Trainer

__all__ = [
    "Callback",
    "LambdaCallback",
]


class Callback:
    """Code that runs at fixed points of `Trainer.fit`, `evaluate` and
    `predict`.

    Override the hooks you need; every hook receives the trainer first,
    which exposes the run's state (``model``, ``optimizer``,
    ``accelerator``, ``epochs``, ``epoch``, ``global_step``,
    ``num_batches``, ``history``). A callback
    stops training by setting ``trainer.should_stop = True``; training then
    ends after the current batch, once the epoch has been evaluated and
    ``on_epoch_end`` has run.

    Hooks run on every process. Callbacks that write files or print should
    check ``trainer.accelerator.is_main_process``.

    `fit` runs::

        on_fit_begin
        for each epoch:
            on_epoch_begin
            for each batch:
                on_train_batch_begin
                on_before_optimizer_step   # when the optimizer steps
                on_train_batch_end
            on_evaluate_*                  # with an eval dataloader
            on_epoch_end
        on_fit_end

    and `evaluate` and `predict` run their own ``on_evaluate_*`` and
    ``on_predict_*`` hooks.
    """

    # --- fit ---------------------------------------------------------------

    def on_fit_begin(self, trainer: Trainer) -> None:
        """Called once before the first epoch."""

    def on_fit_end(self, trainer: Trainer) -> None:
        """Called once after the last epoch."""

    def on_epoch_begin(self, trainer: Trainer) -> None:
        """Called at the start of epoch ``trainer.epoch`` (zero-based)."""

    def on_epoch_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:
        """Called with the epoch's mean training losses and, when there is
        an eval dataloader, its ``valid_``-prefixed results."""

    def on_train_batch_begin(self, trainer: Trainer, batch: Any) -> None:
        """Called before each training batch."""

    def on_train_batch_end(
        self, trainer: Trainer, batch: Any, logs: dict[str, float]
    ) -> None:
        """Called after each training batch with its loss and logged
        terms."""

    def on_before_optimizer_step(self, trainer: Trainer) -> None:
        """Called before each optimizer step, after gradients are
        accumulated and clipped."""

    # --- evaluate ----------------------------------------------------------

    def on_evaluate_begin(self, trainer: Trainer) -> None:
        """Called before evaluation starts."""

    def on_evaluate_batch_begin(self, trainer: Trainer, batch: Any) -> None:
        """Called before each evaluation batch."""

    def on_evaluate_batch_end(
        self, trainer: Trainer, batch: Any, outputs: Any
    ) -> None:
        """Called after each evaluation batch with its `predict_step`
        outputs."""

    def on_evaluate_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:
        """Called with the evaluation results."""

    # --- predict -----------------------------------------------------------

    def on_predict_begin(self, trainer: Trainer) -> None:
        """Called before prediction starts."""

    def on_predict_batch_begin(self, trainer: Trainer, batch: Any) -> None:
        """Called before each prediction batch."""

    def on_predict_batch_end(
        self, trainer: Trainer, batch: Any, outputs: Any
    ) -> None:
        """Called after each prediction batch with its `predict_step`
        outputs."""

    def on_predict_end(self, trainer: Trainer) -> None:
        """Called after prediction ends."""


#: Names of the hooks a callback can implement.
HOOKS = tuple(name for name in vars(Callback) if name.startswith("on_"))


class LambdaCallback(Callback):
    """A callback made of functions passed as hooks.

    Parameters
    ----------
    **hooks : callable
        Functions keyed by hook name, taking the hook's arguments, e.g.
        ``on_epoch_end=lambda trainer, logs: ...``.

    Examples
    --------
    >>> log_steps = LambdaCallback(
    ...     on_train_batch_end=lambda trainer, batch, logs: print(
    ...         trainer.global_step, logs["loss"]
    ...     )
    ... )
    """

    def __init__(self, **hooks: Callable[..., None]) -> None:
        for name, hook in hooks.items():
            if name not in HOOKS:
                msg = f"{name!r} is not a hook of Callback"
                raise TypeError(msg)
            if not callable(hook):
                msg = f"the {name} hook must be callable, got {hook!r}"
                raise TypeError(msg)
            # the trainer calls getattr(callback, name)(trainer, ...)
            setattr(self, name, hook)
