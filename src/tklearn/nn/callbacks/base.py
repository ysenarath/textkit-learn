from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from tklearn.nn.trainer import Trainer

__all__ = [
    "Callback",
]


class Callback:
    """Code that runs at fixed points of `Trainer.fit`, `evaluate` and
    `predict`.

    Override the hooks you need; every hook receives the trainer first,
    which exposes the run's state (``model``, ``optimizer``,
    ``accelerator``, ``epochs``, ``epoch``, ``global_step``,
    ``num_batches``, ``history``). A callback stops training by setting
    ``trainer.should_stop = True``; training then ends after the current
    batch, once the epoch has been evaluated and ``on_epoch_end`` has run.

    Hooks run on every process. Callbacks that write files or print should
    check ``trainer.accelerator.is_main_process``.

    The hooks are named as in Keras. `fit` runs::

        on_train_begin
        for each epoch:
            on_epoch_begin
            for each batch:
                on_train_batch_begin
                on_before_optimizer_step   # when the optimizer steps
                on_train_batch_end
            on_test_*                      # with an eval dataloader
            on_epoch_end
        on_train_end

    and `evaluate` and `predict` run their own ``on_test_*`` and
    ``on_predict_*`` hooks.

    When a run raises, including on ``KeyboardInterrupt``, the hooks of
    the unfinished run, such as ``on_epoch_end`` and ``on_train_end``,
    do not run; `on_exception` runs instead, then the exception goes on
    to the caller. Release there what the ``*_begin`` hooks opened.

    Callbacks get each hook in the order they were given, except
    wrappers (`wrapper`), which enclose the others.

    A callback keeps the state of the run it is in, so it can be in one
    at a time: a trainer raises when it starts a run while one of its
    callbacks is in the run of another trainer, e.g. one evaluating from
    a callback. One callback can serve several trainers in turn.

    Attributes
    ----------
    wrapper : bool
        Whether the callback wraps the others: its ``*_begin`` hooks run
        before theirs and its ``*_end`` hooks after theirs, so that what
        it opens, such as a span or a timer, encloses their work. Several
        wrappers nest in the order they were given: the first is the
        outermost. Other hooks, such as `on_before_optimizer_step`, run
        in the order given. False by default. A wrapper gets
        `on_exception` after the others too.
    """

    wrapper: ClassVar[bool] = False

    # --- fit ---------------------------------------------------------------

    def on_train_begin(self, trainer: Trainer) -> None:
        """Called once before the first epoch."""

    def on_train_end(self, trainer: Trainer) -> None:
        """Called once after the last epoch."""

    def on_epoch_begin(self, trainer: Trainer) -> None:
        """Called at the start of epoch ``trainer.epoch`` (zero-based)."""

    def on_epoch_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:
        """Called with the epoch's mean training losses and, when there is
        an eval dataloader, its ``valid_``-prefixed results."""

    def on_train_batch_begin(self, trainer: Trainer, batch: Any) -> None:
        """Called before each training batch."""

    def on_train_batch_end(
        self, trainer: Trainer, batch: Any, logs: Mapping[str, float]
    ) -> None:
        """Called after each training batch with its loss and logged
        terms.

        `logs` is a read-only mapping that converts each value to a float
        the first time it is read. On a GPU, that read waits for the
        batch's forward and backward passes to finish, so read the values
        only on the batches that need them.
        """

    def on_before_optimizer_step(self, trainer: Trainer) -> None:
        """Called before each optimizer step, after gradients are
        accumulated and clipped."""

    # --- evaluate ----------------------------------------------------------

    def on_test_begin(self, trainer: Trainer) -> None:
        """Called before evaluation starts."""

    def on_test_batch_begin(self, trainer: Trainer, batch: Any) -> None:
        """Called before each evaluation batch."""

    def on_test_batch_end(
        self, trainer: Trainer, batch: Any, outputs: Any
    ) -> None:
        """Called after each evaluation batch with its `predict_step`
        outputs."""

    def on_test_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:
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

    # --- errors ------------------------------------------------------------

    def on_exception(self, trainer: Trainer, exception: BaseException) -> None:
        """Called when `fit`, `evaluate` or `predict` raises, before the
        exception goes on to the caller.

        It runs once, for the outermost of these calls: an exception in
        the evaluation after an epoch of `fit` reaches it once, from
        `fit`. It also runs when the run raises before any ``*_begin``
        hook. ``trainer.accelerator`` is still available.

        On several processes, the others may not have raised, so it must
        not call collectives such as `Trainer.reduce_mean`: they would
        wait for processes that never call them.

        An exception raised here is logged, and does not keep the other
        callbacks from getting the hook or replace the run's exception.
        """
