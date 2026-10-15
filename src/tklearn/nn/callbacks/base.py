from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

from tklearn.logging import get_logger

if TYPE_CHECKING:
    from tklearn.nn.session import Session

__all__ = [
    "Callback",
    "Dispatcher",
]

# errors of the trainer's sessions, so logged as the trainer's
logger = get_logger("tklearn.nn.trainer")


class Callback:
    """Code that runs at fixed points of `Trainer.fit`, `evaluate` and
    `predict`.

    Override the hooks you need. In a hook, `session` is the `Session`
    the callback is in (``model``, ``callbacks``, ``optimizer``,
    ``scheduler``, ``accelerator``, ``history``, ``epochs``, ``epoch``,
    ``global_step``, ``num_batches``, ``grad_norm``), and
    ``session.trainer`` its configuration. A callback stops training by
    setting ``self.session.should_stop = True``; training then ends after the
    current batch, once the epoch has been evaluated and ``on_epoch_end``
    has run.

    Hooks run on every process. Callbacks that write files or print should
    check ``self.session.accelerator.is_main_process``.

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

    When a session raises, including on ``KeyboardInterrupt``, the hooks
    of the unfinished session, such as ``on_epoch_end`` and ``on_train_end``,
    do not run; `on_exception` runs instead, then the exception goes on
    to the caller. Release there what the ``*_begin`` hooks opened.

    Callbacks get each hook in the order they were given, except
    wrappers (`wrapper`), which enclose the others.

    A callback keeps the session it is in, so it can be in one at a
    time: a session raises when one of its callbacks is in another
    session, e.g. of a trainer evaluating from a callback. A callback
    evaluates within its own session with ``self.session.evaluate``. One
    callback can serve several sessions in turn.

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

    # the session the callback is in, set for the session (see
    # Dispatcher)
    _session: Session | None = None

    @property
    def session(self) -> Session:
        """The session the callback is in.

        Set when a `fit`, `evaluate` or `predict` starts, and unset when
        it ends; only available in hooks.
        """
        if self._session is None:
            msg = (
                f"{type(self).__name__} has a session only during a fit, "
                "evaluate or predict"
            )
            raise RuntimeError(msg)
        return self._session

    # --- fit ---------------------------------------------------------------

    def on_train_begin(self) -> None:
        """Called once before the first epoch."""

    def on_train_end(self) -> None:
        """Called once after the last epoch."""

    def on_epoch_begin(self) -> None:
        """Called at the start of epoch ``self.session.epoch`` (zero-based)."""

    def on_epoch_end(self, logs: dict[str, Any]) -> None:
        """Called with the epoch's mean training losses and, when there is
        an eval dataloader, its ``valid_``-prefixed results."""

    def on_train_batch_begin(self, batch: Any) -> None:
        """Called before each training batch."""

    def on_train_batch_end(
        self, batch: Any, logs: Mapping[str, float]
    ) -> None:
        """Called after each training batch with its loss and logged
        terms.

        `logs` is a read-only mapping that converts each value to a float
        the first time it is read. On a GPU, that read waits for the
        batch's forward and backward passes to finish, so read the values
        only on the batches that need them.
        """

    def on_before_optimizer_step(self) -> None:
        """Called before each optimizer step, after gradients are
        accumulated and clipped."""

    # --- evaluate ----------------------------------------------------------

    def on_test_begin(self) -> None:
        """Called before evaluation starts."""

    def on_test_batch_begin(self, batch: Any) -> None:
        """Called before each evaluation batch."""

    def on_test_batch_end(self, batch: Any, outputs: Any) -> None:
        """Called after each evaluation batch with its `predict_step`
        outputs."""

    def on_test_end(self, logs: dict[str, Any]) -> None:
        """Called with the evaluation results."""

    # --- predict -----------------------------------------------------------

    def on_predict_begin(self) -> None:
        """Called before prediction starts."""

    def on_predict_batch_begin(self, batch: Any) -> None:
        """Called before each prediction batch."""

    def on_predict_batch_end(self, batch: Any, outputs: Any) -> None:
        """Called after each prediction batch with its `predict_step`
        outputs."""

    def on_predict_end(self) -> None:
        """Called after prediction ends."""

    # --- errors ------------------------------------------------------------

    def on_exception(self, exception: BaseException) -> None:
        """Called when `fit`, `evaluate` or `predict` raises, before the
        exception goes on to the caller.

        It runs once, for the outermost of these calls: an exception in
        the evaluation after an epoch of `fit` reaches it once, from
        `fit`. It also runs when the session raises before any ``*_begin``
        hook. ``self.session.accelerator`` is still available.

        On several processes, the others may not have raised, so it must
        not call collectives such as `RunningMean.compute`: they would
        wait for processes that never call them.

        An exception raised here is logged, and does not keep the other
        callbacks from getting the hook or replace the session's
        exception.
        """


class Dispatcher:
    """Calls the hooks of a session's callbacks.

    `Trainer` creates one for each `fit`, `evaluate` or `predict`; a
    custom training loop can use one the same way.

    Each callback gets the session as `Callback.session`: when the
    session starts, and a callback added during the session before its
    first hook. A callback keeps its session, so it can be in one at a
    time; taking one that is in another session raises a RuntimeError.
    `close` gives every callback back when the session ends, and must be
    called even when the session raises, or the callbacks stay in it and
    every later session of them raises.

    Parameters
    ----------
    session : Session
        The session, whose `callbacks` get the hooks.

    Examples
    --------
    >>> dispatcher = Dispatcher(session)
    >>> try:
    ...     dispatcher("on_train_begin")
    ...     ...
    ...     dispatcher("on_train_end")
    ... except BaseException as e:
    ...     dispatcher.on_exception(e)
    ...     raise
    ... finally:
    ...     dispatcher.close()
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._taken: list[Callback] = []
        try:
            for callback in session.callbacks:
                self._take(callback)
        except BaseException:
            self.close()
            raise

    def __call__(self, hook: str, *args: Any) -> None:
        """Call `hook` of every callback with `args`."""
        for callback in _hook_order(self._session.callbacks, hook):
            self._take(callback)
            getattr(callback, hook)(*args)

    def on_exception(self, exception: BaseException) -> None:
        """Call `on_exception` of every callback.

        Every callback gets the hook, so that each releases what it
        opened, and the session's exception is the one that goes on: one
        raised by a callback is logged.
        """
        for callback in _hook_order(self._session.callbacks, "on_exception"):
            if callback._session not in (None, self._session):
                continue  # in another session
            try:
                self._take(callback)
                callback.on_exception(exception)
            except Exception:
                logger.exception(
                    f"{type(callback).__name__}.on_exception raised while "
                    f"handling {type(exception).__name__}"
                )

    def close(self) -> None:
        """Give back every callback taken for the session."""
        for callback in self._taken:
            callback._session = None
        self._taken = []

    def _take(self, callback: Callback) -> None:
        if callback._session is self._session:
            return
        if callback._session is not None:
            msg = (
                f"{type(callback).__name__} is in another session; give each "
                "trainer its own callback, and evaluate from a callback "
                "with self.session.evaluate"
            )
            raise RuntimeError(msg)
        callback._session = self._session
        self._taken.append(callback)


def _hook_order(callbacks: list[Callback], hook: str) -> tuple[Callback, ...]:
    """The callbacks in the order they get `hook`: wrappers first on
    ``*_begin`` hooks and last, in reverse, on ``*_end`` hooks and
    ``on_exception``.

    A copy, so that a callback changing `Session.callbacks` during
    the hook changes who gets the next hook, not this one.
    """
    wrappers = tuple(c for c in callbacks if c.wrapper)
    if wrappers and hook.endswith("_begin"):
        others = tuple(c for c in callbacks if not c.wrapper)
        return (*wrappers, *others)
    if wrappers and (hook.endswith("_end") or hook == "on_exception"):
        others = tuple(c for c in callbacks if not c.wrapper)
        return (*others, *reversed(wrappers))
    return tuple(callbacks)
