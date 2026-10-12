"""How a trainer calls its callbacks during a run: which trainer each
callback belongs to, the order of each hook, and the logged and isolated
`on_exception`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tklearn.logging import get_logger
from tklearn.nn.callbacks import Callback

if TYPE_CHECKING:
    from tklearn.nn.trainer import Trainer

__all__ = [
    "Dispatcher",
]

# errors of the trainer's runs, so logged as the trainer's
logger = get_logger("tklearn.nn.trainer")


class Dispatcher:
    """Calls the hooks of a trainer's callbacks for one run.

    Each callback gets the trainer as `Callback.trainer` for the run: when
    the run starts, and a callback added during the run before its first
    hook. A callback keeps the state of its run, so it can be in one at a
    time; taking one that is in the run of another trainer raises a
    RuntimeError. `close` gives every callback back when the run ends.
    """

    def __init__(self, trainer: Trainer) -> None:
        self._trainer = trainer
        self._taken: list[Callback] = []
        try:
            for callback in trainer.callbacks:
                self._take(callback)
        except BaseException:
            self.close()
            raise

    def __call__(self, hook: str, *args: Any) -> None:
        """Call `hook` of every callback with `args`."""
        for callback in _hook_order(self._trainer.callbacks, hook):
            self._take(callback)
            getattr(callback, hook)(*args)

    def on_exception(self, exception: BaseException) -> None:
        """Call `on_exception` of every callback.

        Every callback gets the hook, so that each releases what it
        opened, and the run's exception is the one that goes on: one
        raised by a callback is logged.
        """
        for callback in _hook_order(self._trainer.callbacks, "on_exception"):
            if callback._trainer not in (None, self._trainer):
                continue  # in the run of another trainer
            try:
                self._take(callback)
                callback.on_exception(exception)
            except Exception:
                logger.exception(
                    f"{type(callback).__name__}.on_exception raised while "
                    f"handling {type(exception).__name__}"
                )

    def close(self) -> None:
        """Give back every callback taken for the run."""
        for callback in self._taken:
            callback._trainer = None
        self._taken = []

    def _take(self, callback: Callback) -> None:
        if callback._trainer is self._trainer:
            return
        if callback._trainer is not None:
            msg = (
                f"{type(callback).__name__} is in a run of another "
                "trainer; give each trainer its own callback"
            )
            raise RuntimeError(msg)
        callback._trainer = self._trainer
        self._taken.append(callback)


def _hook_order(callbacks: list[Callback], hook: str) -> tuple[Callback, ...]:
    """The callbacks in the order they get `hook`: wrappers first on
    ``*_begin`` hooks and last, in reverse, on ``*_end`` hooks and
    ``on_exception``.

    A copy, so that a callback changing `Trainer.callbacks` during the
    hook changes who gets the next hook, not this one.
    """
    wrappers = tuple(c for c in callbacks if c.wrapper)
    if wrappers and hook.endswith("_begin"):
        others = tuple(c for c in callbacks if not c.wrapper)
        return (*wrappers, *others)
    if wrappers and (hook.endswith("_end") or hook == "on_exception"):
        others = tuple(c for c in callbacks if not c.wrapper)
        return (*others, *reversed(wrappers))
    return tuple(callbacks)
