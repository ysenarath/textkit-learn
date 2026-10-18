from __future__ import annotations

from typing import Callable

from tklearn.nn.callbacks.base import Callback

__all__ = [
    "LambdaCallback",
]

#: Names of the hooks a callback can implement.
HOOKS = tuple(name for name in vars(Callback) if name.startswith("on_"))


class LambdaCallback(Callback):
    """A callback made of functions passed as hooks.

    Parameters
    ----------
    **hooks : callable
        Functions keyed by hook name, taking the hook's arguments, e.g.
        ``on_epoch_end=lambda logs: ...``.

    Examples
    --------
    >>> log_epochs = LambdaCallback(
    ...     on_epoch_end=lambda logs: print(logs["loss"])
    ... )
    >>> trainer.fit(train_loader, epochs=3, callbacks=[log_epochs])

    The functions do not get the `Callback.session`; subclass
    `Callback` to read it.
    """

    def __init__(self, **hooks: Callable[..., None]) -> None:
        for name, hook in hooks.items():
            if name not in HOOKS:
                msg = f"{name!r} is not a hook of Callback"
                raise TypeError(msg)
            if not callable(hook):
                msg = f"the {name} hook must be callable, got {hook!r}"
                raise TypeError(msg)
            # the trainer calls getattr(callback, name)(*args)
            setattr(self, name, hook)
