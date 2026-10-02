from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

from torch.optim.optimizer import Optimizer

__all__ = [
    "Callback",
    "CallbackList",
    "CallbacksMixin",
]

if TYPE_CHECKING:
    from tklearn.nn.base.module import Module
    from tklearn.nn.base.trainer import Trainer


class Callback:
    """Base class for hooks into `Trainer`, `Evaluator` and `Predictor`.

    Override any of the ``on_*`` methods. Before a run starts, the runner
    calls `set_model` and `set_params`; `Trainer` also calls `set_trainer`,
    which lets a callback stop training with
    ``self.trainer.stop_training = True``.

    Which hooks fire:

    - `Trainer.fit`: ``on_train_begin``/``on_train_end``,
      ``on_epoch_begin``/``on_epoch_end``,
      ``on_train_batch_begin``/``on_train_batch_end`` and the gradient hooks
      ``on_before_zero_grad``, ``on_before_backward``, ``on_after_backward``,
      ``on_before_optimizer_step``.
    - `Evaluator.evaluate`: ``on_test_begin``/``on_test_end`` and
      ``on_test_batch_begin``/``on_test_batch_end``.
    - `Predictor.predict` and `Encoder.encode`: ``on_predict_begin``/
      ``on_predict_end`` and ``on_predict_batch_begin``/
      ``on_predict_batch_end``.

    `logs` is always a dict. ``on_epoch_end`` receives the epoch's mean
    training losses plus any evaluation results; ``on_train_end`` receives the
    logs of the last epoch.
    """

    def __init__(self) -> None:
        self._model: Module | None = None
        self._trainer: Trainer | None = None
        self._params: dict[str, Any] = {}

    @property
    def model(self) -> Module | None:
        """The model being trained, evaluated or run."""
        return self._model

    def set_model(self, model: Module | None) -> None:
        self._model = model

    @property
    def trainer(self) -> Trainer | None:
        """The running `Trainer`, or None outside of `Trainer.fit`."""
        return self._trainer

    def set_trainer(self, trainer: Trainer | None) -> None:
        self._trainer = trainer

    @property
    def params(self) -> dict[str, Any]:
        """Run parameters set by the runner.

        `Trainer` sets ``epochs``, ``steps`` (batches per epoch) and
        ``batch_size``; `Evaluator` sets ``test_steps``; `Predictor` sets
        ``pred_steps``.
        """
        return self._params

    def set_params(self, params: dict[str, Any] | None) -> None:
        self._params = dict(params or {})

    # --- training --------------------------------------------------------

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        """Called once before the first epoch."""

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        """Called once after the last epoch, with that epoch's logs."""

    def on_epoch_begin(
        self, epoch: int, logs: dict[str, Any] | None = None
    ) -> None:
        """Called at the start of each epoch (zero-based)."""

    def on_epoch_end(
        self, epoch: int, logs: dict[str, Any] | None = None
    ) -> None:
        """Called at the end of each epoch with training and eval results."""

    def on_train_batch_begin(
        self, batch_idx: int, logs: dict[str, Any] | None = None
    ) -> None:
        """Called before each training batch."""

    def on_train_batch_end(
        self, batch_idx: int, logs: dict[str, Any] | None = None
    ) -> None:
        """Called after each training batch, with its losses."""

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        """Called before ``optimizer.zero_grad()``."""

    def on_before_backward(self, logs: dict[str, Any] | None = None) -> None:
        """Called before ``loss.backward()``."""

    def on_after_backward(self, logs: dict[str, Any] | None = None) -> None:
        """Called after gradients are computed (and clipped)."""

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        """Called before ``optimizer.step()``."""

    # --- evaluation ------------------------------------------------------

    def on_test_begin(self, logs: dict[str, Any] | None = None) -> None:
        """Called before evaluation starts."""

    def on_test_end(self, logs: dict[str, Any] | None = None) -> None:
        """Called after evaluation, with the evaluation results."""

    def on_test_batch_begin(
        self, batch_idx: int, logs: dict[str, Any] | None = None
    ) -> None:
        """Called before each evaluation batch."""

    def on_test_batch_end(
        self, batch_idx: int, logs: dict[str, Any] | None = None
    ) -> None:
        """Called after each evaluation batch."""

    # --- prediction ------------------------------------------------------

    def on_predict_begin(self, logs: dict[str, Any] | None = None) -> None:
        """Called before prediction starts."""

    def on_predict_end(self, logs: dict[str, Any] | None = None) -> None:
        """Called after prediction ends."""

    def on_predict_batch_begin(
        self, batch_idx: int, logs: dict[str, Any] | None = None
    ) -> None:
        """Called before each prediction batch."""

    def on_predict_batch_end(
        self, batch_idx: int, logs: dict[str, Any] | None = None
    ) -> None:
        """Called after each prediction batch."""


_HOOKS = tuple(
    name
    for name in vars(Callback)
    if name.startswith("on_") or name.startswith("set_")
)


def _dispatch(name: str):
    def method(self: CallbackList, *args: Any, **kwargs: Any) -> None:
        getattr(Callback, name)(self, *args, **kwargs)
        for callback in self._callbacks:
            getattr(callback, name)(*args, **kwargs)

    method.__name__ = name
    method.__doc__ = f"Call `{name}` on every callback in the list."
    return method


class CallbackList(Callback, Sequence[Callback]):
    """An ordered group of callbacks that is itself a callback.

    Calling a hook on the list calls it on each callback in order. Exceptions
    raised by a callback propagate to the caller.
    """

    def __init__(self, callbacks: Iterable[Callback] | None = None) -> None:
        super().__init__()
        self._callbacks: list[Callback] = []
        for callback in callbacks or []:
            self.append(callback)

    def append(self, callback: Callback) -> None:
        if not isinstance(callback, Callback):
            msg = (
                f"expected a {Callback.__name__}, got "
                f"{type(callback).__name__}"
            )
            raise TypeError(msg)
        self._callbacks.append(callback)

    def remove(self, callback: Callback | type[Callback]) -> None:
        """Remove a callback, or every callback of the given type."""
        if isinstance(callback, type):
            self._callbacks = [
                cb for cb in self._callbacks if not isinstance(cb, callback)
            ]
        else:
            self._callbacks.remove(callback)

    def get(
        self,
        callback: type[Callback],
        default: Callback | None = None,
    ) -> Callback | None:
        """Return the first callback of the given type, or `default`."""
        for cb in self._callbacks:
            if isinstance(cb, callback):
                return cb
        return default

    def __getitem__(self, index: int) -> Callback:
        return self._callbacks[index]

    def __len__(self) -> int:
        return len(self._callbacks)

    def __iter__(self) -> Iterator[Callback]:
        return iter(self._callbacks)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, type):
            return any(isinstance(cb, item) for cb in self._callbacks)
        return item in self._callbacks

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._callbacks!r})"


for _name in _HOOKS:
    setattr(CallbackList, _name, _dispatch(_name))
del _name


class CallbacksMixin:
    """Gives a runner a `callbacks` attribute that is always a CallbackList."""

    @property
    def callbacks(self) -> CallbackList:
        return self._callbacks

    @callbacks.setter
    def callbacks(
        self, value: CallbackList | Iterable[Callback] | Callback | None
    ) -> None:
        if value is None:
            value = []
        elif isinstance(value, Callback) and not isinstance(
            value, CallbackList
        ):
            value = [value]
        self._callbacks = CallbackList(value)
