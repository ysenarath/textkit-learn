from __future__ import annotations

import numbers
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from tqdm.auto import tqdm

from tklearn.nn.callbacks.base import Callback

if TYPE_CHECKING:
    from tklearn.nn.trainer import Trainer

__all__ = [
    "ProgbarLogger",
]


class ProgbarLogger(Callback):
    """Show progress bars with the running logs.

    `fit` shows a bar per epoch with the running mean of the training
    logs, replaced at the end of the epoch by the epoch's logs, including
    the evaluation results. `evaluate` and `predict` show a bar that is
    cleared when they finish. Only the main process shows bars.

    Parameters
    ----------
    **kwargs
        Arguments of every `tqdm` bar, e.g. ``ncols`` or ``file``.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        # open bars, innermost last; fit's epoch bar stays open while the
        # epoch is evaluated
        self._bars: list[tqdm] = []
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def on_train_begin(self, trainer: Trainer) -> None:
        # bars left open by a run that raised
        while self._bars:
            self._bars.pop().close()

    def on_epoch_begin(self, trainer: Trainer) -> None:
        self._totals, self._counts = {}, {}
        desc = f"Epoch {trainer.epoch + 1}/{trainer.epochs}"
        self._open(trainer, desc, leave=True)

    def on_train_batch_end(
        self, trainer: Trainer, batch: Any, logs: dict[str, float]
    ) -> None:
        for key, value in logs.items():
            self._totals[key] = self._totals.get(key, 0.0) + value
            self._counts[key] = self._counts.get(key, 0) + 1
        means = {k: v / self._counts[k] for k, v in self._totals.items()}
        bar = self._bars[-1]
        bar.set_postfix(_format(means), refresh=False)
        bar.update()

    def on_epoch_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:
        bar = self._bars.pop()
        bar.set_postfix(_format(logs), refresh=False)
        bar.close()

    def on_test_begin(self, trainer: Trainer) -> None:
        self._open(trainer, "Evaluating", leave=False)

    def on_test_batch_end(
        self, trainer: Trainer, batch: Any, outputs: Any
    ) -> None:
        self._bars[-1].update()

    def on_test_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:
        self._bars.pop().close()

    def on_predict_begin(self, trainer: Trainer) -> None:
        self._open(trainer, "Predicting", leave=False)

    def on_predict_batch_end(
        self, trainer: Trainer, batch: Any, outputs: Any
    ) -> None:
        self._bars[-1].update()

    def on_predict_end(self, trainer: Trainer) -> None:
        self._bars.pop().close()

    def _open(self, trainer: Trainer, desc: str, leave: bool) -> None:
        kwargs = {
            "desc": desc,
            "total": trainer.num_batches,
            "unit": "batch",
            "leave": leave,
            "dynamic_ncols": True,
            "disable": not trainer.accelerator.is_main_process,
            **self.kwargs,
        }
        self._bars.append(tqdm(**kwargs))


def _format(logs: Mapping[str, Any]) -> dict[str, str]:
    """The numeric logs, formatted for a progress bar."""
    return {
        key: f"{value:.4g}"
        for key, value in logs.items()
        if isinstance(value, numbers.Real) and not isinstance(value, bool)
    }
