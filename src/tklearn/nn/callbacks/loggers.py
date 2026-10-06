from __future__ import annotations

import csv
import numbers
import os
from collections.abc import Mapping
from typing import IO, TYPE_CHECKING, Any

from tqdm.auto import tqdm

from tklearn.nn.callbacks.base import Callback

if TYPE_CHECKING:
    from tklearn.nn.trainer import Trainer

__all__ = [
    "CSVLogger",
    "ProgbarLogger",
]


class CSVLogger(Callback):
    """Write the logs of each epoch to a CSV file.

    Each row holds the (zero-based) ``epoch`` followed by the logs. The
    columns are the keys of the first epoch's logs; a key missing from a
    later epoch is written as ``NA``. Arrays, such as a confusion matrix,
    are written as lists.

    Parameters
    ----------
    filename : str or path-like
        The CSV file. Only the main process writes it.
    separator : str, default=","
        Column separator.
    append : bool, default=False
        Append to an existing file instead of overwriting it, without
        repeating its header.
    """

    def __init__(
        self,
        filename: str | os.PathLike[str],
        separator: str = ",",
        append: bool = False,
    ) -> None:
        self.filename = os.fspath(filename)
        self.separator = separator
        self.append = append
        self._file: IO[str] | None = None
        self._writer: csv.DictWriter | None = None
        self._write_header = True

    def on_fit_begin(self, trainer: Trainer) -> None:
        self._close()
        if not trainer.accelerator.is_main_process:
            return
        self._write_header = not (
            self.append
            and os.path.exists(self.filename)
            and os.path.getsize(self.filename) > 0
        )
        mode = "a" if self.append else "w"
        self._file = open(self.filename, mode, newline="", encoding="utf-8")

    def on_epoch_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:
        if self._file is None:
            return
        if self._writer is None:
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=["epoch", *logs],
                delimiter=self.separator,
                restval="NA",
                extrasaction="ignore",
            )
            if self._write_header:
                self._writer.writeheader()
        row = {k: _to_builtin(v) for k, v in logs.items()}
        self._writer.writerow({**row, "epoch": trainer.epoch})
        self._file.flush()

    def on_fit_end(self, trainer: Trainer) -> None:
        self._close()

    def _close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._writer = None


def _to_builtin(value: Any) -> Any:
    # numpy scalars and arrays as Python numbers and nested lists
    return value.tolist() if hasattr(value, "tolist") else value


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

    def on_fit_begin(self, trainer: Trainer) -> None:
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

    def on_evaluate_begin(self, trainer: Trainer) -> None:
        self._open(trainer, "Evaluating", leave=False)

    def on_evaluate_batch_end(
        self, trainer: Trainer, batch: Any, outputs: Any
    ) -> None:
        self._bars[-1].update()

    def on_evaluate_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:
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
