from __future__ import annotations

import csv
import os
from typing import IO, TYPE_CHECKING, Any

from tklearn.nn.callbacks.base import Callback

if TYPE_CHECKING:
    from tklearn.nn.trainer import Trainer

__all__ = [
    "CSVLogger",
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

    def on_train_begin(self, trainer: Trainer) -> None:
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

    def on_train_end(self, trainer: Trainer) -> None:
        self._close()

    def _close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._writer = None


def _to_builtin(value: Any) -> Any:
    # numpy scalars and arrays as Python numbers and nested lists
    return value.tolist() if hasattr(value, "tolist") else value
