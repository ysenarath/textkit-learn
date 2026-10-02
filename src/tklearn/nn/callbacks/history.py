from __future__ import annotations

from typing import Any

import pandas as pd

from tklearn.nn.callbacks.base import Callback

__all__ = [
    "History",
]


class History(Callback):
    """Records the logs of every epoch. Returned by `Trainer.fit`.

    Attributes
    ----------
    epoch : list of int
        The epochs that completed.
    history : dict of str to list
        One list of values per log key, aligned with `epoch`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.epoch: list[int] = []
        self.history: dict[str, list[Any]] = {}

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        self.epoch = []
        self.history = {}

    def on_epoch_end(
        self, epoch: int, logs: dict[str, Any] | None = None
    ) -> None:
        self.epoch.append(epoch)
        for key, value in (logs or {}).items():
            self.history.setdefault(key, []).append(value)

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self.history, index=self.epoch)
