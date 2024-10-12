from __future__ import annotations

from typing import Dict, List

import pandas as pd

from tklearn.nn.callbacks.base import Callback

__all__ = [
    "History",
]


class History(Callback):
    def __init__(self):
        super().__init__()
        self.history: Dict[str, List[float | int | str]] = {}

    def on_train_begin(self, logs=None):
        self.epoch = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        # Set the history attribute on the model after the epoch ends. This will
        # make sure that the state which is set is the latest one.
        self.model.history = self

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self.history, index=self.epoch)
