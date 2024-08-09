from typing import Optional

import mlflow

from tklearn.nn.callbacks.base import Callback

__all__ = [
    "TrackingCallback",
]


class TrackingCallback(Callback):
    def __init__(self, prefix: Optional[str] = None, exclude: Optional[str] = None):
        super().__init__()
        # the actual run used to track the trainer progress
        if prefix is None:
            prefix = ""
        self.prefix = str(prefix).strip()
        self.delemiter = "_"
        self.exclude = exclude

    @property
    def exclude(self) -> set:
        return self._exclude

    @exclude.setter
    def exclude(self, values):
        if values is None:
            values = []
        elif isinstance(values, str):
            values = [values]
        self._exclude = set(values)

    def on_train_begin(self, logs: Optional[dict] = None): ...

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        prefix = ""
        if len(self.prefix) > 0:
            prefix = f"{self.prefix}{self.delemiter}"
        mlflow.log_metric(f"{prefix}epoch", epoch, step=epoch)
        logs = {f"{prefix}{k}": v for k, v in logs.items() if k not in self.exclude}
        # and isinstance(v, (int, float, str))
        mlflow.log_metrics(logs, step=epoch)
