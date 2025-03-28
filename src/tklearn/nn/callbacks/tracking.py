from typing import Optional, Union

import mlflow
from mlflow.tracking.fluent import ActiveRun

from tklearn.nn.callbacks.base import Callback

__all__ = [
    "TrackingCallback",
]


class TrackingCallback(Callback):
    def __init__(
        self,
        prefix: Optional[str] = None,
        exclude: Optional[str] = None,
        run: Union[ActiveRun, str, None] = None,
    ):
        super().__init__()
        # the actual run used to track the trainer progress
        if prefix is None:
            prefix = ""
        self.prefix = str(prefix).strip()
        self.exclude = exclude
        if isinstance(run, str):
            self.run_id = run
        elif run is not None:
            self.run_id = run.info.run_id
        else:
            self.run_id = None

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

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        mlflow.log_metric(
            f"{self.prefix}epoch", epoch, step=epoch, run_id=self.run_id
        )
        logs = {
            f"{self.prefix}{k}": v
            for k, v in logs.items()
            if self.is_loggable(k, v)
        }
        # metrics must be a Dict[str, float]
        mlflow.log_metrics(logs, step=epoch, run_id=self.run_id)

    def is_loggable(self, key: str, value) -> bool:
        return key not in self.exclude and isinstance(value, (int, float))
