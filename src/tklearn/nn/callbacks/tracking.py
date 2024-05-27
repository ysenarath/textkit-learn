from typing import TYPE_CHECKING, Optional

from tklearn.nn.callbacks.base import Callback

__all__ = [
    "TrackingCallback",
]

if TYPE_CHECKING:
    from octoflow.tracking.models import Run


class TrackingCallback(Callback):
    def __init__(
        self,
        run: "Run",
        step: Optional[int] = None,
        prefix: Optional[str] = None,
        exclude: Optional[str] = None,
    ):
        super().__init__()
        # the actual run used to track the trainer progress
        self.run = run
        self.step = step
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

    def on_train_begin(self, logs: Optional[dict] = None):
        if self.run is None:
            return

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        if self.run is None:
            return
        prefix = ""
        if len(self.prefix) > 0:
            prefix = f"{self.prefix}{self.delemiter}"
        epoch_log_ix = self.run.log_param(f"{prefix}epoch", epoch, step=self.step)
        logs = {f"{prefix}{k}": v for k, v in logs.items() if k not in self.exclude}
        # and isinstance(v, (int, float, str))
        self.run.log_metrics(logs, step=epoch_log_ix)
