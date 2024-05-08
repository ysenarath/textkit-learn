from typing import TYPE_CHECKING, Optional

from tklearn.nn.callbacks.base import Callback

__all__ = [
    "TrackingCallback",
]

if TYPE_CHECKING:
    from octoflow import Run, Value


class TrackingCallback(Callback):
    def __init__(
        self,
        run: "Run",
        step: Optional["Value"] = None,
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
        self.epoch_key_fmt = "{prefix}{delemiter}epoch"
        self.delemiter = "."
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

    def on_train_begin(
        self,
        logs: Optional[dict] = None,
    ):
        if self.run is None:
            return

    def on_epoch_end(
        self,
        epoch: int,
        logs: Optional[dict] = None,
    ):
        if self.run is None:
            return
        epoch_key = self.epoch_key_fmt.format(
            prefix=self.prefix,
            delemiter=self.delemiter if len(self.prefix) > 0 else "",
        )
        epoch_val = self.run.log_param(
            epoch_key,
            epoch,
            step=self.step,
        )
        logs = {k: v for k, v in logs.items() if k not in self.exclude}
        # and isinstance(v, (int, float, str))
        self.run.log_metrics(logs, step=epoch_val, prefix=self.prefix)
