from typing import Optional

from octoflow import Run, Value
from tklearn.nn.callbacks.base import ModelCallback

__all__ = [
    "TrackingCallback",
]


class TrackingCallback(ModelCallback):
    def __init__(
        self,
        run: Run,
        step: Optional[Value] = None,
        prefix: Optional[str] = None,
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
        self.run.log_metrics(logs, step=epoch_val, prefix=self.prefix)
