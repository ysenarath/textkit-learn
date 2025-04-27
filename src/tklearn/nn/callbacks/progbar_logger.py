from typing import Dict, Optional

from rich.console import Group
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table

from tklearn.nn.callbacks.base import Callback

__all__ = [
    "ProgbarLogger",
]


def get_performance_report(logs, exclude=None, epoch=None) -> Dict[str, str]:
    if logs is None:
        logs = {}
    row = {}
    for k, v in logs.items():
        if exclude is not None and k in exclude:
            continue
        if isinstance(v, float):
            v = f"{v:0.4f}"
        elif isinstance(v, int):
            v = f"{v:d}"
        elif isinstance(v, str):
            v = (v[:10] + "...") if len(v) > 10 else v
        else:
            continue
        row[k] = v
    if epoch is not None:
        row["Epoch"] = f"{epoch}"
    return row


class ProgbarLogger(Callback):
    table: Table
    progress: Progress
    live: Live

    def __init__(self, exclude: Optional[list] = None):
        super().__init__()
        self.exclude = exclude
        self.live = Live(
            Group(
                Table(),
                Progress(
                    SpinnerColumn(),
                    *Progress.get_default_columns(),
                    TimeElapsedColumn(),
                ),
            ),
            vertical_overflow="visible",
        )
        group: Group = self.live.renderable
        self.table, self.progress = group.renderables
        self.training = False
        self.train_epoch_tracker = None
        self.train_batch_tracker = None
        self.pred_batch_tracker = None
        self._zero_based_epoch = False
        self._zero_based_step = False

    def on_train_begin(self, logs=None):
        self.training = True
        num_epochs = self.params["epochs"]
        num_steps = self.params["steps"]
        self.train_epoch_tracker = self.progress.add_task(
            "Epoch", total=num_epochs
        )
        self.train_batch_tracker = self.progress.add_task(
            "Batch[Train]", total=num_steps
        )
        self.live.start()
        self.progress.update(self.train_epoch_tracker, completed=0)
        self.live.refresh()

    def on_train_batch_begin(self, batch, logs=None):
        if batch == 0:
            self._zero_based_step = True

    def on_train_batch_end(self, batch, logs=None):
        if self.train_batch_tracker is None:
            return
        # update by 1 or to batch
        self.progress.update(self.train_batch_tracker, advance=1)
        self.live.refresh()

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            self._zero_based_epoch = True

    def on_epoch_end(self, epoch, logs=None):
        if self._zero_based_epoch:
            natural_epoch = epoch + 1
        else:
            natural_epoch = epoch
        report = get_performance_report(
            logs,
            exclude=self.exclude,
            epoch=natural_epoch,
        )
        if not self.table.columns:
            for column in sorted(report.keys()):
                self.table.add_column(column, justify="left")
        self.table.add_row(
            *[
                report.get(column.header, "N/A")
                for column in self.table.columns
            ],
        )

        if self.train_epoch_tracker is not None:
            self.progress.update(self.train_epoch_tracker, advance=1)

        if self.train_batch_tracker:
            self.progress.update(self.train_batch_tracker, completed=0)

        self.live.refresh()

    def on_predict_begin(self, logs=None):
        if "pred_steps" not in self.params:
            return
        self.pred_batch_tracker = self.progress.add_task(
            "Batch[Predict]", total=self.params["pred_steps"]
        )
        self.live.start()
        self.progress.update(self.pred_batch_tracker, completed=0)
        self.live.refresh()

    def on_predict_batch_end(self, batch, logs=None):
        if self.pred_batch_tracker is None:
            return
        self.progress.update(self.pred_batch_tracker, advance=1)
        self.live.refresh()

    def on_predict_end(self, logs=None):
        if self.pred_batch_tracker is not None:
            self.progress.remove_task(self.pred_batch_tracker)
            self.pred_batch_tracker = None
        if self.live is None or self.training:
            return
        self.live.stop()

    def on_train_end(self, logs=None):
        if self.train_batch_tracker is not None:
            self.progress.remove_task(self.train_batch_tracker)
            self.train_batch_tracker = None
        if self.live is None:
            return
        self.live.stop()
        self.training = False
