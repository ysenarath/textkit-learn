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


def progress_display() -> Progress:
    return Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    )


class ProgbarLogger(Callback):
    def __init__(self, exclude: Optional[list] = None):
        super().__init__()
        self.exclude = exclude
        self.progress = None
        self.table: Table = None
        self.live: Live = None
        self.epoch_tracker = None
        self.batch_tracker_train = None
        self.batch_tracker_valid = None
        self._zero_based_epoch = False
        self._zero_based_step = False
        self._started_by_predict = False

    def on_train_begin(self, logs=None):
        """
        Called at the beginning of training.

        Parameters
        ----------
        logs : dict, optional
            Dictionary of logs. Default is None.
        """
        self.progress = progress_display()
        self.table = Table()
        layout = Group(self.table, self.progress)
        num_epochs = self.params["epochs"]
        num_steps = self.params["steps"]
        self.live = Live(layout, vertical_overflow="visible")
        self.epoch_tracker = self.progress.add_task("Epoch", total=num_epochs)
        self.batch_tracker_train = self.progress.add_task(
            "Batch[Train]", total=num_steps
        )
        self.live.start()

    def on_train_batch_end(self, batch, logs=None):
        """
        Called at the end of each training batch.

        Parameters
        ----------
        batch : int
            The batch index.
        logs : dict, optional
            Dictionary of logs. Default is None.
        """
        if batch == 0:
            self._zero_based_step = True
        if self.batch_tracker_train is None:
            return
        # update by 1 or to batch
        self.progress.update(self.batch_tracker_train, advance=1)
        self.live.refresh()

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch.

        Parameters
        ----------
        epoch : int
            The epoch index.
        logs : dict, optional
            Dictionary of logs. Default is None.
        """
        if epoch == 0:
            self._zero_based_epoch = True
        if self.epoch_tracker is None:
            return
        report = get_performance_report(
            logs,
            exclude=self.exclude,
            epoch=epoch + 1 if self._zero_based_epoch else epoch,
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
        self.progress.update(self.epoch_tracker, advance=1)
        self.live.refresh()

    def on_train_end(self, logs=None):
        """
        Called at the end of training.

        Parameters
        ----------
        logs : dict, optional
            Dictionary of logs. Default is None.
        """
        if self.batch_tracker_train is not None:
            self.progress.remove_task(self.batch_tracker_train)
        if self.live is None:
            return
        self.live.stop()
        self.progress = None
        self.live = None

    def on_predict_begin(self, logs=None):
        if "pred_steps" not in self.params:
            return
        if self.progress is None:
            self.progress = progress_display()
            layout = Group(self.progress)
            self.live = Live(layout, vertical_overflow="visible")
            self.live.start()
            self._started_by_predict = True
        self.batch_tracker_valid = self.progress.add_task(
            "Batch[Valid]", total=self.params["pred_steps"]
        )

    def on_predict_batch_end(self, batch, logs=None):
        if self.batch_tracker_valid is None:
            return
        self.progress.update(self.batch_tracker_valid, advance=1)
        self.live.refresh()

    def on_predict_end(self, logs=None):
        if self.batch_tracker_valid is None:
            return
        self.progress.remove_task(self.batch_tracker_valid)
        self.batch_tracker_valid = None
        if not self._started_by_predict:
            return
        if self.live is None:
            return
        self.live.stop()
        self.progress = None
        self.live = None
        self._started_by_predict = False
