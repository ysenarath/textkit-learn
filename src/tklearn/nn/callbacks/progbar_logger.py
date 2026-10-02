from __future__ import annotations

from typing import Any

import pandas as pd
from tabulate import tabulate
from tqdm import auto as tqdm

from tklearn.nn.callbacks.base import Callback

__all__ = ["ProgbarLogger"]


def get_performance_report(logs, exclude=None, epoch=None) -> dict[str, str]:
    """
    Formats logs into a performance report dictionary.
    """
    if logs is None:
        logs = {}
    row = {}
    for k, v in logs.items():
        if exclude is not None and k in exclude:
            continue
        # Format float values to 4 decimal places
        if isinstance(v, float):
            v = f"{v:0.4f}"
        # Format integer values
        elif isinstance(v, int):
            v = f"{v:d}"
        # Truncate string values if too long
        elif isinstance(v, str):
            v = (v[:10] + "...") if len(v) > 10 else v
        else:
            # Skip other types
            continue
        row[k] = v
    # Add epoch number if provided
    if epoch is not None:
        row["epoch"] = f"{epoch}"
    return row


class ProgbarLogger(Callback):
    """Show tqdm progress bars and a table of epoch results.

    During `Trainer.fit` it shows an epoch bar and a per-epoch batch bar, and
    prints the logs of every finished epoch as a table. During
    `Evaluator.evaluate`, `Predictor.predict` and `Encoder.encode` it shows a
    batch bar.

    Parameters
    ----------
    exclude : list of str, optional
        Log keys to leave out of the table.
    epoch_desc, batch_desc, test_desc, pred_desc : str
        Labels of the epoch, training batch, evaluation and prediction bars.
    """

    def __init__(
        self,
        exclude: list[str] | None = None,
        epoch_desc: str = "Epoch",
        batch_desc: str = "Batch",
        test_desc: str = "Evaluating",
        pred_desc: str = "Predicting",
    ) -> None:
        super().__init__()
        self.exclude = exclude
        self.epoch_desc = epoch_desc
        self.batch_desc = batch_desc
        self.test_desc = test_desc
        self.pred_desc = pred_desc
        self.epoch_bar: tqdm.tqdm | None = None
        self.batch_bar: tqdm.tqdm | None = None
        self.eval_bar: tqdm.tqdm | None = None
        # one formatted row per finished epoch
        self.history: list[dict[str, str]] = []

    @staticmethod
    def _close(bar: tqdm.tqdm | None) -> None:
        if bar is not None:
            bar.close()

    # --- training --------------------------------------------------------

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        self.history = []
        self.epoch_bar = tqdm.tqdm(
            total=self.params.get("epochs"),
            desc=self.epoch_desc,
            unit="epoch",
            leave=True,
        )

    def on_epoch_begin(
        self, epoch: int, logs: dict[str, Any] | None = None
    ) -> None:
        self.batch_bar = tqdm.tqdm(
            total=self.params.get("steps"),
            desc=self.batch_desc,
            unit="batch",
            leave=False,
        )

    def on_train_batch_end(
        self, batch_idx: int, logs: dict[str, Any] | None = None
    ) -> None:
        if self.batch_bar is not None:
            self.batch_bar.update(1)

    def on_epoch_end(
        self, epoch: int, logs: dict[str, Any] | None = None
    ) -> None:
        self._close(self.batch_bar)
        self.batch_bar = None
        if self.epoch_bar is not None:
            self.epoch_bar.update(1)
        # epochs are zero-based internally; show them one-based
        report = get_performance_report(
            logs, exclude=self.exclude, epoch=epoch + 1
        )
        self.history.append(report)
        table = tabulate(
            pd.DataFrame(self.history), headers="keys", tablefmt="pipe"
        )
        tqdm.tqdm.write(f"\n{table}\n")

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        self._close(self.epoch_bar)
        self.epoch_bar = None

    # --- evaluation and prediction ---------------------------------------

    def _open_eval_bar(self, desc: str, total: int | None) -> None:
        self._close(self.eval_bar)
        self.eval_bar = tqdm.tqdm(
            total=total, desc=desc, unit="batch", leave=False
        )

    def _advance_eval_bar(self) -> None:
        if self.eval_bar is not None:
            self.eval_bar.update(1)

    def _close_eval_bar(self) -> None:
        self._close(self.eval_bar)
        self.eval_bar = None

    def on_test_begin(self, logs: dict[str, Any] | None = None) -> None:
        self._open_eval_bar(self.test_desc, self.params.get("test_steps"))

    def on_test_batch_end(
        self, batch_idx: int, logs: dict[str, Any] | None = None
    ) -> None:
        self._advance_eval_bar()

    def on_test_end(self, logs: dict[str, Any] | None = None) -> None:
        self._close_eval_bar()

    def on_predict_begin(self, logs: dict[str, Any] | None = None) -> None:
        self._open_eval_bar(self.pred_desc, self.params.get("pred_steps"))

    def on_predict_batch_end(
        self, batch_idx: int, logs: dict[str, Any] | None = None
    ) -> None:
        self._advance_eval_bar()

    def on_predict_end(self, logs: dict[str, Any] | None = None) -> None:
        self._close_eval_bar()
