from __future__ import annotations

from typing import Optional

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
    """
    Callback to display progress bars and performance reports during training and prediction.
    Uses tqdm for progress visualization.
    """

    def __init__(self, exclude: Optional[list] = None):
        super().__init__()
        self.exclude = exclude

        self.training = False

        self.train_epochs = None
        self.train_steps = None  # Corrected typo from original code
        self.pred_steps = None

        self._zero_based_train_step = False
        self._zero_based_train_epoch = False
        self._zero_based_pred_step = False

        # tqdm progress bar instances
        self.epoch_progress_bar: Optional[tqdm.tqdm] = None
        self.batch_progress_bar: Optional[tqdm.tqdm] = None
        self.predict_progress_bar: Optional[tqdm.tqdm] = None

        # history of performance reports
        self.history = []

    def on_train_begin(self, logs=None):
        """
        Initializes the progress bar for training epochs.
        """
        self.training = True
        self.train_epochs = self.params.get("epochs")  # Use .get() for safety
        self.train_steps = self.params.get("steps")  # Use .get() for safety

        if self.train_epochs is not None:
            # initialize the progress bar for training epochs
            self.epoch_progress_bar = tqdm.tqdm(
                total=self.train_epochs,
                desc="Epochs",
                unit="epoch",
                leave=True,  # Keep the bar after completion
            )

        self.history = []  # Reset history at the start of training

    def on_train_batch_begin(self, batch, logs=None):
        """
        Notes if batch indexing is zero-based.
        Initializes batch progress bar at the start of each epoch's first batch.
        """
        if batch == 0:
            self._zero_based_train_step = True
            # Initialize batch progress bar at the start of a new epoch
            if self.train_steps is not None:
                self.batch_progress_bar = tqdm.tqdm(
                    total=self.train_steps,
                    desc=f"Epochs {self.current_epoch + 1} Batches",  # Assuming current_epoch is available from base Callback
                    unit="batch",
                    leave=False,  # Remove the bar after completion of the epoch
                )

    def on_train_batch_end(self, batch, logs=None):
        """
        Updates the training step progress bar with current batch logs.
        """
        # if self._zero_based_train_step:
        #     natural_step = batch + 1
        # else:
        #     natural_step = batch
        # update the training step progress bar
        if self.batch_progress_bar is not None:
            # Increment by 1 for the completed batch
            self.batch_progress_bar.update(1)

    def on_epoch_begin(self, epoch, logs=None):
        """
        Notes if epoch indexing is zero-based.
        """
        # Assuming 'epoch' here is the zero-based index provided by the trainer
        self.current_epoch = epoch  # Store current epoch index
        if epoch == 0:
            self._zero_based_train_epoch = True

    def on_epoch_end(self, epoch, logs=None):
        """Finalizes the progress bar for the current epoch."""
        # Increment by 1 for the completed epoch
        if self.epoch_progress_bar is not None:
            self.epoch_progress_bar.update(1)

        # Close the batch progress bar for this epoch
        if self.batch_progress_bar is not None:
            self.batch_progress_bar.close()
            self.batch_progress_bar = None  # Reset for the next epoch

        # update table for the current epoch
        if self._zero_based_train_epoch:
            natural_epoch = epoch + 1
        else:
            natural_epoch = epoch
        report = get_performance_report(
            logs,
            exclude=self.exclude,
            epoch=natural_epoch,
        )
        self.history.append(report)

        # Use print directly as tqdm manages console output
        tqdm.tqdm.write(
            "\n{}\n".format(
                tabulate(
                    pd.DataFrame(self.history), headers="keys", tablefmt="pipe"
                )
            )
        )

    def on_predict_begin(self, logs=None):
        """
        Initializes the progress bar for prediction steps.
        """
        if "pred_steps" not in self.params:
            return
        total = self.params["pred_steps"]
        # initialize the progress bar for prediction steps
        self.predict_progress_bar = tqdm.tqdm(
            total=total,
            desc="Predicting",
            unit="step",
            leave=True,  # Keep the bar after completion
        )

    def on_predict_batch_begin(self, batch, logs=None):
        """
        Notes if prediction batch indexing is zero-based.
        """
        if batch == 0:
            self._zero_based_pred_step = True

    def on_predict_batch_end(self, batch, logs=None):
        """
        Updates the prediction step progress bar.
        """
        # update the prediction step progress bar
        if self.predict_progress_bar is None:
            return
        # Increment by 1 for the completed batch
        self.predict_progress_bar.update(1)

    def on_predict_end(self, logs=None):
        """
        Finalizes the prediction progress bar.
        """
        if self.predict_progress_bar is None:
            return
        # finalize anything stated during prediction
        self.predict_progress_bar.close()
        self.predict_progress_bar = None

    def on_train_end(self, logs=None):
        """
        Finalizes the training epoch progress bar.
        """
        if not self.training:
            return
        # finalize anything stated during training
        if self.epoch_progress_bar is None:
            return
        self.epoch_progress_bar.close()
        self.epoch_progress_bar = None
