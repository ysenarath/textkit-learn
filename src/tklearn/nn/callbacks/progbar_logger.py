import math
from typing import Optional

from tklearn.nn.callbacks.base import Callback
from tklearn.nn.utils.progbar import ProgressBar

__all__ = [
    "ProgbarLogger",
]


class ProgbarLogger(Callback):
    """
    Callback that logs the progress of training using a progress bar.

    Parameters
    ----------
    desc : str, optional
        The description template for the progress bar.
    prefix : str, optional
        The prefix for the progress bar. Default is "Training".

    Attributes
    ----------
    desc : str
        The description template for the progress bar.
    prefix : str
        The prefix for the progress bar.
    bar_format : str
        The format string for the progress bar.
    pbar : Progbar
        The progress bar object.

    Methods
    -------
    on_train_begin(logs=None)
        Called at the beginning of training.
    on_train_batch_end(batch, logs=None)
        Called at the end of each training batch.
    on_epoch_end(epoch, logs=None)
        Called at the end of each epoch.
    on_train_end(logs=None)
        Called at the end of training.
    """

    def __init__(
        self,
        desc: Optional[
            str
        ] = "Training | Epoch: {epoch:4d} | Loss: {loss:0.4f} | Progress",
        prefix: Optional[str] = "Training",
    ):
        super().__init__()
        self.desc = desc
        self.prefix = prefix
        l_bar = "{desc}: {percentage:3.0f}% |"
        self.bar_format = f"{l_bar}{{bar}}{{r_bar}}"
        self.pbar = None
        self.pred_pbar = None

    def on_train_begin(self, logs=None):
        """
        Called at the beginning of training.

        Parameters
        ----------
        logs : dict, optional
            Dictionary of logs. Default is None.
        """
        total = self.params["epochs"] * self.params["steps"]
        self.pbar = ProgressBar(
            total=total,
            desc=self.desc.format(epoch=0, loss=math.inf),
            postfix={},
            bar_format=self.bar_format,
        )

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
        self.pbar.update(1)
        self.pbar.refresh()

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
        if logs is None:
            logs = {}
        essential_logs = {
            key: value for key, value in logs.items() if "loss" in key or "acc" in key
        }
        self.pbar.set_postfix(essential_logs, refresh=False)
        self.pbar.set_description_str(
            self.desc.format(
                epoch=epoch + 1,
                loss=logs.get("loss", math.inf),
            ),
            refresh=False,
        )

    def on_train_end(self, logs=None):
        """
        Called at the end of training.

        Parameters
        ----------
        logs : dict, optional
            Dictionary of logs. Default is None.
        """
        self.pbar.close()
        self.pbar = None

    def on_predict_begin(self, logs=None):
        self.pred_pbar = ProgressBar(
            total=self.params["pred_steps"],
            desc="Predicting | Progress",
            postfix={},
            bar_format=self.bar_format,
            leave=False,
        )

    def on_predict_batch_end(self, batch, logs=None):
        self.pred_pbar.update(1)
        self.pred_pbar.refresh()

    def on_predict_end(self, logs=None):
        self.pred_pbar.close()
        self.pred_pbar = None
