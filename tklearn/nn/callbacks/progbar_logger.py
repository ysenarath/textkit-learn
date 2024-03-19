import math
from typing import Optional

from tklearn.nn.callbacks.base import ModelCallback
from tklearn.utils.progbar import Progbar

__all__ = [
    "ProgbarLogger",
]


class ProgbarLogger(ModelCallback):
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
        self.bar_format = f"{l_bar}{{bar}} |"
        # {{r_bar}}
        self.pbar = None

    def on_train_begin(self, logs=None):
        total = self.params["epochs"] * self.params["steps"]
        self.pbar = Progbar(
            total=total,
            desc=self.desc.format(epoch=0, loss=math.inf),
            postfix={},
            bar_format=self.bar_format,
        )

    def on_train_batch_end(self, batch, logs=None):
        self.pbar.update(1)
        self.pbar.refresh()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.pbar.set_postfix(logs, refresh=False)
        self.pbar.set_description_str(
            self.desc.format(
                epoch=epoch + 1,
                loss=logs.get("loss", math.inf),
            ),
            refresh=False,
        )

    def on_train_end(self, logs=None):
        self.pbar.close()
