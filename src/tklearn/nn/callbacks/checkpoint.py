from pathlib import Path
from typing import Union

import torch

from tklearn.nn.callbacks.base import Callback

__all__ = [
    "ModelCheckpoint",
]


class ModelCheckpoint(Callback):
    def __init__(
        self,
        filepath: Union[str, Path],
        # monitor: str = 'val_loss',
        verbose: int = 0,
        # save_best_only: bool = False,
        # save_weights_only: bool = False,
        # mode: str = 'auto',
        save_freq="epoch",
        # options=None,
        # initial_value_threshold=None,
        **kwargs,
    ):
        super().__init__()
        self.filepath: Path = Path(filepath)
        self.verbose = verbose
        self.save_freq = save_freq
        self.step = 0
        self.epoch = 0

    def on_train_begin(self, logs=None):
        # reset the value of step
        self.filepath.mkdir(parents=True, exist_ok=True)
        self.step = 0

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.save_freq == "batch":
            self._save_model()

    def on_epoch_end(self, epoch: int, logs=None):
        self.epoch = epoch
        if self.save_freq == "epoch":
            self._save_model()

    def _save_model(self):
        if self.model is None:
            msg = "model not found"
            raise ValueError(msg)
        filename = f"checkpoints-epoch-{self.epoch}-step-{self.step}.pt"
        filepath = self.filepath / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if not isinstance(self.model, torch.nn.Module):
            msg = (
                "model should be an instance of torch.nn.Module,"
                f" got {self.model.__class__.__name__}"
            )
            raise TypeError(msg)
        torch.save(self.model.state_dict(), filepath)
