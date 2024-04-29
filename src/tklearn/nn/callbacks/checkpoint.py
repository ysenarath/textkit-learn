from pathlib import Path
from typing import Union

import torch
from transformers import PreTrainedModel

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

    def on_train_begin(self, logs=None):
        # reset the value of step
        self.filepath.mkdir(parents=True, exist_ok=True)
        self.step = 0

    def on_epoch_end(self, epoch: int, logs=None):
        if self.save_freq == "epoch":
            self.save_model()

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.save_freq == "batch":
            self.save_model()

    def save_model(self):
        model = self.model
        if model is None:
            msg = "model not found"
            raise ValueError(msg)
        if isinstance(model, PreTrainedModel):
            model_path = self.filepath / f"checkpoint-{self.step:0004d}"
            model.save_pretrained(model_path)
        elif isinstance(model, torch.nn.Module):
            torch.save(
                model.state_dict(),
                self.filepath / f"checkpoint-{self.step:0004d}.pt",
            )
        else:
            msg = f"unsupported model: {type(model).__name__}"
            raise TypeError(msg)
