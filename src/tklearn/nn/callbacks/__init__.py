"""Callbacks of `Trainer`, modelled on ``keras.callbacks``.

`Callback` is the base class; `LambdaCallback` builds one from functions.
The rest are ready-made:

- `EarlyStopping` and `TerminateOnNaN` stop training.
- `ModelCheckpoint` saves the model weights.
- `ReduceLROnPlateau` lowers the learning rate when progress stalls.
- `ProgbarLogger` shows progress bars, and `CSVLogger` writes the epoch
  logs to a file.

`Trainer.history` holds the epoch logs (Keras' ``History``), and
``Trainer(lr_scheduler=...)`` schedules the learning rate (Keras'
``LearningRateScheduler``).
"""

from tklearn.nn.callbacks.base import Callback, LambdaCallback
from tklearn.nn.callbacks.checkpoint import ModelCheckpoint
from tklearn.nn.callbacks.early_stopping import EarlyStopping, TerminateOnNaN
from tklearn.nn.callbacks.loggers import CSVLogger, ProgbarLogger
from tklearn.nn.callbacks.reduce_lr import ReduceLROnPlateau

__all__ = [
    "CSVLogger",
    "Callback",
    "EarlyStopping",
    "LambdaCallback",
    "ModelCheckpoint",
    "ProgbarLogger",
    "ReduceLROnPlateau",
    "TerminateOnNaN",
]
