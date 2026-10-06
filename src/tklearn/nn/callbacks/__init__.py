"""Callbacks of `Trainer`, modelled on ``keras.callbacks``.

`Callback` is the base class; `LambdaCallback` builds one from functions.
The rest are ready-made:

- `EarlyStopping` and `TerminateOnNaN` stop training.
- `ModelCheckpoint` saves the model weights.
- `ReduceLROnPlateau` lowers the learning rate when progress stalls.
- `ProgbarLogger` shows progress bars, and `CSVLogger` writes the epoch
  logs to a file.
- `RunLogger` records the configuration, per-step and per-epoch metrics
  of runs as plain files, safe on NFS shared by SLURM jobs; `load_runs`
  reads them.

`Trainer.history` holds the epoch logs (Keras' ``History``), and
``Trainer(lr_scheduler=...)`` schedules the learning rate (Keras'
``LearningRateScheduler``).
"""

from tklearn.nn.callbacks.base import Callback
from tklearn.nn.callbacks.checkpoint import ModelCheckpoint
from tklearn.nn.callbacks.csv_logger import CSVLogger
from tklearn.nn.callbacks.early_stopping import EarlyStopping
from tklearn.nn.callbacks.lambda_callback import LambdaCallback
from tklearn.nn.callbacks.progbar_logger import ProgbarLogger
from tklearn.nn.callbacks.reduce_lr import ReduceLROnPlateau
from tklearn.nn.callbacks.run_logger import RunLogger, load_runs
from tklearn.nn.callbacks.terminate_on_nan import TerminateOnNaN

__all__ = [
    "CSVLogger",
    "Callback",
    "EarlyStopping",
    "LambdaCallback",
    "ModelCheckpoint",
    "ProgbarLogger",
    "ReduceLROnPlateau",
    "RunLogger",
    "TerminateOnNaN",
    "load_runs",
]
