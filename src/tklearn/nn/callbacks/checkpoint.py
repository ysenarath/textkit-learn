import warnings
from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch
from safetensors.torch import save_model

from octoflow import logging
from tklearn.nn.callbacks.base import Callback

__all__ = [
    "ModelCheckpoint",
]

logger = logging.get_logger(__name__)


def _get_monitor_attrs(cls: type, monitor: str, mode: str, best: float = None):
    if mode not in ["auto", "min", "max"]:
        warnings.warn(
            f"{cls.__name__} mode '{mode}' is unknown, fallback to auto mode.",
            stacklevel=2,
        )
        mode = "auto"
    if mode == "min":
        monitor_op = np.less
        if best is None:
            best = np.Inf
    elif mode == "max":
        monitor_op = np.greater
        if best is None:
            best = -np.Inf
    else:
        if (
            monitor.endswith("acc")
            or monitor.endswith("accuracy")
            or monitor.endswith("auc")
            or monitor.endswith("_score")
        ):
            monitor_op = np.greater
            if best is None:
                best = -np.Inf
        elif monitor.endswith("loss") or monitor.endswith("error"):
            monitor_op = np.less
            if best is None:
                best = np.Inf
        else:
            msg = f"could not infer the metric direction for {monitor}."
            raise ValueError(msg)
    return monitor_op, best


def _validate_save_freq(
    cls, save_freq: Union[str, int]
) -> Union[Literal["epoch"], int]:
    if save_freq == "batch":
        save_freq = 1
    if isinstance(save_freq, float):
        save_freq = int(save_freq)
    if save_freq != "epoch" and not isinstance(save_freq, int):
        msg = (
            f"{cls.__name__} save_freq should be 'epoch' or an integer, got {save_freq}"
        )
        raise ValueError(msg)
    return save_freq


def _validate_save_weights_only(save_weights_only: bool, filepath: Union[Path, str]):
    if not isinstance(save_weights_only, bool):
        raise ValueError(
            f"save_weights_only should be a boolean, got {save_weights_only}"
        )
    if filepath.endswith(".pt"):
        return save_weights_only
    msg = (
        "save_weights_only should be True when filepath"
        f" is not a '.pt' file, got {save_weights_only}"
    )
    if save_weights_only:
        if filepath.endswith(".safetensors"):
            return save_weights_only
        msg = (
            "save_weights_only should be False when filepath"
            f" is not a '.pt' or '.safetensors' file, got {save_weights_only}"
        )
    raise ValueError(msg)


class ModelCheckpoint(Callback):
    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        verbose: int = 0,
        save_best_only: bool = False,
        save_weights_only: bool = False,
        mode: Literal["auto", "min", "max"] = "auto",
        save_freq: Union[
            Literal["epoch", "batch"], int
        ] = "epoch",  # "epoch" or "batch" or integer
        initial_value_threshold: Union[float, int, None] = None,
    ):
        super().__init__()
        cls = type(self)
        self.filepath: str = str(filepath)
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_weights_only = _validate_save_weights_only(
            save_weights_only, self.filepath
        )
        self.mode = mode
        self.save_freq = _validate_save_freq(cls, save_freq)
        self.initial_value_threshold = initial_value_threshold
        # set mode and initialize the best value
        self.monitor_op, self.best = _get_monitor_attrs(
            cls, self.monitor, self.mode, self.initial_value_threshold
        )
        self.step = 0
        self.epoch = 0

    def on_train_begin(self, logs=None):
        cls = type(self)
        self.monitor_op, self.best = _get_monitor_attrs(
            cls, self.monitor, self.mode, self.initial_value_threshold
        )
        self.step = 0
        self.epoch = 0

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if not isinstance(self.save_freq, int):
            return
        if self.step % self.save_freq == 0:
            self._save_model(logs)

    def on_epoch_end(self, epoch: int, logs=None):
        self.epoch = epoch
        if self.save_freq == "epoch":
            self._save_model(logs)

    def _save_model(self, logs=None):
        logs = logs or {}
        filepath = Path(self.filepath.format(epoch=self.epoch, step=self.step, **logs))
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                return
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    logger.debug(
                        f"Monitor {self.monitor} improved from {self.best:.5f} to {current:.5f}"
                        f" at epoch {self.epoch}, saving model to {filepath}."
                    )
                self.best = current
                self._save_model_internal(filepath)
            else:
                if self.verbose > 0:
                    logger.debug(
                        f"Monitor {self.monitor} did not improve from"
                        f" {self.best:.5f} at epoch {self.epoch}."
                    )
        else:
            if self.verbose > 0:
                logger.debug(f"Save model at epoch {self.epoch} to {filepath}.")
            self._save_model_internal(filepath)

    def _save_model_internal(self, filepath: Path):
        if self.model is None:
            msg = f"{self.__class__.__name__} model is None, cannot save model"
            raise ValueError(msg)
        if not isinstance(self.model, torch.nn.Module):
            msg = (
                f"model should be an instance of torch.nn.Module,"
                f" got {self.model.__class__.__name__}"
            )
            raise TypeError(msg)
        if self.save_weights_only:
            if filepath.suffix == ".safetensors":
                save_model(self.model, filepath)
            else:
                torch.save(self.model.state_dict(), filepath)
        else:
            torch.save(self.model, filepath)
