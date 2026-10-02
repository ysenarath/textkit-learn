from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Union

import torch
from safetensors.torch import save_model

from tklearn.nn.callbacks._monitor import (
    Mode,
    is_improvement,
    is_valid_value,
    resolve_mode,
    worst_value,
)
from tklearn.nn.callbacks.base import Callback

__all__ = [
    "ModelCheckpoint",
]

logger = logging.getLogger(__name__)


def _validate_save_freq(
    cls, save_freq: Union[str, int]
) -> Union[Literal["epoch"], int]:
    if save_freq == "batch":
        save_freq = 1
    if isinstance(save_freq, float):
        save_freq = int(save_freq)
    if save_freq != "epoch" and not isinstance(save_freq, int):
        msg = f"{cls.__name__} save_freq should be 'epoch' or an integer, got {save_freq}"
        raise ValueError(msg)
    return save_freq


def _validate_save_weights_only(
    save_weights_only: bool, filepath: str
) -> bool:
    if filepath.endswith(".pt"):
        return bool(save_weights_only)
    if filepath.endswith(".safetensors"):
        if not save_weights_only:
            msg = (
                "'.safetensors' checkpoints hold weights only; "
                "set save_weights_only=True"
            )
            raise ValueError(msg)
        return True
    msg = (
        "checkpoint filepath must end in '.pt' or '.safetensors', "
        f"got {filepath!r}"
    )
    raise ValueError(msg)


class ModelCheckpoint(Callback):
    """Save the model during training.

    Parameters
    ----------
    filepath : str
        Destination; may contain ``{epoch}``, ``{step}`` and any log key,
        e.g. ``"ckpt/epoch={epoch}-{valid_loss:.3f}.pt"``. Use ``.pt`` (full
        model, or state dict with `save_weights_only`) or ``.safetensors``
        (weights only).
    monitor : str, default="valid_loss"
        Log key compared when `save_best_only` is set.
    verbose : int, default=0
        Log decisions at debug level when > 0.
    save_best_only : bool, default=False
        Only save when `monitor` improves.
    save_weights_only : bool, default=False
        Save the state dict instead of the whole model.
    mode : {"auto", "min", "max"}, default="auto"
        Whether lower or higher `monitor` is better.
    save_freq : "epoch", "batch" or int, default="epoch"
        Save after every epoch, every batch, or every N batches.
    initial_value_threshold : float, optional
        Only save once `monitor` beats this value.
    """

    def __init__(
        self,
        filepath: str | Path,
        monitor: str = "valid_loss",
        verbose: int = 0,
        save_best_only: bool = False,
        save_weights_only: bool = False,
        mode: Mode = "auto",
        save_freq: Union[Literal["epoch", "batch"], int] = "epoch",
        initial_value_threshold: float | None = None,
    ) -> None:
        super().__init__()
        self.filepath = str(filepath)
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_weights_only = _validate_save_weights_only(
            save_weights_only, self.filepath
        )
        self.mode = resolve_mode(monitor, mode)
        self.save_freq = _validate_save_freq(type(self), save_freq)
        self.initial_value_threshold = initial_value_threshold
        self._reset()

    def _reset(self) -> None:
        if self.initial_value_threshold is None:
            self.best = worst_value(self.mode)
        else:
            self.best = self.initial_value_threshold
        self.step = 0
        self.epoch = 0

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        self._reset()

    def on_train_batch_end(
        self, batch_idx: int, logs: dict[str, Any] | None = None
    ) -> None:
        self.step += 1
        if isinstance(self.save_freq, int) and self.step % self.save_freq == 0:
            self._save_model(logs)

    def on_epoch_end(
        self, epoch: int, logs: dict[str, Any] | None = None
    ) -> None:
        self.epoch = epoch
        if self.save_freq == "epoch":
            self._save_model(logs)

    def _save_model(self, logs=None):
        logs = logs or {}
        filepath = Path(
            self.filepath.format(epoch=self.epoch, step=self.step, **logs)
        )
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if self.save_best_only:
            current = logs.get(self.monitor)
            if not is_valid_value(current):
                return
            if is_improvement(current, self.best, self.mode):
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
                logger.debug(
                    f"Save model at epoch {self.epoch} to {filepath}."
                )
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
