from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union

import torch
from opentelemetry import trace
from safetensors.torch import save_model

from tklearn.logging import get_logger
from tklearn.nn.callbacks._monitor import Mode, MonitorCallback

if TYPE_CHECKING:
    from tklearn.nn.trainer import Trainer

__all__ = [
    "ModelCheckpoint",
]

logger = get_logger(__name__)

_SUFFIXES = (".safetensors", ".pt", ".pth")


class ModelCheckpoint(MonitorCallback):
    """Save the model weights during training.

    Parameters
    ----------
    filepath : str or path-like
        Where to save. The suffix picks the format: ``.safetensors`` (load
        with `safetensors.torch.load_model`), or ``.pt`` and ``.pth`` (a
        `torch.save` state dict). It is formatted with ``epoch``
        (one-based), ``step`` (`Trainer.global_step`) and the logs, e.g.
        ``"ckpt/{epoch:02d}-{valid_loss:.3f}.safetensors"``; without any
        field, each save overwrites the last.
    monitor : str, default="valid_loss"
        Key of the value compared when `save_best_only` is set.
    save_best_only : bool, default=False
        Save only when the monitored value improves on the best seen by
        this callback, across `fit` calls.
    mode : {"auto", "min", "max"}, default="auto"
        Whether lower or higher values are better; ``"auto"`` infers it
        from the key.
    save_freq : "epoch" or int, default="epoch"
        Save after every epoch, or every this many optimizer steps. With
        steps, `monitor` is read from the training batch logs.
    initial_value_threshold : float, optional
        Value the monitored value must improve on for the first save when
        `save_best_only` is set.
    verbose : bool, default=False
        Log every save.

    Attributes
    ----------
    best : float
        Best monitored value seen so far.

    Notes
    -----
    Only the main process writes. The weights are read from
    `Trainer.model`, which holds all of them on every process with plain
    data parallelism, but not with FSDP or DeepSpeed ZeRO-3.

    Examples
    --------
    >>> checkpoint = ModelCheckpoint(
    ...     "ckpt/best.safetensors", monitor="valid_f1", save_best_only=True
    ... )
    >>> trainer = Trainer(model, optimizer, callbacks=[checkpoint])
    >>> trainer.fit(train_loader, valid_loader, epochs=10)
    >>> safetensors.torch.load_model(model, "ckpt/best.safetensors")
    """

    def __init__(
        self,
        filepath: str | os.PathLike[str],
        monitor: str = "valid_loss",
        *,
        save_best_only: bool = False,
        mode: Mode = "auto",
        save_freq: Union[Literal["epoch"], int] = "epoch",
        initial_value_threshold: float | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(monitor, mode, 0.0)
        filepath = os.fspath(filepath)
        if not filepath.endswith(_SUFFIXES):
            msg = (
                f"filepath must end with one of {', '.join(_SUFFIXES)}, "
                f"got {filepath!r}"
            )
            raise ValueError(msg)
        if save_freq != "epoch" and (
            isinstance(save_freq, bool)
            or not isinstance(save_freq, int)
            or save_freq < 1
        ):
            msg = (
                "save_freq must be 'epoch' or a positive number of steps, "
                f"got {save_freq!r}"
            )
            raise ValueError(msg)
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.verbose = verbose
        if initial_value_threshold is None:
            self.best = self._initial_best()
        else:
            self.best = float(initial_value_threshold)

    def on_train_batch_end(
        self, trainer: Trainer, batch: Any, logs: dict[str, float]
    ) -> None:
        # global_step only advances on batches that step the optimizer
        if (
            self.save_freq != "epoch"
            and trainer.accelerator.sync_gradients
            and trainer.global_step % self.save_freq == 0
        ):
            self._save(trainer, logs)

    def on_epoch_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:
        if self.save_freq == "epoch":
            self._save(trainer, logs)

    def _save(self, trainer: Trainer, logs: dict[str, Any]) -> None:
        if self.save_best_only:
            current = self._get_monitor_value(logs)
            if current is None or not self._is_improvement(current, self.best):
                return
            self.best = current
        path = self._format_path(trainer, logs)
        if not trainer.accelerator.is_main_process:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".safetensors":
            # unlike save_file, save_model handles tied weights
            save_model(trainer.model, path)
        else:
            torch.save(trainer.model.state_dict(), path)
        event = {"path": str(path), "step": trainer.global_step}
        if self.save_best_only:
            event[self.monitor] = self.best
        trace.get_current_span().add_event("checkpoint", event)
        if self.verbose:
            logger.info(
                f"Step {trainer.global_step}: saved the model to {path}"
            )

    def _format_path(self, trainer: Trainer, logs: dict[str, Any]) -> Path:
        fields = {
            **logs,
            "epoch": trainer.epoch + 1,
            "step": trainer.global_step,
        }
        try:
            return Path(self.filepath.format_map(fields))
        except KeyError as e:
            msg = (
                f"cannot format the checkpoint filepath {self.filepath!r}: "
                f"{e} is not in the logs, which have {', '.join(logs)}"
            )
            raise KeyError(msg) from None
