from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Union

import torch
from opentelemetry import trace

from tklearn.logging import get_logger
from tklearn.nn.callbacks.base import Callback

__all__ = [
    "BackupAndRestore",
]

logger = get_logger(__name__)


class BackupAndRestore(Callback):
    """Back up the state of `fit`, and restore it when an interrupted fit
    runs again.

    The callback saves the state of the fit (`Session.state_dict`: the
    model, optimizer, scheduler, gradient scaler, progress and history)
    to `backup_dir`. When a `fit` starts and finds a backup there, such
    as when a job that was preempted, timed out or crashed runs the same
    program again, it restores it, and the fit goes on from the start of
    the epoch the backup was taken in, or after it for a backup at its
    end. `fit` returns the logs of every epoch, including those before
    the interruption.

    As in Keras, a backup taken during an epoch (with an int
    `save_freq`) holds the weights and steps of when it was taken, and
    the restored fit trains that epoch again from its first batch: the
    batches before the backup are trained twice, and the fit takes that
    many more optimizer steps than planned, which a learning rate
    schedule spends past its end (a linear one stays at 0, a cosine one
    rises again). The random number generators are not restored, so the
    resumed fit shuffles and drops out differently from the interrupted
    one, and ends close to it rather than equal.

    As in Keras, the backup is deleted when the fit completes, including
    when a callback stops it early, but not when it raises. The other
    callbacks are not restored: e.g. `EarlyStopping` counts its patience
    and `ModelCheckpoint` its best value from the restart, and a
    `CSVLogger` needs ``append=True`` to keep the rows written before it.

    Parameters
    ----------
    backup_dir : str or path-like
        Directory of the backup, for this callback alone: not shared with
        another fit or callback, as it is deleted at the end.
    save_freq : "epoch" or int, default="epoch"
        Back up after every epoch, or every this many optimizer steps,
        for epochs long enough that losing one costs more than training
        part of it twice.
    double_checkpoint : bool, default=False
        Keep the previous backup too, and restore it when the last one
        cannot be read, at the cost of twice the space.
    delete_checkpoint : bool, default=True
        Delete the backup when the fit completes. With False, a later
        `fit` with this callback restores the end of this one, so it has
        nothing left to train.

    Notes
    -----
    The main process writes the backup, read from `Trainer.model` as by
    `ModelCheckpoint` (so not with FSDP or DeepSpeed ZeRO-3), and every
    process restores it, on any device and number of processes. It is
    written to a temporary file and renamed, so that an interruption
    while saving leaves the previous backup whole. It is loaded with
    ``torch.load(weights_only=False)``: back up only to directories that
    others cannot write.

    Examples
    --------
    >>> backup = BackupAndRestore("runs/42/backup", save_freq=500)
    >>> trainer = Trainer(model, optimizer, callbacks=[backup])
    >>> trainer.fit(train_loader, valid_loader, epochs=10)
    """

    # restores before the other callbacks begin, and backs up after they
    # end the batch or epoch, e.g. once EarlyStopping decided to stop
    wrapper = True

    def __init__(
        self,
        backup_dir: str | os.PathLike[str],
        save_freq: Union[Literal["epoch"], int] = "epoch",
        double_checkpoint: bool = False,
        delete_checkpoint: bool = True,
    ) -> None:
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
        self.backup_dir = Path(backup_dir)
        self.save_freq = save_freq
        self.double_checkpoint = double_checkpoint
        self.delete_checkpoint = delete_checkpoint

    def on_train_begin(self) -> None:
        self._saved_step = self.session.global_step
        accelerator = self.session.accelerator
        state = self._load()
        if state is None:
            return
        self.session.load_state_dict(state)
        self._saved_step = self.session.global_step
        if accelerator.is_main_process:
            event = {
                "path": str(self.backup_dir),
                "epoch": state["epoch"],
                "step": state["global_step"],
            }
            trace.get_current_span().add_event("restore", event)
            logger.info(
                f"Restored the fit from {self.backup_dir} at step "
                f"{state['global_step']}; resuming at the start of epoch "
                f"{state['epoch'] + 1}"
            )

    def on_train_batch_end(
        self, batch: Any, logs: Mapping[str, float]
    ) -> None:
        # global_step only advances on batches that step the optimizer, and
        # stays put when fp16 skips the step, so back up each step once
        step = self.session.global_step
        if (
            self.save_freq != "epoch"
            and step != self._saved_step
            and step % self.save_freq == 0
        ):
            self._saved_step = step
            self._save()

    def on_epoch_end(self, logs: dict[str, Any]) -> None:
        if self.save_freq == "epoch":
            self._save()

    def on_train_end(self) -> None:
        if not self.delete_checkpoint:
            return
        if self.session.accelerator.is_main_process:
            shutil.rmtree(self.backup_dir, ignore_errors=True)

    # --- files -------------------------------------------------------------

    def _path(self, previous: bool = False) -> Path:
        """The file of the last backup, or of the previous one."""
        return self.backup_dir / ("state.prev.pt" if previous else "state.pt")

    def _save(self) -> None:
        if not self.session.accelerator.is_main_process:
            return
        state = self.session.state_dict()
        path = self._path()
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        torch.save(state, tmp)
        if self.double_checkpoint and path.exists():
            os.replace(path, self._path(previous=True))
        os.replace(tmp, path)
        trace.get_current_span().add_event(
            "backup",
            {"path": str(self.backup_dir), "step": state["global_step"]},
        )

    def _load(self) -> dict[str, Any] | None:
        """The last backup that can be read; None without one."""
        candidates = [False, True] if self.double_checkpoint else [False]
        for previous in candidates:
            path = self._path(previous)
            if not path.exists():
                continue
            try:
                return torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                logger.exception(f"cannot read the backup {path}")
        return None
