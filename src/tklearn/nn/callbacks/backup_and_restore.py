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
    model, optimizer, scheduler, progress, history and random number
    generators) to `backup_dir`. When a `fit` starts and finds a backup
    there, such as when a job that was preempted, timed out or crashed
    runs the same program again, it restores it, and the fit goes on from
    where the backup was taken. A backup taken during an epoch resumes it
    after the batches already done, which the dataloader yields in the
    same order as before when the program seeds torch the same way (see
    `Trainer.create_accelerator`). `fit` returns the logs of every epoch,
    including those before the interruption.

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
        Back up after every epoch, or every this many optimizer steps.
    double_checkpoint : bool, default=False
        Keep the previous backup too, and restore it when the last one
        cannot be read, at the cost of twice the space.
    delete_checkpoint : bool, default=True
        Delete the backup when the fit completes. With False, a later
        `fit` with this callback restores the end of this one, so it has
        nothing left to train.

    Notes
    -----
    The main process writes the shared state, read from `Trainer.model`
    as by `ModelCheckpoint` (so not with FSDP or DeepSpeed ZeRO-3), and
    every process its own random number generators and running means.
    Each file is written to a temporary file and renamed, so that an
    interruption while saving leaves the previous backup whole. Files are
    loaded with ``torch.load(weights_only=False)``: back up only to
    directories that others cannot write.

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
                "batch": state["batch"],
                "step": state["global_step"],
            }
            trace.get_current_span().add_event("restore", event)
            logger.info(
                f"Restored the fit from {self.backup_dir}: epoch "
                f"{state['epoch'] + 1}, after batch {state['batch']}, step "
                f"{state['global_step']}"
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
        accelerator = self.session.accelerator
        # every process is done writing its files
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            shutil.rmtree(self.backup_dir, ignore_errors=True)

    # --- files -------------------------------------------------------------

    def _paths(self, previous: bool = False) -> tuple[Path, Path]:
        """The shared file and this process's file of the last backup, or
        of the previous one."""
        suffix = ".prev.pt" if previous else ".pt"
        index = self.session.accelerator.process_index
        return (
            self.backup_dir / f"state{suffix}",
            self.backup_dir / f"local-{index}{suffix}",
        )

    def _save(self) -> None:
        state = self.session.state_dict()
        local = state.pop("local")
        # the position, to check that the files of a backup go together
        local["position"] = _position(state)
        shared_path, local_path = self._paths()
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        if self.session.accelerator.is_main_process:
            self._write(state, shared_path)
            trace.get_current_span().add_event(
                "backup",
                {"path": str(self.backup_dir), "step": state["global_step"]},
            )
        self._write(local, local_path)

    def _write(self, data: Any, path: Path) -> None:
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        torch.save(data, tmp)
        if self.double_checkpoint and path.exists():
            os.replace(
                path, path.with_name(path.name[: -len(".pt")] + ".prev.pt")
            )
        os.replace(tmp, path)

    def _load(self) -> dict[str, Any] | None:
        """The last backup that can be read, with this process's local
        state when it matches; None without one."""
        candidates = [False, True] if self.double_checkpoint else [False]
        for previous in candidates:
            shared_path, _ = self._paths(previous)
            if not shared_path.exists():
                continue
            try:
                state = torch.load(
                    shared_path, map_location="cpu", weights_only=False
                )
            except Exception:
                logger.exception(f"cannot read the backup {shared_path}")
                continue
            state["local"] = self._load_local(state)
            return state
        return None

    def _load_local(self, state: Mapping[str, Any]) -> dict[str, Any] | None:
        # the local file of the same backup: the last one, or with a double
        # checkpoint, the previous one when an interruption came between
        # the writes of the two
        for previous in (False, True):
            _, path = self._paths(previous)
            if not path.exists():
                continue
            try:
                local = torch.load(
                    path, map_location="cpu", weights_only=False
                )
            except Exception:
                logger.exception(f"cannot read the backup {path}")
                continue
            if local.get("position") == _position(state):
                return local
        logger.warning(
            f"{self.backup_dir} has no random number generator state of "
            f"process {self.session.accelerator.process_index} matching "
            "the backup; resuming without it"
        )
        return None


def _position(state: Mapping[str, Any]) -> tuple[int, int, int]:
    return (state["epoch"], state["batch"], state["global_step"])
