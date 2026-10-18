"""Resume an interrupted fit with `BackupAndRestore`.

A job on a cluster can stop before its fit ends: preempted, past its
time limit, or crashed. Run again, the same program restores the last
backup and goes on from the start of the epoch after it.

The example runs the training program three times as separate
processes, as a scheduler would: once killed in the sixth epoch, at
step 22 (with ``os._exit``, which skips all cleanup, as a SIGKILL
would), once again, which restores the backup of the end of the fifth
epoch and trains the last five, and once without interruption. The
resumed fit's first five epochs are those of the uninterrupted one. The
random number generators are not restored, so the last five shuffle
differently, and end close to the uninterrupted fit rather than equal.

On SLURM, submit with ``#SBATCH --requeue``: a requeued job runs the
same command, which resumes from the last backup.

python examples/callbacks/backup_and_restore.py
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from common import (
    Classifier,
    argument_parser,
    check,
    make_splits,
    prepare_output,
)
from torch.utils.data import DataLoader

from tklearn.nn import Trainer
from tklearn.nn.callbacks import BackupAndRestore, Callback

EPOCHS = 10  # of 4 batches: 40 steps
KILL_AT = 22  # in the sixth epoch
BACKUP_EPOCHS = 5  # backed up at the end of the fifth


class Kill(Callback):
    """Ends the process before the first batch after `step` steps."""

    def __init__(self, step: int) -> None:
        self.step = step

    def on_train_batch_begin(self, batch) -> None:
        if self.session.global_step == self.step:
            os._exit(1)


class CountBatches(Callback):
    count = 0

    def on_train_batch_begin(self, batch) -> None:
        self.count += 1


def train(out: Path, name: str, kill_at: int | None) -> None:
    """The training program: fit, then write the weights, the history and
    the number of batches this process trained."""
    torch.manual_seed(0)  # the same seed on every run
    train, valid, _ = make_splits()
    model = Classifier()
    count = CountBatches()
    callbacks = [BackupAndRestore(out / f"{name}-backup"), count]
    if kill_at is not None:
        callbacks.append(Kill(kill_at))
    trainer = Trainer(
        model,
        functools.partial(torch.optim.AdamW, lr=1e-3),
        lr_scheduler="linear",
        warmup=0.1,
        callbacks=callbacks,
        cpu=True,
    )
    history = trainer.fit(
        DataLoader(train, 32, shuffle=True),
        DataLoader(valid, 128),
        epochs=EPOCHS,
    )
    torch.save(model.state_dict(), out / f"{name}.pt")
    result = {"history": history, "batches": count.count}
    (out / f"{name}.json").write_text(json.dumps(result))


def run(out: Path, name: str, kill_at: int | None = None) -> int:
    """Run the training program in a new process; its exit code."""
    command = [sys.executable, __file__, "--out", str(out), "--worker", name]
    if kill_at is not None:
        command += ["--kill-at", str(kill_at)]
    return subprocess.run(command, check=False).returncode


def main(out: Path) -> None:
    print("Uninterrupted")
    check(run(out, "full") == 0, "trained all 40 steps")

    print("Killed, then run again")
    code = run(out, "resumed", kill_at=KILL_AT)
    backup = out / "resumed-backup"
    check(
        code != 0 and (backup / "state.pt").exists(),
        f"killed at step {KILL_AT}, leaving the backup of epoch "
        f"{BACKUP_EPOCHS}",
    )
    check(run(out, "resumed") == 0, "the second run finished the fit")
    full = json.loads((out / "full.json").read_text())
    resumed = json.loads((out / "resumed.json").read_text())
    left = 4 * (EPOCHS - BACKUP_EPOCHS)
    check(
        resumed["batches"] == left,
        f"it trained only the {left} batches of the last "
        f"{EPOCHS - BACKUP_EPOCHS} epochs",
    )
    done = BACKUP_EPOCHS
    check(
        len(resumed["history"]) == EPOCHS
        and resumed["history"][:done] == full["history"][:done],
        f"it returned all {EPOCHS} epochs, the first {done} those of the "
        "uninterrupted fit",
    )
    a, b = full["history"][-1], resumed["history"][-1]
    print(
        f"  last epoch: valid_loss {b['valid_loss']:.4f}, uninterrupted "
        f"{a['valid_loss']:.4f}"
    )
    check(
        abs(a["valid_loss"] - b["valid_loss"]) < 0.1 * a["valid_loss"],
        "its last validation loss is within 10% of the uninterrupted fit's",
    )
    check(not backup.exists(), "the backup was deleted once the fit ended")


if __name__ == "__main__":
    parser = argument_parser(__doc__, "backup_and_restore")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--kill-at", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        train(args.out, args.worker, args.kill_at)
    else:
        main(prepare_output(args.out))
