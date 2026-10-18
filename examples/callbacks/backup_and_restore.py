"""Resume an interrupted fit with `BackupAndRestore`.

A job on a cluster can stop before its fit ends: preempted, past its
time limit, or crashed. Run again, the same program restores the last
backup and goes on from there, as if it had not stopped.

The example runs the training program three times as separate
processes, as a scheduler would: once killed at step 21 (with
``os._exit``, which skips all cleanup, as a SIGKILL would), once again,
which resumes from the backup of step 18, after the second batch of the
fifth epoch, and once without interruption. The resumed fit trains only the batches
left, and ends with the same weights and history as the uninterrupted
one.

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
SAVE_FREQ = 6
KILL_AT = 21
BACKUP_STEP = 18


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
    callbacks = [BackupAndRestore(out / f"{name}-backup", SAVE_FREQ), count]
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
        f"killed at step {KILL_AT}, leaving the backup of step {BACKUP_STEP}",
    )
    check(run(out, "resumed") == 0, "the second run finished the fit")
    full = json.loads((out / "full.json").read_text())
    resumed = json.loads((out / "resumed.json").read_text())
    check(
        resumed["batches"] == 40 - BACKUP_STEP,
        f"it trained only the {40 - BACKUP_STEP} batches after the backup",
    )
    a, b = torch.load(out / "full.pt"), torch.load(out / "resumed.pt")
    check(
        all(torch.equal(a[k], b[k]) for k in a),
        "its weights equal the uninterrupted fit's",
    )
    check(
        resumed["history"] == full["history"],
        "its history, all 10 epochs, equals the uninterrupted fit's",
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
