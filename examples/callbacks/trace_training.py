"""Trace training with OpenTelemetry into files, then read the runs back.

Two runs of a learning rate sweep are traced by `OpenTelemetryCallback`
into `FileTracerProvider`, which writes plain files that suit shared
storage such as NFS. `ModelCheckpoint` and `EarlyStopping` are listed
before the tracing callback; as it is a wrapper, their events still land
in its epoch and fit spans. The runs are then read with `load_spans` and
`load_events`, and shown as ``tklearn runs monitor`` shows them.

python examples/callbacks/trace_training.py

It takes a few seconds. To follow the runs live, slow it down with a
pause after each training batch, and watch from another terminal:

python examples/callbacks/trace_training.py --delay 0.5
python -m tklearn runs monitor examples/outputs/callbacks/trace_training -n 1
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from common import (
    Classifier,
    argument_parser,
    check,
    make_loaders,
    prepare_output,
)

from tklearn.metrics import F1, Accuracy
from tklearn.nn import Trainer
from tklearn.nn.callbacks import (
    EarlyStopping,
    LambdaCallback,
    ModelCheckpoint,
    OpenTelemetryCallback,
)
from tklearn.tracing import FileTracerProvider, load_events, load_spans
from tklearn.tracing.monitor import snapshot

LRS = [1e-3, 1e-2]
MAX_EPOCHS = 3000
LOG_EVERY_N_STEPS = 2


def train(out: Path, lr: float, delay: float) -> tuple[list[dict], dict]:
    """Train one run of the sweep, pausing `delay` seconds after each
    batch; its history and test results."""
    torch.manual_seed(0)
    train_loader, valid_loader, test_loader = make_loaders()
    model = Classifier()
    # one run per directory under out; the resource holds its settings
    provider = FileTracerProvider(
        out, name=f"sweep/lr={lr:g}", resource={"lr": lr}
    )
    trainer = Trainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=lr),
        metrics={"accuracy": Accuracy(), "f1": F1(average="macro")},
        callbacks=[
            ModelCheckpoint(
                out / "checkpoints" / f"lr={lr:g}.pt",
                monitor="valid_loss",
                save_best_only=True,
            ),
            EarlyStopping("valid_loss", patience=3, restore_best_weights=True),
            # a stand-in for a slower model, to follow the run live
            LambdaCallback(
                on_train_batch_end=lambda trainer, batch, logs: time.sleep(
                    delay
                )
            ),
            # last in the list, but it encloses the callbacks above
            OpenTelemetryCallback(
                tracer_provider=provider, log_every_n_steps=LOG_EVERY_N_STEPS
            ),
        ],
    )
    history = trainer.fit(train_loader, valid_loader, epochs=MAX_EPOCHS)
    # outside fit, evaluate is a span of its own
    test_results = trainer.evaluate(test_loader, prefix="test_")
    return history, test_results


def main(out: Path, delay: float = 0.0) -> None:
    results = {lr: train(out, lr, delay) for lr in LRS}

    epochs = load_spans(out, "epoch")
    summary = epochs.groupby("resource.lr").agg(
        epochs=("epoch", "size"),
        best_valid_loss=("valid_loss", "min"),
        best_valid_f1=("valid_f1", "max"),
    )
    tests = load_spans(out, "evaluate")
    tests = tests[tests["parent_id"].isna()].set_index("resource.lr")
    summary["test_f1"] = tests["test_f1"]
    print(f"\nRuns in {out}:\n{summary.round(4)}\n")

    for lr, (history, test_results) in results.items():
        run = f"sweep/lr={lr:g}"
        print(run)
        spans = load_spans(out)
        spans = spans[spans["run"] == run]
        (fit,) = spans[spans["name"] == "fit"].itertuples()
        # float in the table, as other spans have no step
        fit_steps = int(fit.step)
        run_epochs = spans[spans["name"] == "epoch"]
        steps = spans[spans["name"] == "steps"]
        check(
            fit.parent_id is None
            and len(run_epochs) == len(history)
            and (run_epochs["parent_id"] == fit.span_id).all()
            and (steps["parent_id"] == fit.span_id).all(),
            f"a fit span holding {len(run_epochs)} epoch and {len(steps)} "
            "steps spans",
        )
        check(
            all(
                math.isclose(row[key], logs[key], rel_tol=1e-9)
                for row, logs in zip(run_epochs.to_dict("records"), history)
                for key in ("loss", "valid_loss", "valid_f1")
            ),
            "the epoch spans hold Trainer.history",
        )
        check(
            len(steps) == fit_steps // LOG_EVERY_N_STEPS,
            f"a steps span every {LOG_EVERY_N_STEPS} of {fit_steps} steps",
        )
        evaluates = spans[spans["name"] == "evaluate"]
        test = evaluates[evaluates["parent_id"].isna()].iloc[0]
        check(
            len(evaluates) == len(history) + 1
            and all(math.isclose(test[k], v) for k, v in test_results.items()),
            "an evaluate span per epoch, and one for the test set",
        )

        events = load_events(out)
        events = events[events["run"] == run]
        losses = [logs["valid_loss"] for logs in history]
        improvements = sum(
            loss < min(losses[:i], default=math.inf)
            for i, loss in enumerate(losses)
        )
        checkpoints = events[events["name"] == "checkpoint"]
        check(
            len(checkpoints) == improvements
            and (checkpoints["span"] == "epoch").all(),
            f"{improvements} checkpoint event{'s' * (improvements != 1)}, "
            "on the epochs that improved",
        )
        stopped = len(history) < MAX_EPOCHS
        names = set(events["name"])
        check(
            ({"early_stopping", "restore_best_weights"} <= names) == stopped,
            "early stopping and the weights it restored are events"
            if stopped
            else "no early stopping",
        )

    print("Monitor")
    statuses = {s.run: s for s in snapshot(out)}
    for lr, (history, _) in results.items():
        status = statuses[f"sweep/lr={lr:g}"]
        expected = "stopped" if len(history) < MAX_EPOCHS else "finished"
        check(
            status.status == expected
            and status.epoch == len(history) - 1
            and status.epochs == MAX_EPOCHS,
            f"lr={lr:g} is {status.status} after epoch {status.epoch + 1}",
        )
    print(flush=True)  # before the other process writes
    # what `tklearn runs monitor` prints, from another process
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tklearn",
            "runs",
            "monitor",
            str(out),
            "--once",
        ],
        check=True,
        env={**os.environ, "COLUMNS": "130"},
    )


if __name__ == "__main__":
    parser = argument_parser(__doc__, "trace_training")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="pause after each training batch, to follow the runs live "
        "(default: %(default)s)",
    )
    args = parser.parse_args()
    main(prepare_output(args.out), args.delay)
