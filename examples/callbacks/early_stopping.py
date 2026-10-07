"""Stop an overfitting model early, keep its best weights and log epochs.

A large model on 128 noisy examples overfits within a few epochs, so the
validation loss turns upward. The built-in callbacks react to that:

- `ReduceLROnPlateau` halves the learning rate when it stalls;
- `EarlyStopping` stops training and restores the best weights;
- `ModelCheckpoint` saves the best weights, and every epoch's weights;
- `CSVLogger` writes the epoch logs and `ProgbarLogger` shows progress.

python examples/callbacks/early_stopping.py
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import torch
from common import Classifier, check, make_loaders, parse_args
from safetensors.torch import load_model

from tklearn.metrics import Accuracy
from tklearn.nn import Trainer
from tklearn.nn.callbacks import (
    CSVLogger,
    EarlyStopping,
    LambdaCallback,
    ModelCheckpoint,
    ProgbarLogger,
    ReduceLROnPlateau,
)

MAX_EPOCHS = 100
LR = 1e-3
PATIENCE = 4
LR_PATIENCE = 2
LR_FACTOR = 0.5


def main(out: Path) -> None:
    torch.manual_seed(0)
    train, valid, _ = make_loaders()
    model = Classifier()

    early_stopping = EarlyStopping(
        "valid_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=True,
    )
    best_checkpoint = ModelCheckpoint(
        out / "best.safetensors",
        monitor="valid_loss",
        save_best_only=True,
    )
    # the file name is formatted with the epoch and its logs
    epoch_checkpoints = ModelCheckpoint(
        out / "epochs" / "{epoch:02d}-{valid_loss:.4f}.pt"
    )
    reduce_lr = ReduceLROnPlateau(
        "valid_loss", factor=LR_FACTOR, patience=LR_PATIENCE, verbose=True
    )
    # the learning rate each epoch trains with
    lrs = []
    record_lr = LambdaCallback(
        on_epoch_begin=lambda trainer: lrs.append(
            trainer.optimizer.param_groups[0]["lr"]
        )
    )
    trainer = Trainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=LR),
        metrics={"accuracy": Accuracy()},
        callbacks=[
            ProgbarLogger(),
            CSVLogger(out / "history.csv"),
            record_lr,
            reduce_lr,
            best_checkpoint,
            epoch_checkpoints,
            early_stopping,
        ],
    )
    history = trainer.fit(train, valid, epochs=MAX_EPOCHS)

    valid_losses = [logs["valid_loss"] for logs in history]
    best_epoch = min(range(len(history)), key=valid_losses.__getitem__)
    best_loss = valid_losses[best_epoch]
    print(
        f"\nTrained {len(history)} of {MAX_EPOCHS} epochs; best epoch "
        f"{best_epoch + 1} with valid_loss={best_loss:.4f}, final "
        f"valid_loss={valid_losses[-1]:.4f}\n"
    )

    print("EarlyStopping")
    check(
        early_stopping.stopped_epoch == len(history) - 1 < MAX_EPOCHS - 1,
        f"stopped after epoch {len(history)}, before epoch {MAX_EPOCHS}",
    )
    check(
        early_stopping.best_epoch == best_epoch
        and early_stopping.best == best_loss,
        "found the epoch with the lowest valid_loss",
    )
    check(
        len(history) - 1 - best_epoch == PATIENCE,
        f"waited {PATIENCE} epochs without improvement",
    )
    restored_loss = trainer.evaluate(valid)["loss"]
    check(
        math.isclose(restored_loss, best_loss, rel_tol=1e-5),
        f"restored the best weights (valid_loss {restored_loss:.4f})",
    )

    print("ModelCheckpoint")
    fresh = Classifier()
    load_model(fresh, out / "best.safetensors")
    checkpoint_loss = Trainer(fresh).evaluate(valid)["loss"]
    check(
        math.isclose(checkpoint_loss, best_loss, rel_tol=1e-5),
        f"best.safetensors holds the best weights "
        f"(valid_loss {checkpoint_loss:.4f})",
    )
    names = sorted(p.name for p in (out / "epochs").iterdir())
    expected = [
        f"{epoch + 1:02d}-{logs['valid_loss']:.4f}.pt"
        for epoch, logs in enumerate(history)
    ]
    check(names == expected, f"saved one file per epoch ({names[0]}, ...)")
    last = Classifier()
    last.load_state_dict(torch.load(out / "epochs" / names[-1]))
    check(
        not torch.equal(last.head.weight, fresh.head.weight.cpu()),
        "the last epoch's weights differ from the best",
    )

    print("ReduceLROnPlateau")
    expected_lrs = _replay_reduce_lr(valid_losses)
    check(
        all(math.isclose(a, b) for a, b in zip(lrs, expected_lrs)),
        f"learning rates follow the plateaus: {_format_lrs(lrs)}",
    )
    check(min(lrs) < LR, "reduced the learning rate at least once")

    print("CSVLogger")
    csv = pd.read_csv(out / "history.csv")
    check(
        csv["epoch"].tolist() == list(range(len(history))),
        f"wrote one row per epoch, with columns {', '.join(csv.columns)}",
    )
    check(
        all(
            math.isclose(csv[key][i], logs[key], rel_tol=1e-9)
            for i, logs in enumerate(history)
            for key in logs
        ),
        "the rows hold Trainer.history",
    )


def _replay_reduce_lr(valid_losses: list[float]) -> list[float]:
    """The learning rate of each epoch, by the rule of ReduceLROnPlateau
    (min_delta=1e-4, no cooldown)."""
    lr, best, wait = LR, math.inf, 0
    lrs = [lr]
    for loss in valid_losses[:-1]:
        if loss + 1e-4 < best:
            best, wait = loss, 0
        else:
            wait += 1
            if wait >= LR_PATIENCE:
                lr, wait = lr * LR_FACTOR, 0
        lrs.append(lr)
    return lrs


def _format_lrs(lrs: list[float]) -> str:
    # runs of equal learning rates, e.g. "0.01 x5, 0.005 x3"
    runs: list[list] = []
    for lr in lrs:
        if runs and runs[-1][0] == lr:
            runs[-1][1] += 1
        else:
            runs.append([lr, 1])
    return ", ".join(f"{lr:.3g} x{n}" for lr, n in runs)


if __name__ == "__main__":
    main(parse_args(__doc__, "early_stopping"))
