"""Stop training when the loss turns NaN, with `TerminateOnNaN`.

One training example has a NaN feature, as a corrupted row of a dataset
would. The batch holding it has a NaN loss, and its optimizer step turns
every weight NaN. `TerminateOnNaN` stops training after that batch; a
`ModelCheckpoint` saved every few steps keeps the last good weights.

It runs on the CPU. On Apple MPS, ReLU turns NaN into 0, so the loss
stays finite while the first layer's weights still turn NaN, and
`TerminateOnNaN` cannot see the damage.

python examples/callbacks/terminate_on_nan.py
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from accelerate import Accelerator
from common import Classifier, check, make_splits, parse_args
from torch.utils.data import DataLoader

from tklearn.nn import Trainer
from tklearn.nn.callbacks import ModelCheckpoint, TerminateOnNaN

BATCH_SIZE = 16
BAD_EXAMPLE = 100  # in batch 7, so the loss is NaN at step 7


def fit(out: Path, callbacks: list) -> tuple[Trainer, list[dict]]:
    torch.manual_seed(0)
    train, valid, _ = make_splits()
    train[BAD_EXAMPLE] = {
        "x": torch.full_like(train[BAD_EXAMPLE]["x"], math.nan),
        "labels": train[BAD_EXAMPLE]["labels"],
    }
    model = Classifier()
    trainer = Trainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        callbacks=callbacks,
        accelerator=Accelerator(cpu=True),
    )
    # not shuffled, so the bad example is always in the same batch
    history = trainer.fit(
        DataLoader(train, BATCH_SIZE), DataLoader(valid, 128), epochs=5
    )
    return trainer, history


def main(out: Path) -> None:
    bad_step = BAD_EXAMPLE // BATCH_SIZE + 1

    print("Without TerminateOnNaN")
    trainer, history = fit(out, [])
    check(
        len(history) == 5 and all(math.isnan(h["loss"]) for h in history),
        "trained all 5 epochs, with a NaN loss in every one",
    )

    print("With TerminateOnNaN")
    trainer, history = fit(
        out,
        [
            TerminateOnNaN(),
            ModelCheckpoint(out / "step-{step:02d}.pt", save_freq=2),
        ],
    )
    check(
        trainer.global_step == bad_step,
        f"stopped after step {bad_step}, the batch with the NaN",
    )
    check(
        len(history) == 1 and math.isnan(history[0]["valid_loss"]),
        "the epoch was still evaluated, with the NaN weights",
    )
    check(
        all(torch.isnan(p).all() for p in trainer.model.parameters()),
        "the model's weights are NaN",
    )
    saved = sorted(p.name for p in out.iterdir())
    last_good = Classifier()
    last_good.load_state_dict(torch.load(out / saved[-1]))
    check(
        saved == ["step-02.pt", "step-04.pt", "step-06.pt"]
        and all(torch.isfinite(p).all() for p in last_good.parameters()),
        f"{saved[-1]} keeps the last finite weights, from before step "
        f"{bad_step}",
    )


if __name__ == "__main__":
    main(parse_args(__doc__, "terminate_on_nan"))
