"""Write callbacks: subclass `Callback`, or pass functions to
`LambdaCallback`.

1. Which hooks run, and in which order, in `fit`, `evaluate` and
   `predict`, with gradient accumulation.
2. `FreezeEncoder`, a `Callback` subclass, trains only the head for the
   first epochs.
3. `StopAfterSteps` stops training in the middle of an epoch by setting
   ``trainer.should_stop``.

python examples/callbacks/custom_callbacks.py
"""

from __future__ import annotations

from pathlib import Path

import torch
from common import Classifier, check, make_loaders, parse_args

from tklearn.nn import Trainer
from tklearn.nn.callbacks import Callback, LambdaCallback

# every hook a callback can implement
HOOKS = [name for name in vars(Callback) if name.startswith("on_")]


class FreezeEncoder(Callback):
    """Train only the head of a `Classifier` for the first `epochs` epochs.

    Frozen parameters get no gradients, so the optimizer leaves them
    alone. (With several processes, DistributedDataParallel needs
    ``find_unused_parameters=True`` for this.)
    """

    def __init__(self, epochs: int) -> None:
        self.epochs = epochs

    def on_epoch_begin(self, trainer: Trainer) -> None:
        trainable = trainer.epoch >= self.epochs
        for param in trainer.model.encoder.parameters():
            param.requires_grad_(trainable)

    def on_train_end(self, trainer: Trainer) -> None:
        for param in trainer.model.encoder.parameters():
            param.requires_grad_(True)


class StopAfterSteps(Callback):
    """Stop training after `max_steps` optimizer steps."""

    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps

    def on_train_batch_end(self, trainer, batch, logs) -> None:
        if trainer.global_step >= self.max_steps:
            # fit ends after this batch, once the epoch is evaluated
            trainer.should_stop = True


def hook_order() -> None:
    print("1. Hook order")
    calls: list[str] = []
    # a function per hook, recording its name; name=name binds each name
    recorder = LambdaCallback(**{
        name: lambda *args, name=name: calls.append(name) for name in HOOKS
    })
    # 96 training examples: 3 batches of 32; 256 for validation: 2 of 128
    train, valid, test = make_loaders(sizes=(96, 256, 256))
    model = Classifier()
    trainer = Trainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        callbacks=[recorder],
        gradient_accumulation_steps=2,
    )
    trainer.fit(train, valid, epochs=2)

    test_hooks = (
        ["on_test_begin"]
        + ["on_test_batch_begin", "on_test_batch_end"] * 2
        + ["on_test_end"]
    )
    # the optimizer steps after 2 batches, and after the last batch
    epoch_hooks = (
        ["on_epoch_begin"]
        + ["on_train_batch_begin", "on_train_batch_end"]
        + ["on_train_batch_begin", "on_before_optimizer_step"]
        + ["on_train_batch_end"]
        + ["on_train_batch_begin", "on_before_optimizer_step"]
        + ["on_train_batch_end"]
        + test_hooks
        + ["on_epoch_end"]
    )
    expected = ["on_train_begin", *epoch_hooks * 2, "on_train_end"]
    check(calls == expected, f"fit ran {len(calls)} hooks in order")
    check(
        trainer.global_step == calls.count("on_before_optimizer_step") == 4,
        "the optimizer stepped twice per epoch of 3 batches",
    )

    calls.clear()
    trainer.evaluate(valid)
    check(calls == test_hooks, "evaluate ran the on_test_* hooks")

    calls.clear()
    trainer.predict(test)
    expected = (
        ["on_predict_begin"]
        + ["on_predict_batch_begin", "on_predict_batch_end"] * 2
        + ["on_predict_end"]
    )
    check(calls == expected, "predict ran the on_predict_* hooks")


def freeze_encoder() -> None:
    print("2. FreezeEncoder")
    torch.manual_seed(0)
    train, valid, _ = make_loaders()
    model = Classifier()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    # the weights after each epoch
    snapshots: list[dict[str, torch.Tensor]] = []
    trainer = Trainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        callbacks=[
            FreezeEncoder(epochs=2),
            LambdaCallback(
                on_epoch_end=lambda trainer, logs: snapshots.append({
                    k: v.detach().cpu().clone()
                    for k, v in trainer.model.state_dict().items()
                })
            ),
        ],
    )
    trainer.fit(train, valid, epochs=3)

    def changed(epoch: int, key: str) -> bool:
        return not torch.equal(snapshots[epoch][key], initial[key])

    check(
        not changed(0, "encoder.0.weight")
        and not changed(1, "encoder.0.weight"),
        "the encoder was frozen in epochs 1 and 2",
    )
    check(changed(2, "encoder.0.weight"), "the encoder trained in epoch 3")
    check(
        changed(0, "head.weight")
        and not torch.equal(
            snapshots[0]["head.weight"], snapshots[1]["head.weight"]
        ),
        "the head trained from epoch 1",
    )
    check(
        all(p.requires_grad for p in model.parameters()),
        "the encoder is trainable again after fit",
    )


def stop_after_steps() -> None:
    print("3. StopAfterSteps")
    train, valid, _ = make_loaders(sizes=(96, 256, 256))
    model = Classifier()
    losses: list[float] = []
    trainer = Trainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        callbacks=[
            StopAfterSteps(7),
            LambdaCallback(
                on_train_batch_end=lambda trainer, batch, logs: losses.append(
                    logs["loss"]
                )
            ),
        ],
    )
    history = trainer.fit(train, valid, epochs=10)
    check(
        trainer.global_step == len(losses) == 7,
        "stopped after step 7, in the middle of epoch 3",
    )
    check(
        len(history) == 3 and "valid_loss" in history[-1],
        "the interrupted epoch was still evaluated and logged",
    )
    check(
        abs(history[-1]["loss"] - losses[-1]) < 1e-6,
        "its training loss is the mean of the batches it trained",
    )


def main(out: Path) -> None:
    hook_order()
    freeze_encoder()
    stop_after_steps()


if __name__ == "__main__":
    main(parse_args(__doc__, "custom_callbacks"))
