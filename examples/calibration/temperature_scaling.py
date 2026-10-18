"""Calibrate an overconfident classifier with temperature scaling.

A model trained to fit its training set closely is overconfident: its
probabilities are higher than its accuracy. Temperature scaling (Guo et
al., 2017, "On Calibration of Modern Neural Networks") divides the
logits by one number, the temperature, fit on held-out data to minimize
the negative log-likelihood. The argmax does not change, so neither do
the predictions; only the probabilities do.

The example trains a classifier on a small noisy training set, then
`tklearn.nn.calibration.calibrate` fits the temperature on the
validation set's logits, from `Trainer.predict`, and returns the
calibrated model: a `CalibratedModule` wrapping the trained one. Both
models are evaluated on the test set through `Trainer`, with the
calibration metrics of `tklearn.metrics`.

python examples/calibration/temperature_scaling.py
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tklearn.metrics import (
    Accuracy,
    BrierScore,
    ExpectedCalibrationError,
    LogLoss,
)
from tklearn.nn import Module, Trainer
from tklearn.nn.calibration import calibrate

NUM_FEATURES = 16
NUM_CLASSES = 3


class Classifier(Module):
    """A two-layer perceptron whose `predict_step` also returns the
    logits, for calibration."""

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(NUM_FEATURES, hidden), torch.nn.ReLU()
        )
        self.head = torch.nn.Linear(hidden, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    def predict_step(self, batch):
        logits = self(batch["x"])
        outputs = {
            "logits": logits,  # what calibration scales
            "y_pred": logits.argmax(-1),
            "y_score": logits.softmax(-1),
        }
        if "labels" in batch:  # absent at inference time
            outputs["y_true"] = batch["labels"]
            outputs["loss"] = F.cross_entropy(logits, batch["labels"])
        return outputs


def make_splits(
    sizes: tuple[int, ...] = (256, 1024, 2048),
    *,
    label_noise: float = 0.2,
    seed: int = 0,
) -> list[list[dict[str, torch.Tensor]]]:
    """Splits of a noisy linear 3-class task, as lists of examples.

    Each label is the class of the highest of three random linear
    scores, replaced by a random class for `label_noise` of them, so a
    large model overfits the small training split.
    """
    generator = torch.Generator().manual_seed(seed)
    n = sum(sizes)
    x = torch.randn(n, NUM_FEATURES, generator=generator)
    w = torch.randn(NUM_FEATURES, NUM_CLASSES, generator=generator)
    labels = (x @ w).argmax(-1)
    noise = torch.rand(n, generator=generator) < label_noise
    random = torch.randint(NUM_CLASSES, (n,), generator=generator)
    labels = torch.where(noise, random, labels)
    examples = [{"x": x[i], "labels": labels[i]} for i in range(n)]
    splits, start = [], 0
    for size in sizes:
        splits.append(examples[start : start + size])
        start += size
    return splits


def check(condition: bool, message: str) -> None:
    """Print a passed check, or fail the example."""
    if not condition:
        raise AssertionError(f"FAILED: {message}")
    print(f"  ok  {message}")


def main() -> None:
    torch.manual_seed(0)
    train, valid, test = make_splits()
    generator = torch.Generator().manual_seed(0)
    train = DataLoader(train, 32, shuffle=True, generator=generator)
    valid, test = DataLoader(valid, 256), DataLoader(test, 256)
    metrics = {
        "accuracy": Accuracy(),
        "nll": LogLoss(),
        "brier": BrierScore(),
        "ece": ExpectedCalibrationError(n_bins=10),
    }

    model = Classifier()
    trainer = Trainer(
        model,
        functools.partial(torch.optim.AdamW, lr=1e-3),
        metrics=metrics,
        cpu=True,
    )
    trainer.fit(train, epochs=100)

    # on validation data: the training data's logits are the ones the
    # model was fit to, so they would make it look even more confident
    model = calibrate(trainer, valid, activation="softmax")
    temperature = model.temperature.item()
    calibrated = Trainer(model, metrics=metrics, cpu=True)

    before, after = trainer.evaluate(test), calibrated.evaluate(test)
    del before["loss"], after["loss"]  # the NLL, as LogLoss
    print(f"temperature: {temperature:.2f}\n")
    print(f"{'test':<10} {'before':>8} {'after':>8}")
    for name in before:
        print(f"{name:<10} {before[name]:>8.4f} {after[name]:>8.4f}")
    print()

    check(temperature > 1, "the model is overconfident: temperature > 1")
    check(
        after["accuracy"] == before["accuracy"],
        "calibration leaves the accuracy unchanged",
    )
    for name in ("nll", "brier", "ece"):
        check(after[name] < before[name], f"calibration lowers the {name}")


if __name__ == "__main__":
    main()
