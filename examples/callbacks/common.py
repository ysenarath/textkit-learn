"""Data, model and helpers shared by the callback examples."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tklearn.nn import Module

NUM_FEATURES = 16

# examples/outputs is ignored by git
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "callbacks"


class Classifier(Module):
    """A two-layer perceptron for binary classification."""

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(NUM_FEATURES, hidden), torch.nn.ReLU()
        )
        self.head = torch.nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    def predict_step(self, batch):
        logits = self(batch["x"])
        outputs = {
            "y_pred": logits.argmax(-1),
            "y_score": logits.softmax(-1),
        }
        if "labels" in batch:  # absent at inference time
            outputs["y_true"] = batch["labels"]
            outputs["loss"] = F.cross_entropy(logits, batch["labels"])
        return outputs


def make_splits(
    sizes: tuple[int, ...] = (128, 512, 512),
    *,
    label_noise: float = 0.2,
    seed: int = 0,
) -> list[list[dict[str, torch.Tensor]]]:
    """Splits of a noisy linear classification task, as lists of examples.

    The labels follow one random hyperplane, with `label_noise` of them
    flipped, so a large model overfits a small training split.
    """
    generator = torch.Generator().manual_seed(seed)
    n = sum(sizes)
    x = torch.randn(n, NUM_FEATURES, generator=generator)
    w = torch.randn(NUM_FEATURES, generator=generator)
    labels = (x @ w > 0).long()
    flip = torch.rand(n, generator=generator) < label_noise
    labels = torch.where(flip, 1 - labels, labels)
    examples = [{"x": x[i], "labels": labels[i]} for i in range(n)]
    splits, start = [], 0
    for size in sizes:
        splits.append(examples[start : start + size])
        start += size
    return splits


def make_loaders(
    batch_size: int = 32, **kwargs
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Train (shuffled), validation and test dataloaders of `make_splits`."""
    train, valid, test = make_splits(**kwargs)
    generator = torch.Generator().manual_seed(0)
    return (
        DataLoader(train, batch_size, shuffle=True, generator=generator),
        DataLoader(valid, 128),
        DataLoader(test, 128),
    )


def parse_args(doc: str | None, name: str) -> Path:
    """The output directory of an example, emptied for a fresh run."""
    return prepare_output(argument_parser(doc, name).parse_args().out)


def argument_parser(doc: str | None, name: str) -> argparse.ArgumentParser:
    """A parser of an example's options, with ``--out``; add others to
    it."""
    parser = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_DIR / name,
        help="directory to write to (default: %(default)s)",
    )
    return parser


def prepare_output(out: Path) -> Path:
    """`out`, emptied for a fresh run."""
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    return out


def check(condition: bool, message: str) -> None:
    """Print a passed check, or fail the example."""
    if not condition:
        raise AssertionError(f"FAILED: {message}")
    print(f"  ok  {message}")
