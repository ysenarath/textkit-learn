"""Train an adapter for a text classifier, save it, and test it.

An adapter (Houlsby et al., 2019, "Parameter-Efficient Transfer Learning
for NLP") is a small bottleneck layer inserted into each layer of a
pretrained Transformer. Training only the adapters and the
classification head, with the pretrained weights frozen, comes close to
fine-tuning the whole model, and what is trained fits in a few MB.

The example adds a bottleneck adapter (``seq_bn``) with the `adapters`
library to DistilBERT, wraps the model in
`tklearn.nn.models.SequenceClassifier` and trains it on the Rotten
Tomatoes movie reviews with `tklearn.nn.Trainer`. It then saves the
adapter with its head, loads them into a fresh copy of the pretrained
model, and tests that model on the test set with `Trainer.evaluate` and
`Trainer.predict`.

python examples/adapters/train_and_test.py
python examples/adapters/train_and_test.py --model roberta-base

The model and dataset are downloaded from the Hugging Face Hub on the
first run. Training takes about a minute on a laptop GPU.
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Iterable
from pathlib import Path

import adapters
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)

from tklearn.metrics import F1, Accuracy
from tklearn.nn import Trainer
from tklearn.nn.callbacks import ProgbarLogger
from tklearn.nn.models import SequenceClassifier

ADAPTER = "rotten_tomatoes"
LABELS = ["negative", "positive"]

# examples/outputs is ignored by git
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "adapters"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_DIR / "train_and_test",
        help="directory to save the adapter to (default: %(default)s)",
    )
    return parser.parse_args()


def load_model(name: str) -> torch.nn.Module:
    """The pretrained model with a new classification head, set up for
    adapters."""
    model = AutoModelForSequenceClassification.from_pretrained(
        name,
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    )
    adapters.init(model)  # adds the adapter methods to the model
    return model


def adamw(lr: float):
    """An optimizer factory for `Trainer`: AdamW over the parameters
    that train, so that the frozen ones get no optimizer state."""

    def create(params: Iterable[torch.nn.Parameter]) -> torch.optim.AdamW:
        return torch.optim.AdamW([p for p in params if p.requires_grad], lr)

    return create


def check(condition: bool, message: str) -> None:
    """Print a passed check, or fail the example."""
    if not condition:
        raise AssertionError(f"FAILED: {message}")
    print(f"  ok  {message}")


def main() -> None:
    args = parse_args()
    shutil.rmtree(args.out, ignore_errors=True)
    args.out.mkdir(parents=True)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data = load_dataset("rotten_tomatoes")
    data = data.map(
        lambda batch: tokenizer(
            batch["text"], truncation=True, max_length=128
        ),
        batched=True,
        remove_columns="text",
    ).rename_column("label", "labels")
    collator = DataCollatorWithPadding(tokenizer)
    generator = torch.Generator().manual_seed(args.seed)
    train = DataLoader(
        data["train"],
        args.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )
    valid = DataLoader(data["validation"], 128, collate_fn=collator)
    test = DataLoader(data["test"], 128, collate_fn=collator)
    metrics = {"accuracy": Accuracy(), "f1": F1(average="macro")}

    # -- train ------------------------------------------------------------

    model = load_model(args.model)
    model.add_adapter(ADAPTER, config="seq_bn")
    # freezes the pretrained weights, leaving the adapter and the head to
    # train, and activates the adapter
    model.train_adapter(ADAPTER)
    frozen = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if not param.requires_grad
    }
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"training {trainable:,} of {total:,} parameters "
        f"({trainable / total:.1%})\n"
    )

    trainer = Trainer(
        SequenceClassifier(model),
        adamw(args.lr),
        lr_scheduler="linear",
        warmup=0.1,
        max_grad_norm=1.0,
        metrics=metrics,
    )
    history = trainer.fit(
        train, valid, epochs=args.epochs, callbacks=[ProgbarLogger()]
    )
    trained = trainer.evaluate(test, callbacks=[ProgbarLogger()])

    # saves the adapter's weights and config, and the head's
    model.save_adapter(str(args.out), ADAPTER)
    saved = sum(f.stat().st_size for f in args.out.iterdir())

    # -- test -------------------------------------------------------------

    model = load_model(args.model)
    model.load_adapter(str(args.out), set_active=True)
    tester = Trainer(SequenceClassifier(model), metrics=metrics)
    tested = tester.evaluate(test, callbacks=[ProgbarLogger()])
    before = trainer.predict(test)
    after = tester.predict(test)

    print(f"\n{'epoch':<6} {'loss':>8} {'valid_loss':>11} {'valid_f1':>9}")
    for epoch, logs in enumerate(history, 1):
        print(
            f"{epoch:<6} {logs['loss']:>8.4f} {logs['valid_loss']:>11.4f} "
            f"{logs['valid_f1']:>9.4f}"
        )
    print(f"\nsaved {saved / 2**20:.1f} MB to {args.out}")
    print(f"\n{'test':<10} {'trained':>8} {'loaded':>8}")
    for name in tested:
        print(f"{name:<10} {trained[name]:>8.4f} {tested[name]:>8.4f}")
    print()

    model = trainer.model.model
    check(
        all(
            "adapter" in name
            or name.startswith(("pre_classifier", "classifier"))
            for name, param in model.named_parameters()
            if param.requires_grad
        ),
        "only the adapter and the head train",
    )
    check(
        all(
            torch.equal(param.cpu(), frozen[name])
            for name, param in model.named_parameters()
            if name in frozen
        ),
        "fit leaves the pretrained weights unchanged",
    )
    check(
        saved < 0.05 * total * 4,  # float32
        "the saved adapter is under 5% of the model's size",
    )
    check(trained["accuracy"] > 0.8, "the adapter learns the task")
    check(
        torch.equal(after["y_pred"], before["y_pred"])
        and torch.allclose(after["y_score"], before["y_score"], atol=1e-5),
        "the loaded adapter predicts as the trained one",
    )


if __name__ == "__main__":
    main()
