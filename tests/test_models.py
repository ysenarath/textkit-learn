import functools
import tempfile
import unittest

import torch
import torch.nn.functional as F
from test_nn import TrainerTestCase
from torch.utils.data import DataLoader
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertModel,
)

from tklearn.metrics import F1, SpanF1
from tklearn.nn import Trainer
from tklearn.nn.calibration import calibrate
from tklearn.nn.models import SequenceClassifier, TokenClassifier

TAGS = ["O", "B-X", "I-X"]


def tiny_config(**kwargs):
    return BertConfig(
        vocab_size=20,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        **kwargs,
    )


def sequence_model(num_labels):
    torch.manual_seed(0)
    return BertForSequenceClassification(tiny_config(num_labels=num_labels))


def token_model(num_labels=3):
    torch.manual_seed(0)
    return BertForTokenClassification(tiny_config(num_labels=num_labels))


def make_texts(n=64, length=8, seed=0):
    """Token ids, an attention mask hiding the last tokens of half the
    sequences, and per-token tags: B-X on token 3, I-X on a 3 after it."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, 6, (n, length), generator=g)
    mask = torch.ones(n, length, dtype=torch.long)
    mask[::2, -2:] = 0
    tags = torch.zeros(n, length, dtype=torch.long)
    tags[ids == 3] = 1
    tags[:, 1:][(ids[:, 1:] == 3) & (ids[:, :-1] == 3)] = 2
    tags[mask == 0] = -100
    return ids, mask, tags


def sequence_batches(task, n=64, batch_size=16):
    ids, mask, _ = make_texts(n)
    has3 = (ids == 3).any(-1)
    has4 = (ids == 4).any(-1)
    labels = {
        "multiclass": has3.long() + has4.long(),
        "binary": has3.long(),
        "multilabel": torch.stack([has3, has4], -1).long(),
        "regression": (ids == 3).sum(-1).float(),
    }[task]
    rows = [
        {
            "input_ids": ids[i],
            "attention_mask": mask[i],
            "labels": labels[i],
            "id": i,  # not a model input
        }
        for i in range(n)
    ]
    return DataLoader(rows, batch_size=batch_size)


class TestSequenceClassifier(TrainerTestCase):
    def test_outputs_of_each_task(self):
        cases = {
            "multiclass": (3, (16, 3)),
            "binary": (None, (16,)),
            "multilabel": (2, (16, 2)),
            "regression": (None, (16,)),
        }
        for task, (num_labels, shape) in cases.items():
            with self.subTest(task):
                model = SequenceClassifier(
                    sequence_model(num_labels or 1), num_labels, task=task
                )
                batch = next(iter(sequence_batches(task)))
                with torch.no_grad():
                    outputs = model.predict_step(batch)
                logits, labels = outputs["logits"], batch["labels"]
                self.assertEqual(logits.shape, shape)
                torch.testing.assert_close(outputs["y_true"], labels)
                self.assertEqual(outputs["loss"].shape, (16,))
                if task == "multiclass":
                    expected = F.cross_entropy(
                        logits, labels, reduction="none"
                    )
                    torch.testing.assert_close(
                        outputs["y_score"], logits.softmax(-1)
                    )
                    self.assertEqual(model.activation, "softmax")
                elif task == "regression":
                    expected = (logits - labels) ** 2
                    self.assertNotIn("y_score", outputs)
                    self.assertIsNone(model.activation)
                else:
                    expected = F.binary_cross_entropy_with_logits(
                        logits, labels.float(), reduction="none"
                    )
                    if task == "multilabel":
                        expected = expected.mean(-1)
                    torch.testing.assert_close(
                        outputs["y_pred"], (logits > 0).long()
                    )
                    self.assertEqual(model.activation, "sigmoid")
                torch.testing.assert_close(outputs["loss"], expected)

    def test_without_labels(self):
        model = SequenceClassifier(sequence_model(3), 3)
        batch = next(iter(sequence_batches("multiclass")))
        del batch["labels"]
        outputs = model.predict_step(batch)
        self.assertNotIn("loss", outputs)
        self.assertNotIn("y_true", outputs)

    def test_fits_evaluates_and_calibrates(self):
        for task, num_labels in (("multiclass", 3), ("binary", None)):
            with self.subTest(task):
                self.setUp()
                model = SequenceClassifier(
                    sequence_model(num_labels or 1), num_labels, task=task
                )
                trainer = Trainer(
                    model,
                    functools.partial(torch.optim.AdamW, lr=1e-2),
                    metrics={"f1": F1(average="macro")},
                    cpu=True,
                    mixed_precision="no",
                )
                loader = sequence_batches(task)
                before = trainer.evaluate(loader)
                trainer.fit(loader, epochs=15)
                after = trainer.evaluate(loader)
                self.assertLess(after["loss"], before["loss"])
                self.assertGreater(after["f1"], before["f1"])
                calibrated = calibrate(
                    trainer, loader, activation=model.activation
                )
                self.assertGreater(calibrated.temperature.item(), 0)

    def test_loads_by_name(self):
        with tempfile.TemporaryDirectory() as path:
            BertModel(tiny_config()).save_pretrained(path)
            model = SequenceClassifier(path, 4)
            self.assertEqual(model.num_labels, 4)
            binary = SequenceClassifier(path, task="binary")
            self.assertEqual(binary.num_labels, 1)
            regression = SequenceClassifier(path, task="regression")
            self.assertEqual(regression.num_labels, 1)

    def test_invalid_arguments(self):
        with self.assertRaisesRegex(ValueError, "task"):
            SequenceClassifier(sequence_model(2), task="ranking")
        with self.assertRaisesRegex(ValueError, "one logit"):
            SequenceClassifier(sequence_model(1), 2, task="binary")
        with self.assertRaisesRegex(ValueError, "2 or more"):
            SequenceClassifier(sequence_model(1))
        with self.assertRaisesRegex(ValueError, "3 outputs"):
            SequenceClassifier(sequence_model(2), 3)
        with self.assertRaisesRegex(TypeError, "loaded model"):
            SequenceClassifier(sequence_model(2), revision="main")


def token_batches(n=64, batch_size=16, lengths=None):
    ids, mask, tags = make_texts(n)
    rows = [
        {"input_ids": ids[i], "attention_mask": mask[i], "labels": tags[i]}
        for i in range(n)
    ]
    return DataLoader(rows, batch_size=batch_size)


class TestTokenClassifier(TrainerTestCase):
    def test_outputs_with_label_names(self):
        model = TokenClassifier(token_model(), label_names=TAGS).eval()
        batch = next(iter(token_batches()))
        with torch.no_grad():
            outputs = model.predict_step(batch)
        labels = batch["labels"]
        self.assertEqual(outputs["logits"].shape, (16, 8, 3))
        torch.testing.assert_close(outputs["mask"], labels != -100)
        # the tags of the tokens with a label: 6 in even rows, 8 in odd
        self.assertEqual(len(outputs["y_true"]), 16)
        self.assertEqual([len(t) for t in outputs["y_true"][:2]], [6, 8])
        self.assertEqual(
            outputs["y_true"][0], [TAGS[i] for i in labels[0][:6].tolist()]
        )
        pred = outputs["logits"].argmax(-1)
        self.assertEqual(
            outputs["y_pred"][1], [TAGS[i] for i in pred[1].tolist()]
        )
        expected = F.cross_entropy(
            outputs["logits"].flatten(0, 1),
            labels.flatten(),
            ignore_index=-100,
        )
        torch.testing.assert_close(outputs["loss"], expected)
        torch.testing.assert_close(model.training_step(batch), expected)

    def test_outputs_without_label_names(self):
        model = TokenClassifier(token_model())
        batch = next(iter(token_batches()))
        with torch.no_grad():
            outputs = model.predict_step(batch)
        mask = batch["labels"] != -100
        pred = outputs["logits"].argmax(-1)
        torch.testing.assert_close(
            outputs["y_pred"], pred.masked_fill(~mask, -100)
        )
        torch.testing.assert_close(outputs["y_true"], batch["labels"])
        # without labels, the attention mask
        del batch["labels"]
        with torch.no_grad():
            outputs = model.predict_step(batch)
        torch.testing.assert_close(
            outputs["mask"], batch["attention_mask"].bool()
        )
        self.assertNotIn("loss", outputs)

    def test_fits_and_evaluates_spans(self):
        model = TokenClassifier(token_model(), label_names=TAGS)
        trainer = Trainer(
            model,
            functools.partial(torch.optim.AdamW, lr=1e-2),
            metrics={"f1": SpanF1()},
            cpu=True,
            mixed_precision="no",
        )
        loader = token_batches()
        before = trainer.evaluate(loader)
        trainer.fit(loader, epochs=20)
        after = trainer.evaluate(loader)
        self.assertLess(after["loss"], before["loss"])
        self.assertGreater(after["f1"], max(before["f1"], 0.5))
        outputs = trainer.predict(loader)
        self.assertEqual(len(outputs["y_pred"]), 64)
        self.assertEqual(outputs["logits"].shape, (64, 8, 3))

    def test_invalid_arguments(self):
        with self.assertRaisesRegex(ValueError, "2 outputs"):
            TokenClassifier(token_model(3), label_names=["O", "B-X"])
        with self.assertRaisesRegex(ValueError, "label names"):
            TokenClassifier(token_model(3), 3, label_names=["O", "B-X"])
        with tempfile.TemporaryDirectory() as path:
            BertModel(tiny_config()).save_pretrained(path)
            model = TokenClassifier(path, label_names=TAGS)
            self.assertEqual(model.num_labels, 3)
            self.assertEqual(model.model.config.id2label[1], "B-X")


if __name__ == "__main__":
    unittest.main()
