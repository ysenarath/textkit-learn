import functools
import unittest

import torch
import torch.nn.functional as F
from test_nn import TrainerTestCase
from torch.utils.data import DataLoader

from tklearn.metrics import Accuracy
from tklearn.nn import Module, Trainer
from tklearn.nn.calibration import (
    CalibratedModule,
    calibrate,
    fit_temperature,
)


def sample(probs, activation, generator):
    """Labels drawn from `probs`: class indices for softmax, 0 or 1 per
    entry for sigmoid."""
    if activation == "softmax":
        return torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    return torch.bernoulli(probs, generator=generator)


class TestFitTemperature(unittest.TestCase):
    def test_recovers_the_temperature(self):
        # labels drawn from softmax(z) or sigmoid(z); the logits are z * 3,
        # so the temperature that fits them is 3
        g = torch.Generator().manual_seed(0)
        cases = {
            "multiclass": (torch.randn(20000, 4, generator=g), "softmax"),
            "binary": (torch.randn(20000, generator=g), "sigmoid"),
            "multilabel": (torch.randn(20000, 3, generator=g), "sigmoid"),
        }
        for name, (z, activation) in cases.items():
            with self.subTest(name):
                probs = (
                    z.softmax(-1) if activation == "softmax" else z.sigmoid()
                )
                y_true = sample(probs, activation, g)
                temperature = fit_temperature(
                    z * 3, y_true, activation=activation
                )
                self.assertAlmostEqual(temperature, 3, delta=0.15)

    def test_invalid_shapes(self):
        indices = torch.zeros(4, dtype=torch.long)
        cases = [
            (torch.zeros(4, 3), torch.zeros(5, dtype=torch.long), "softmax"),
            (torch.zeros(4), torch.zeros(4, 2), "sigmoid"),
            (torch.zeros(4, 3, 2), torch.zeros(4, 3, 2), "sigmoid"),
            # the wrong activation for the shapes
            (torch.zeros(4, 3), indices, "sigmoid"),
            (torch.zeros(4, 3), torch.zeros(4, 3), "softmax"),
            (torch.zeros(4, 1), indices, "sigmoid"),
        ]
        for logits, y_true, activation in cases:
            with self.subTest(
                logits=logits.shape, y_true=y_true.shape, activation=activation
            ):
                with self.assertRaisesRegex(ValueError, "expects logits"):
                    fit_temperature(logits, y_true, activation=activation)

    def test_invalid_activation(self):
        with self.assertRaisesRegex(ValueError, "softmax or sigmoid"):
            fit_temperature(
                torch.zeros(4, 3),
                torch.zeros(4, dtype=torch.long),
                activation="tanh",
            )


def make_data(n, num_classes, seed=0):
    # noisy labels, so that a large model fit to few of them is
    # overconfident
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 8, generator=g)
    w = torch.randn(8, num_classes, generator=g)
    labels = (x @ w).argmax(-1)
    noise = torch.rand(n, generator=g) < 0.3
    labels = torch.where(
        noise, torch.randint(num_classes, (n,), generator=g), labels
    )
    return [{"x": a, "labels": b} for a, b in zip(x, labels)]


class Classifier(Module):
    """Multiclass, or binary with the positive class's score alone."""

    def __init__(self, num_classes=3, positive_score=False):
        super().__init__()
        torch.manual_seed(0)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes),
        )
        self.positive_score = positive_score

    def forward(self, x):
        return self.net(x)

    def predict_step(self, batch):
        logits = self(batch["x"])
        y_score = logits.softmax(-1)
        outputs = {
            "logits": logits,
            "y_pred": logits.argmax(-1),
            "y_score": y_score[:, 1] if self.positive_score else y_score,
        }
        if "labels" in batch:
            outputs["y_true"] = batch["labels"]
            outputs["loss"] = F.cross_entropy(logits, batch["labels"])
        return outputs


class TestCalibrate(TrainerTestCase):
    def fit(self, model):
        trainer = Trainer(
            model,
            functools.partial(torch.optim.AdamW, lr=1e-2),
            metrics={"accuracy": Accuracy()},
            cpu=True,
            mixed_precision="no",
        )
        num_classes = model.net[-1].out_features
        train = DataLoader(make_data(128, num_classes), 32, shuffle=False)
        trainer.fit(train, epochs=30)
        return trainer

    def test_calibrates_an_overconfident_model(self):
        trainer = self.fit(Classifier())
        valid = DataLoader(make_data(512, 3, seed=1), 128)
        test = DataLoader(make_data(512, 3, seed=2), 128)
        calibrated = calibrate(trainer, valid, activation="softmax")
        self.assertIsInstance(calibrated, CalibratedModule)
        self.assertIs(calibrated.model, trainer.model)
        self.assertGreater(calibrated.temperature.item(), 1)
        before = trainer.evaluate(test)
        after = Trainer(
            calibrated, metrics={"accuracy": Accuracy()}, cpu=True
        ).evaluate(test)
        self.assertEqual(after["accuracy"], before["accuracy"])
        self.assertLess(after["loss"], before["loss"])

    def test_outputs(self):
        model = Classifier()
        calibrated = CalibratedModule(model, 2.0, "softmax")
        batch = {
            "x": torch.randn(5, 8),
            "labels": torch.tensor([0, 1, 2, 0, 1]),
        }
        with torch.no_grad():
            original = model.predict_step(batch)
            outputs = calibrated.predict_step(batch)
        logits = original["logits"] / 2
        torch.testing.assert_close(outputs["logits"], logits)
        torch.testing.assert_close(outputs["y_score"], logits.softmax(-1))
        torch.testing.assert_close(
            outputs["loss"],
            F.cross_entropy(logits, batch["labels"], reduction="none"),
        )
        torch.testing.assert_close(outputs["y_pred"], original["y_pred"])
        # without labels, there is no loss
        outputs = calibrated.predict_step({"x": batch["x"]})
        self.assertNotIn("loss", outputs)

    def test_keeps_the_shape_of_y_score(self):
        # a binary model giving the positive class's score alone
        trainer = self.fit(Classifier(num_classes=2, positive_score=True))
        calibrated = calibrate(
            trainer,
            DataLoader(make_data(256, 2, seed=1), 64),
            activation="softmax",
        )
        outputs = Trainer(calibrated, cpu=True).predict(
            DataLoader(make_data(64, 2, seed=2), 64)
        )
        self.assertEqual(outputs["y_score"].shape, (64,))
        torch.testing.assert_close(
            outputs["y_score"], outputs["logits"].softmax(-1)[:, 1]
        )

    def test_needs_logits_and_labels(self):
        class NoLogits(Classifier):
            def predict_step(self, batch):
                outputs = super().predict_step(batch)
                del outputs["logits"]
                return outputs

        loader = DataLoader(make_data(64, 3), 32)
        unlabelled = DataLoader([{"x": d["x"]} for d in make_data(64, 3)], 32)
        with self.assertRaisesRegex(ValueError, "logits"):
            calibrate(
                Trainer(NoLogits(), cpu=True), loader, activation="softmax"
            )
        with self.assertRaisesRegex(ValueError, "y_true"):
            calibrate(
                Trainer(Classifier(), cpu=True),
                unlabelled,
                activation="softmax",
            )

    def test_invalid_arguments(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            CalibratedModule(Classifier(), 0.0, "softmax")
        with self.assertRaisesRegex(ValueError, "softmax or sigmoid"):
            CalibratedModule(Classifier(), 1.0, "tanh")


if __name__ == "__main__":
    unittest.main()
