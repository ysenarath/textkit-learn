import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tklearn.metrics import F1, Accuracy
from tklearn.nn import Encoder, Evaluator, Module, Predictor, Trainer
from tklearn.nn.callbacks import Callback, History
from tklearn.nn.loss import LossDict


def collate(items):
    xs, ys = zip(*items)
    # 'ids' cannot be moved to a device and must pass through untouched
    return {
        "x": torch.stack(xs),
        "labels": torch.stack(ys),
        "ids": np.arange(len(xs)),
    }


class TinyClassifier(Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 8)
        self.head = torch.nn.Linear(8, 2)

    def predict_step(self, batch):
        hidden = torch.relu(self.encoder(batch["x"]))
        return {"logits": self.head(hidden), "pooler_output": hidden}

    def compute_loss(self, batch, output):
        return torch.nn.functional.cross_entropy(
            output["logits"], batch["labels"]
        )

    def compute_metric_inputs(self, batch, output):
        return {
            "y_true": batch["labels"],
            "y_pred": output["logits"].argmax(-1),
        }


class HookRecorder(Callback):
    def __init__(self):
        super().__init__()
        self.events = []

    def on_train_begin(self, logs=None):
        self.events.append("train_begin")

    def on_epoch_end(self, epoch, logs=None):
        self.events.append(("epoch_end", epoch, sorted(logs)))

    def on_train_batch_end(self, batch_idx, logs=None):
        self.events.append(("train_batch_end", batch_idx, sorted(logs)))

    def on_train_end(self, logs=None):
        self.events.append("train_end")

    def on_test_end(self, logs=None):
        self.events.append(("test_end", sorted(logs)))

    def on_predict_end(self, logs=None):
        self.events.append("predict_end")


class TestTrainingLoop(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        x = torch.randn(40, 4)
        y = (x[:, 0] > 0).long()
        dataset = TensorDataset(x, y)
        self.loader = DataLoader(dataset, batch_size=16, collate_fn=collate)
        self.model = TinyClassifier()

    def make_trainer(self, **kwargs):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        return Trainer(self.model, optimizer, **kwargs)

    def test_fit_returns_history_with_eval_results(self):
        evaluator = Evaluator(self.model, {"acc": Accuracy(), "f1": F1()})
        trainer = self.make_trainer(evaluator=evaluator)
        history = trainer.fit(
            self.loader, epochs=3, eval_dataloader=self.loader
        )
        self.assertIsInstance(history, History)
        self.assertEqual(history.epoch, [0, 1, 2])
        self.assertEqual(
            list(history.to_pandas().columns),
            ["loss", "valid_loss", "valid_acc", "valid_f1"],
        )

    def test_training_reduces_loss(self):
        history = self.make_trainer().fit(self.loader, epochs=20)
        losses = history.history["loss"]
        self.assertLess(losses[-1], losses[0])

    def test_hooks(self):
        recorder = HookRecorder()
        trainer = self.make_trainer(callbacks=[recorder])
        trainer.fit(self.loader, epochs=1)
        self.assertEqual(recorder.events[0], "train_begin")
        self.assertEqual(recorder.events[1], ("train_batch_end", 0, ["loss"]))
        self.assertEqual(recorder.events[-2], ("epoch_end", 0, ["loss"]))
        self.assertEqual(recorder.events[-1], "train_end")
        self.assertIsNone(recorder.trainer)  # released after fit

    def test_callback_errors_propagate(self):
        class Broken(Callback):
            def on_epoch_end(self, epoch, logs=None):
                raise RuntimeError("boom")

        with self.assertRaisesRegex(RuntimeError, "boom"):
            self.make_trainer(callbacks=Broken()).fit(self.loader)

    def test_external_loss_and_loss_dict(self):
        def loss(batch, output):
            ce = torch.nn.functional.cross_entropy(
                output["logits"], batch["labels"]
            )
            return LossDict(ce=ce, l2=1e-3 * output["logits"].pow(2).mean())

        history = self.make_trainer(loss=loss).fit(self.loader)
        self.assertEqual(sorted(history.history), ["ce", "l2"])

    def test_eval_dataloader_requires_evaluator(self):
        with self.assertRaises(ValueError):
            self.make_trainer().fit(self.loader, eval_dataloader=self.loader)

    def test_stop_training(self):
        class StopAfterFirstEpoch(Callback):
            def on_epoch_end(self, epoch, logs=None):
                self.trainer.stop_training = True

        trainer = self.make_trainer(callbacks=StopAfterFirstEpoch())
        self.assertEqual(trainer.fit(self.loader, epochs=5).epoch, [0])
        # a new fit starts fresh
        self.assertEqual(trainer.fit(self.loader, epochs=5).epoch, [0])

    def test_evaluate(self):
        recorder = HookRecorder()
        evaluator = Evaluator(
            self.model, {"acc": Accuracy()}, callbacks=recorder
        )
        results = evaluator.evaluate(self.loader, prefix="test_")
        self.assertEqual(sorted(results), ["test_acc", "test_loss"])
        self.assertIsInstance(results["test_loss"], float)
        self.assertEqual(
            recorder.events, [("test_end", ["test_acc", "test_loss"])]
        )

    def test_evaluate_without_loss(self):
        evaluator = Evaluator(
            self.model, {"acc": Accuracy()}, include_loss=False
        )
        self.assertEqual(list(evaluator.evaluate(self.loader)), ["acc"])

    def test_predict(self):
        predictor = Predictor(self.model)
        logits = predictor.predict(self.loader, output_key="logits")
        self.assertEqual(logits.shape, (40, 2))
        self.assertFalse(logits.requires_grad)
        outputs = predictor.predict(self.loader)
        self.assertEqual(set(outputs), {"logits", "pooler_output"})
        self.assertEqual(outputs["pooler_output"].shape, (40, 8))

    def test_encode(self):
        encoder = Encoder(self.model)
        self.assertEqual(encoder.encode(self.loader).shape, (40, 8))
        array = encoder.encode(self.loader, return_tensors="np")
        self.assertIsInstance(array, np.ndarray)
        as_lists = encoder.encode(
            self.loader, return_tensors=None, return_list=True
        )
        self.assertEqual(len(as_lists), 40)
        self.assertIsInstance(as_lists[0], list)
        with self.assertRaises(ValueError):
            encoder.encode(self.loader, return_tensors=None)


if __name__ == "__main__":
    unittest.main()
