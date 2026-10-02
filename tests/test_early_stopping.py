import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tklearn.nn import Module, Trainer
from tklearn.nn.callbacks import EarlyStopping


class MockModel:
    def __init__(self):
        self.state = {"weights": np.random.rand(10)}

    def state_dict(self):
        return self.state.copy()

    def load_state_dict(self, state, strict=True):
        self.state = state.copy()


class MockTrainer:
    def __init__(self):
        self.stop_training = False


def make_callback(**kwargs):
    model, trainer = MockModel(), MockTrainer()
    es = EarlyStopping(**kwargs)
    es.set_model(model)
    es.set_trainer(trainer)
    es.on_train_begin()
    return es, model, trainer


class TestEarlyStopping(unittest.TestCase):
    def test_init_defaults(self):
        es = EarlyStopping()
        self.assertEqual(es.monitor, "valid_loss")
        self.assertEqual(es.min_delta, 0)
        self.assertEqual(es.patience, 0)
        self.assertEqual(es.verbose, 0)
        self.assertEqual(es.mode, "min")
        self.assertIsNone(es.baseline)
        self.assertTrue(es.restore_best_weights)
        self.assertEqual(es.start_from_epoch, 0)

    def test_explicit_mode(self):
        self.assertEqual(
            EarlyStopping(monitor="valid_accuracy", mode="max").mode, "max"
        )
        self.assertEqual(
            EarlyStopping(monitor="valid_loss", mode="min").mode, "min"
        )
        with self.assertRaises(ValueError):
            EarlyStopping(mode="invalid")

    def test_auto_mode(self):
        for monitor in ("valid_accuracy", "valid_auc", "valid_f1"):
            self.assertEqual(EarlyStopping(monitor=monitor).mode, "max")
        for monitor in ("valid_loss", "valid_error"):
            self.assertEqual(EarlyStopping(monitor=monitor).mode, "min")
        with self.assertRaises(ValueError):
            EarlyStopping(monitor="unknown_metric")

    def test_min_delta(self):
        es, _, trainer = make_callback(monitor="valid_loss", min_delta=0.1)
        es.on_epoch_end(0, logs={"valid_loss": 1.0})
        es.on_epoch_end(1, logs={"valid_loss": 0.95})  # within min_delta
        self.assertEqual(es.best, 1.0)
        es.on_epoch_end(2, logs={"valid_loss": 0.8})
        self.assertEqual(es.best, 0.8)

    def test_patience(self):
        es, _, trainer = make_callback(monitor="valid_loss", patience=2)
        es.on_epoch_end(0, logs={"valid_loss": 1.0})
        self.assertFalse(trainer.stop_training)
        es.on_epoch_end(1, logs={"valid_loss": 1.1})  # wait = 1
        self.assertFalse(trainer.stop_training)
        es.on_epoch_end(2, logs={"valid_loss": 1.2})  # wait = 2
        self.assertTrue(trainer.stop_training)
        self.assertEqual(es.stopped_epoch, 2)

    def test_zero_patience_keeps_training_while_improving(self):
        es, _, trainer = make_callback(monitor="valid_loss", patience=0)
        for epoch, loss in enumerate([1.0, 0.9, 0.8, 0.7]):
            es.on_epoch_end(epoch, logs={"valid_loss": loss})
            self.assertFalse(trainer.stop_training)
        es.on_epoch_end(4, logs={"valid_loss": 0.75})
        self.assertTrue(trainer.stop_training)

    def test_restore_best_weights(self):
        es, model, _ = make_callback(
            monitor="valid_loss", patience=2, restore_best_weights=True
        )
        es.on_epoch_end(0, logs={"valid_loss": 1.0})
        initial_weights = model.state_dict()["weights"].copy()
        model.state.update({"weights": np.random.rand(10)})
        es.on_epoch_end(1, logs={"valid_loss": 0.9})
        best_weights = model.state_dict()["weights"].copy()
        model.state.update({"weights": np.random.rand(10)})
        es.on_epoch_end(2, logs={"valid_loss": 1.1})
        es.on_epoch_end(3, logs={"valid_loss": 1.2})
        np.testing.assert_array_almost_equal(
            model.state["weights"], best_weights
        )
        self.assertFalse(
            np.array_equal(model.state["weights"], initial_weights)
        )

    def test_start_from_epoch(self):
        es, _, trainer = make_callback(
            monitor="valid_loss", patience=2, start_from_epoch=2
        )
        es.on_epoch_end(0, logs={"valid_loss": 1.0})
        es.on_epoch_end(1, logs={"valid_loss": 1.1})
        es.on_epoch_end(2, logs={"valid_loss": 1.2})
        self.assertFalse(trainer.stop_training)
        es.on_epoch_end(3, logs={"valid_loss": 1.3})
        self.assertFalse(trainer.stop_training)
        es.on_epoch_end(4, logs={"valid_loss": 1.4})
        self.assertEqual(es.stopped_epoch, 4)
        self.assertTrue(trainer.stop_training)

    def test_baseline(self):
        es, _, _ = make_callback(monitor="valid_loss", baseline=0.5)
        es.on_epoch_end(0, logs={"valid_loss": 1.0})
        self.assertEqual(es.wait, 1)
        es.on_epoch_end(1, logs={"valid_loss": 0.6})
        self.assertEqual(es.wait, 2)
        es.on_epoch_end(2, logs={"valid_loss": 0.4})
        self.assertEqual(es.wait, 0)

    def test_ignores_missing_and_nan_values(self):
        es, _, trainer = make_callback(monitor="valid_loss", patience=0)
        es.on_epoch_end(0, logs={})
        es.on_epoch_end(1, logs={"valid_loss": float("nan")})
        self.assertEqual(es.wait, 0)
        self.assertFalse(trainer.stop_training)

    def test_stops_trainer(self):
        class ConstantLoss(Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.zeros(1))

            def predict_step(self, batch):
                return self.w * 0

            def compute_loss(self, batch, output):
                return output.sum() + 1.0

        model = ConstantLoss()
        es = EarlyStopping(monitor="loss", patience=1)
        loader = DataLoader(TensorDataset(torch.zeros(4, 1)), batch_size=2)
        trainer = Trainer(
            model, torch.optim.SGD(model.parameters(), lr=0.1), callbacks=es
        )
        history = trainer.fit(loader, epochs=10)
        self.assertEqual(history.epoch, [0, 1])
        self.assertEqual(es.stopped_epoch, 1)


if __name__ == "__main__":
    unittest.main()
