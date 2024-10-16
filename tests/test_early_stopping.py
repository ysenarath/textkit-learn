import unittest

import numpy as np
from tklearn.nn.callbacks.early_stopping import EarlyStopping


class MockModel:
    def __init__(self):
        self.stop_training = False
        self.state = {"weights": np.random.rand(10)}

    def state_dict(self):
        return self.state.copy()

    def load_state_dict(self, state, strict=True):
        self.state = state.copy()


class TestEarlyStopping(unittest.TestCase):
    def test_early_stopping_init(self):
        es = EarlyStopping()
        self.assertEqual(es.monitor, "valid_loss")
        self.assertEqual(es.min_delta, 0)
        self.assertEqual(es.patience, 0)
        self.assertEqual(es.verbose, 0)
        self.assertEqual(es.mode, "auto")
        self.assertIsNone(es.baseline)
        self.assertTrue(es.restore_best_weights)
        self.assertEqual(es.start_from_epoch, 0)

    def test_early_stopping_mode(self):
        es = EarlyStopping(monitor="val_accuracy", mode="max")
        self.assertEqual(es.monitor_op, np.greater)

        es = EarlyStopping(monitor="val_loss", mode="min")
        self.assertEqual(es.monitor_op, np.less)

        with self.assertRaises(ValueError):
            EarlyStopping(mode="invalid")

    def test_early_stopping_improvement(self):
        es = EarlyStopping(monitor="val_loss", min_delta=0.1)
        self.assertTrue(es._is_improvement(0.8, 1.0))  # Improvement
        self.assertFalse(
            es._is_improvement(0.95, 1.0)
        )  # Not an improvement (less than min_delta)
        self.assertFalse(
            es._is_improvement(1.1, 1.0)
        )  # Not an improvement (higher loss)

    def test_early_stopping_patience(self):
        model = MockModel()
        es = EarlyStopping(monitor="val_loss", patience=2)
        es.set_model(model)

        es.on_train_begin()
        es.on_epoch_end(0, logs={"val_loss": 1.0})  # Initial value set
        self.assertFalse(model.stop_training)

        es.on_epoch_end(1, logs={"val_loss": 1.1})  # No improvement, wait = 1
        self.assertFalse(model.stop_training)

        es.on_epoch_end(2, logs={"val_loss": 1.2})  # No improvement, wait = 2
        self.assertTrue(model.stop_training)
        self.assertEqual(es.stopped_epoch, 2)

    def test_early_stopping_restore_best_weights(self):
        model = MockModel()
        es = EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )
        es.set_model(model)

        es.on_train_begin()
        es.on_epoch_end(0, logs={"val_loss": 1.0})
        initial_weights = model.state_dict()["weights"].copy()

        # train: update the weights
        model.state.update({"weights": np.random.rand(10)})

        es.on_epoch_end(1, logs={"val_loss": 0.9})
        # update the best weights
        best_weights = model.state_dict()["weights"].copy()

        es.on_epoch_end(2, logs={"val_loss": 1.1})
        es.on_epoch_end(3, logs={"val_loss": 1.2})

        # The best weights (from epoch 1) should be restored
        np.testing.assert_array_almost_equal(
            model.state["weights"], best_weights
        )
        self.assertFalse(
            np.array_equal(model.state["weights"], initial_weights)
        )

    def test_early_stopping_start_from_epoch(self):
        model = MockModel()
        es = EarlyStopping(monitor="val_loss", patience=2, start_from_epoch=2)
        es.set_model(model)

        es.on_train_begin()
        es.on_epoch_end(0, logs={"val_loss": 1.0})
        es.on_epoch_end(1, logs={"val_loss": 1.1})

        # Early stopping should start from here
        es.on_epoch_end(2, logs={"val_loss": 1.2})
        self.assertFalse(model.stop_training)
        es.on_epoch_end(3, logs={"val_loss": 1.3})
        self.assertFalse(model.stop_training)
        es.on_epoch_end(4, logs={"val_loss": 1.4})

        self.assertEqual(es.stopped_epoch, 4)
        self.assertTrue(model.stop_training)

    def test_early_stopping_baseline(self):
        model = MockModel()
        es = EarlyStopping(monitor="val_loss", baseline=0.5)
        es.set_model(model)

        es.on_train_begin()
        es.on_epoch_end(0, logs={"val_loss": 1.0})
        self.assertEqual(es.wait, 1)

        es.on_epoch_end(1, logs={"val_loss": 0.6})
        self.assertEqual(es.wait, 2)

        es.on_epoch_end(2, logs={"val_loss": 0.4})
        self.assertEqual(es.wait, 0)

    def test_early_stopping_auto_mode(self):
        es = EarlyStopping(monitor="val_accuracy")
        self.assertEqual(es.monitor_op, np.greater)

        es = EarlyStopping(monitor="val_loss")
        self.assertEqual(es.monitor_op, np.less)

        es = EarlyStopping(monitor="val_auc")
        self.assertEqual(es.monitor_op, np.greater)

        es = EarlyStopping(monitor="val_error")
        self.assertEqual(es.monitor_op, np.less)

        with self.assertRaises(ValueError):
            EarlyStopping(monitor="unknown_metric")

    # def test_early_stopping_patience_with_min_angle(self):
    #     model = MockModel()
    #     es = EarlyStopping(monitor="val_loss", patience=5, min_angle=0.01)
    #     es.set_model(model)
    #     es.on_train_begin()
    #     stopped_epoch = 0
    #     for i, val_loss in enumerate([
    #         1.0,
    #         0.9,
    #         0.8,
    #         0.7,
    #         0.69,
    #         0.68,
    #         0.67,
    #         0.66,
    #     ]):
    #         es.on_epoch_end(0, logs={"val_loss": val_loss})
    #         if model.stop_training:
    #             stopped_epoch = i
    #             break
    #     self.assertEqual(es.stopped_epoch, stopped_epoch)


if __name__ == "__main__":
    unittest.main()
