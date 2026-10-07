import csv
import io
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_model
from test_nn import Classifier, TrainerTestCase

from tklearn.metrics import ConfusionMatrix
from tklearn.nn.callbacks import (
    Callback,
    CSVLogger,
    EarlyStopping,
    LambdaCallback,
    ModelCheckpoint,
    ProgbarLogger,
    ReduceLROnPlateau,
    TerminateOnNaN,
)


class Feed(Callback):
    """Logs ``values[epoch]`` as `key` and keeps the weights of each epoch.

    Put it before the callback under test, which then sees the value.
    """

    def __init__(self, values, key="score"):
        self.values, self.key = values, key
        self.weights = []

    def on_epoch_end(self, trainer, logs):
        logs[self.key] = self.values[trainer.epoch]
        self.weights.append({
            k: v.clone() for k, v in trainer.model.state_dict().items()
        })


def assert_weights_equal(model, weights):
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[key])


class TestTrainerState(TrainerTestCase):
    def test_epochs_and_num_batches(self):
        seen = []

        def record(name):
            def hook(trainer, *args):
                seen.append((name, trainer.epochs, trainer.num_batches))

            return hook

        hooks = ["on_epoch_begin", "on_test_begin", "on_epoch_end"]
        callback = LambdaCallback(**{name: record(name) for name in hooks})
        trainer = self.trainer(callbacks=[callback])
        # 6 training batches, 2 evaluation batches
        trainer.fit(self.loader(), self.loader(self.data[:32]), epochs=2)
        self.assertEqual(
            seen,
            [
                ("on_epoch_begin", 2, 6),
                ("on_test_begin", 2, 2),
                ("on_epoch_end", 2, 6),
            ]
            * 2,
        )
        seen.clear()
        trainer.callbacks = [LambdaCallback(on_predict_begin=record("p"))]
        trainer.predict(self.loader(batch_size=48))
        self.assertEqual(seen, [("p", 2, 2)])
        self.assertEqual(trainer.num_batches, 6)


class TestLambdaCallback(TrainerTestCase):
    def test_calls_hooks(self):
        calls = []
        callback = LambdaCallback(
            on_train_begin=lambda trainer: calls.append("begin"),
            on_epoch_end=lambda trainer, logs: calls.append(logs["loss"]),
        )
        history = self.trainer(callbacks=[callback]).fit(
            self.loader(), epochs=2
        )
        self.assertEqual(calls, ["begin", *(logs["loss"] for logs in history)])

    def test_rejects_unknown_hooks(self):
        with self.assertRaisesRegex(TypeError, "on_fit_end"):
            LambdaCallback(on_fit_end=print)
        with self.assertRaisesRegex(TypeError, "callable"):
            LambdaCallback(on_train_end=1)


class Recorder(Callback):
    """Records ``(name, hook)`` of every hook it gets into `calls`."""

    def __init__(self, name, calls, wrapper=False):
        self.name, self.calls = name, calls
        self.wrapper = wrapper

    def __getattribute__(self, attr):
        if attr.startswith("on_"):
            calls, name = self.calls, self.name
            return lambda trainer, *args: calls.append((name, attr))
        return super().__getattribute__(attr)


class TestWrapperCallbacks(TrainerTestCase):
    def order(self, hook, calls):
        return [name for name, h in calls if h == hook]

    def test_wrappers_enclose_the_others(self):
        calls = []
        callbacks = [
            Recorder("a", calls),
            Recorder("outer", calls, wrapper=True),
            Recorder("b", calls),
            Recorder("inner", calls, wrapper=True),
        ]
        trainer = self.trainer(callbacks=callbacks)
        trainer.fit(self.loader(), self.loader(), epochs=1)
        trainer.predict(self.loader())
        begin = ["outer", "inner", "a", "b"]
        end = ["a", "b", "inner", "outer"]
        for kind in ("train", "epoch", "train_batch", "test", "predict"):
            with self.subTest(kind=kind):
                self.assertEqual(
                    self.order(f"on_{kind}_begin", calls)[:4], begin
                )
                self.assertEqual(self.order(f"on_{kind}_end", calls)[:4], end)
        # other hooks run in the order given
        self.assertEqual(
            self.order("on_before_optimizer_step", calls)[:4],
            ["a", "outer", "b", "inner"],
        )

    def test_changes_to_the_callbacks_apply_from_the_next_hook(self):
        for wrapper in (False, True):
            with self.subTest(wrapper=wrapper):
                calls = []
                added = Recorder("added", calls)

                def add(trainer):
                    trainer.callbacks.append(added)

                callbacks = [
                    LambdaCallback(on_train_begin=add),
                    Recorder("first", calls, wrapper=wrapper),
                ]
                trainer = self.trainer(callbacks=callbacks)
                trainer.fit(self.loader(), epochs=1)
                # not this hook, but the ones after it
                self.assertEqual(
                    self.order("on_train_begin", calls), ["first"]
                )
                self.assertIn("added", self.order("on_epoch_begin", calls))

    def test_without_wrappers_hooks_run_in_order(self):
        calls = []
        callbacks = [Recorder("a", calls), Recorder("b", calls)]
        self.trainer(callbacks=callbacks).fit(self.loader(), epochs=1)
        self.assertEqual(self.order("on_train_end", calls), ["a", "b"])
        self.assertFalse(Callback.wrapper)


class TestEarlyStopping(TrainerTestCase):
    def fit(self, values, epochs=None, **kwargs):
        feed = Feed(values)
        early_stopping = EarlyStopping("score", mode="min", **kwargs)
        trainer = self.trainer(callbacks=[feed, early_stopping])
        trainer.fit(self.loader(), epochs=epochs or len(values))
        return trainer, feed, early_stopping

    def test_stops_after_patience(self):
        values = [1.0, 0.5, 0.6, 0.7, 0.8, 0.1]
        trainer, _, early_stopping = self.fit(values, patience=2)
        self.assertEqual(len(trainer.history), 4)
        self.assertEqual(early_stopping.stopped_epoch, 3)
        self.assertEqual(early_stopping.best, 0.5)
        self.assertEqual(early_stopping.best_epoch, 1)

    def test_zero_patience_stops_at_first_epoch_without_improvement(self):
        trainer, _, _ = self.fit([1.0, 0.5, 0.6, 0.1])
        self.assertEqual(len(trainer.history), 3)

    def test_runs_all_epochs_while_improving(self):
        trainer, _, early_stopping = self.fit([4.0, 3.0, 2.0, 1.0])
        self.assertEqual(len(trainer.history), 4)
        self.assertIsNone(early_stopping.stopped_epoch)

    def test_min_delta(self):
        trainer, _, _ = self.fit([1.0, 0.95, 0.9, 0.5], min_delta=0.1)
        self.assertEqual(len(trainer.history), 2)

    def test_max_mode(self):
        feed = Feed([0.5, 0.7, 0.6, 0.9], key="valid_f1")
        early_stopping = EarlyStopping("valid_f1")  # inferred
        self.assertEqual(early_stopping.mode, "max")
        trainer = self.trainer(callbacks=[feed, early_stopping])
        trainer.fit(self.loader(), epochs=4)
        self.assertEqual(len(trainer.history), 3)
        self.assertEqual(early_stopping.best, 0.7)

    def test_baseline(self):
        # improvements that do not reach the baseline count as patience
        trainer, _, _ = self.fit(
            [0.9, 0.8, 0.7, 0.6], patience=2, baseline=0.5
        )
        self.assertEqual(len(trainer.history), 2)

    def test_start_from_epoch(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        trainer, _, early_stopping = self.fit(values, start_from_epoch=2)
        self.assertEqual(len(trainer.history), 4)
        self.assertEqual(early_stopping.best_epoch, 2)

    def test_restores_best_weights(self):
        values = [1.0, 0.5, 0.6, 0.7]
        trainer, feed, _ = self.fit(
            values, patience=2, restore_best_weights=True
        )
        assert_weights_equal(trainer.model, feed.weights[1])
        self.assertFalse(
            torch.equal(
                feed.weights[1]["linear.weight"],
                feed.weights[3]["linear.weight"],
            )
        )

    def test_restores_best_weights_without_stopping(self):
        trainer, feed, _ = self.fit(
            [1.0, 0.5, 0.6], patience=5, restore_best_weights=True
        )
        assert_weights_equal(trainer.model, feed.weights[1])

    def test_keeps_weights_by_default(self):
        trainer, feed, _ = self.fit([1.0, 0.5, 0.6, 0.7], patience=2)
        assert_weights_equal(trainer.model, feed.weights[-1])

    def test_nan_does_not_improve(self):
        trainer, _, early_stopping = self.fit([1.0, math.nan, 0.5])
        self.assertEqual(len(trainer.history), 2)
        self.assertEqual(early_stopping.best, 1.0)

    def test_resets_between_fits(self):
        feed = Feed([1.0, 2.0, 3.0])
        early_stopping = EarlyStopping("score", mode="min")
        trainer = self.trainer(callbacks=[feed, early_stopping])
        trainer.fit(self.loader(), epochs=3)
        self.assertEqual(len(trainer.history), 2)
        feed.values = [5.0, 4.0, 3.0]
        trainer.fit(self.loader(), epochs=3)
        self.assertEqual(len(trainer.history), 3)
        self.assertIsNone(early_stopping.stopped_epoch)

    def test_warns_when_monitor_is_missing(self):
        trainer = self.trainer(callbacks=[EarlyStopping("valid_loss")])
        with self.assertWarnsRegex(UserWarning, "'valid_loss'.*loss"):
            trainer.fit(self.loader(), epochs=2)
        self.assertEqual(len(trainer.history), 2)

    def test_validates_arguments(self):
        with self.assertRaisesRegex(ValueError, "mode='min'"):
            EarlyStopping("perplexity_of_x")
        with self.assertRaisesRegex(ValueError, "mode must be"):
            EarlyStopping(mode="lowest")
        with self.assertRaisesRegex(ValueError, "patience"):
            EarlyStopping(patience=-1)

    def test_rejects_non_numeric_monitor(self):
        trainer = self.trainer(
            metrics={"cm": ConfusionMatrix()},
            callbacks=[EarlyStopping("valid_cm", mode="max")],
        )
        with self.assertRaisesRegex(TypeError, "must be a number"):
            trainer.fit(self.loader(), self.loader(), epochs=1)


class TestTerminateOnNaN(TrainerTestCase):
    def test_stops_on_nan_loss(self):
        model = Classifier()
        step = model.training_step
        calls = []

        def training_step(batch):
            calls.append(None)
            loss = step(batch)
            return loss * math.nan if len(calls) == 3 else loss

        model.training_step = training_step
        trainer = self.trainer(model, callbacks=[TerminateOnNaN()])
        history = trainer.fit(self.loader(), epochs=3)
        self.assertEqual(len(history), 1)
        self.assertEqual(trainer.global_step, 3)
        self.assertTrue(math.isnan(history[0]["loss"]))

    def test_keeps_training_on_finite_loss(self):
        trainer = self.trainer(callbacks=[TerminateOnNaN()])
        self.assertEqual(len(trainer.fit(self.loader(), epochs=2)), 2)


class TestModelCheckpoint(TrainerTestCase):
    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def files(self):
        return sorted(p.name for p in self.dir.rglob("*") if p.is_file())

    def test_saves_every_epoch(self):
        feed = Feed([0.5, 0.25, 0.125])
        checkpoint = ModelCheckpoint(
            self.dir / "{epoch:02d}-{score:.3f}.safetensors"
        )
        trainer = self.trainer(callbacks=[feed, checkpoint])
        trainer.fit(self.loader(), epochs=3)
        self.assertEqual(
            self.files(),
            [
                "01-0.500.safetensors",
                "02-0.250.safetensors",
                "03-0.125.safetensors",
            ],
        )
        model = Classifier()
        load_model(model, self.dir / "02-0.250.safetensors")
        assert_weights_equal(model, feed.weights[1])

    def test_saves_best_only(self):
        feed = Feed([1.0, 0.5, 0.7, 0.4], key="valid_loss")
        checkpoint = ModelCheckpoint(
            self.dir / "ckpt-{epoch}.pt", save_best_only=True
        )
        trainer = self.trainer(callbacks=[feed, checkpoint])
        trainer.fit(self.loader(), epochs=4)
        self.assertEqual(self.files(), ["ckpt-1.pt", "ckpt-2.pt", "ckpt-4.pt"])
        self.assertEqual(checkpoint.best, 0.4)
        state = torch.load(self.dir / "ckpt-2.pt")
        for key, value in feed.weights[1].items():
            torch.testing.assert_close(state[key], value)

    def test_best_carries_over_between_fits(self):
        feed = Feed([0.5, 0.4], key="valid_loss")
        path = self.dir / "best.safetensors"
        checkpoint = ModelCheckpoint(path, save_best_only=True)
        trainer = self.trainer(callbacks=[feed, checkpoint])
        trainer.fit(self.loader(), epochs=2)
        feed.values = [0.9, 0.8]
        trainer.fit(self.loader(), epochs=2)
        model = Classifier()
        load_model(model, path)
        assert_weights_equal(model, feed.weights[1])

    def test_initial_value_threshold(self):
        feed = Feed([0.5, 0.3, 0.1], key="valid_loss")
        checkpoint = ModelCheckpoint(
            self.dir / "{epoch}.pt",
            save_best_only=True,
            initial_value_threshold=0.2,
        )
        self.trainer(callbacks=[feed, checkpoint]).fit(self.loader(), epochs=3)
        self.assertEqual(self.files(), ["3.pt"])

    def test_saves_every_n_steps(self):
        # 6 batches, 2 per optimizer step: steps 1..3 per epoch
        checkpoint = ModelCheckpoint(self.dir / "step{step}.pt", save_freq=2)
        trainer = self.trainer(accumulation=2, callbacks=[checkpoint])
        trainer.fit(self.loader(), epochs=2)
        self.assertEqual(self.files(), ["step2.pt", "step4.pt", "step6.pt"])

    def test_missing_format_key(self):
        checkpoint = ModelCheckpoint(self.dir / "{valid_f1}.pt")
        trainer = self.trainer(callbacks=[checkpoint])
        with self.assertRaisesRegex(KeyError, "'valid_f1'"):
            trainer.fit(self.loader(), epochs=1)

    def test_validates_arguments(self):
        with self.assertRaisesRegex(ValueError, "filepath"):
            ModelCheckpoint("model.h5")
        for save_freq in ("batch", 0, 1.5, True):
            with self.subTest(save_freq=save_freq):
                with self.assertRaisesRegex(ValueError, "save_freq"):
                    ModelCheckpoint("model.pt", save_freq=save_freq)


class TestReduceLROnPlateau(TrainerTestCase):
    def fit(self, values, lr=0.5, **kwargs):
        lrs = []
        feed = Feed(values, key="valid_loss")
        reduce_lr = ReduceLROnPlateau(**kwargs)
        record = LambdaCallback(
            on_epoch_end=lambda trainer, logs: lrs.append(
                trainer.optimizer.param_groups[0]["lr"]
            )
        )
        trainer = self.trainer(lr=lr, callbacks=[feed, reduce_lr, record])
        trainer.fit(self.loader(), epochs=len(values))
        return lrs

    def test_reduces_after_patience(self):
        lrs = self.fit([1.0, 0.5, 0.6, 0.7, 0.8, 0.9], patience=2, factor=0.5)
        self.assertEqual(lrs, [0.5, 0.5, 0.5, 0.25, 0.25, 0.125])

    def test_cooldown(self):
        lrs = self.fit(
            [1.0, 1.1, 1.2, 1.3, 1.4], patience=1, factor=0.5, cooldown=2
        )
        self.assertEqual(lrs, [0.5, 0.25, 0.25, 0.25, 0.125])

    def test_min_lr(self):
        lrs = self.fit([1.0, 1.1, 1.2, 1.3], patience=1, min_lr=0.01)
        self.assertEqual(lrs, [0.5, 0.05, 0.01, 0.01])

    def test_rejects_lr_scheduler(self):
        trainer = self.trainer(
            lr_scheduler="constant", callbacks=[ReduceLROnPlateau()]
        )
        with self.assertRaisesRegex(ValueError, "lr_scheduler"):
            trainer.fit(self.loader())

    def test_validates_factor(self):
        for factor in (0, 1, 1.5):
            with self.subTest(factor=factor):
                with self.assertRaisesRegex(ValueError, "factor"):
                    ReduceLROnPlateau(factor=factor)


class TestCSVLogger(TrainerTestCase):
    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "log.csv"

    def read(self):
        with open(self.path, newline="") as f:
            return list(csv.reader(f))

    def test_writes_history(self):
        trainer = self.trainer(callbacks=[CSVLogger(self.path)])
        history = trainer.fit(self.loader(), self.loader(), epochs=3)
        rows = self.read()
        self.assertEqual(rows[0], ["epoch", "loss", "valid_loss"])
        self.assertEqual(len(rows), 4)
        for epoch, (row, logs) in enumerate(zip(rows[1:], history)):
            self.assertEqual(int(row[0]), epoch)
            self.assertEqual(float(row[1]), logs["loss"])
            self.assertEqual(float(row[2]), logs["valid_loss"])

    def test_overwrites_or_appends(self):
        trainer = self.trainer(callbacks=[CSVLogger(self.path)])
        trainer.fit(self.loader(), epochs=2)
        trainer.fit(self.loader(), epochs=1)
        self.assertEqual(len(self.read()), 2)
        trainer.callbacks = [CSVLogger(self.path, separator=";", append=True)]
        trainer.fit(self.loader(), epochs=2)
        lines = self.path.read_text().splitlines()
        self.assertEqual(lines[0], "epoch,loss")
        self.assertEqual([line[:2] for line in lines[2:]], ["0;", "1;"])

    def test_missing_keys_and_arrays(self):
        feed = Feed([1.0, None], key="extra")

        class DropNone(Callback):
            def on_epoch_end(self, trainer, logs):
                if logs["extra"] is None:
                    del logs["extra"]

        trainer = self.trainer(
            metrics={"cm": ConfusionMatrix()},
            callbacks=[feed, DropNone(), CSVLogger(self.path)],
        )
        trainer.fit(self.loader(), self.loader(), epochs=2)
        rows = self.read()
        self.assertEqual(
            rows[0], ["epoch", "loss", "valid_loss", "valid_cm", "extra"]
        )
        cm = np.asarray(trainer.history[0]["valid_cm"]).tolist()
        self.assertEqual(rows[1][3], str(cm))
        self.assertEqual(rows[1][4], "1.0")
        self.assertEqual(rows[2][4], "NA")


class TestProgbarLogger(TrainerTestCase):
    def test_shows_progress(self):
        output = io.StringIO()
        progbar = ProgbarLogger(file=output, mininterval=0)
        trainer = self.trainer(callbacks=[progbar])
        trainer.fit(self.loader(), self.loader(self.data[:32]), epochs=2)
        trainer.evaluate(self.loader())
        trainer.predict(self.loader())
        text = output.getvalue()
        self.assertIn("Epoch 1/2", text)
        self.assertIn("Epoch 2/2", text)
        self.assertIn("6/6", text)
        self.assertIn("Evaluating", text)
        self.assertIn("Predicting", text)
        self.assertIn("valid_loss=", text)
        self.assertEqual(progbar._bars, [])

    def test_stopped_and_unsized_runs(self):
        stop = LambdaCallback(
            on_train_batch_end=lambda trainer, batch, logs: setattr(
                trainer, "should_stop", True
            )
        )
        progbar = ProgbarLogger(file=io.StringIO())
        trainer = self.trainer(callbacks=[progbar, stop])
        trainer.fit(self.loader(), epochs=3)
        self.assertEqual(progbar._bars, [])

        trainer.callbacks = [progbar]
        batches = torch.utils.data.DataLoader(
            _Unsized(self.data), batch_size=16
        )
        trainer.predict(batches)
        self.assertEqual(progbar._bars, [])


class _Unsized(torch.utils.data.IterableDataset):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)


if __name__ == "__main__":
    unittest.main()
