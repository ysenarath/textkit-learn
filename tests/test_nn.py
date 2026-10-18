import functools
import gc
import itertools
import json
import socket
import subprocess
import sys
import unittest
import weakref
from collections import namedtuple
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, PartialState
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.state import AcceleratorState, GradientState
from accelerate.utils import gather_object
from sklearn import metrics as sk
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, IterableDataset

from tklearn.metrics import F1, Accuracy, ConfusionMatrix, MetricCollection
from tklearn.nn import (
    Callback,
    Module,
    RunningMean,
    Session,
    Trainer,
    get_scheduler,
)
from tklearn.nn._outputs import BatchLogs
from tklearn.nn.schedules import warmup_steps

HOOKS = [name for name in vars(Callback) if name.startswith("on_")]

Pair = namedtuple("Pair", ["first", "second"])


def make_data(n=96, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=g)
    y = (x[:, 0] + x[:, 1] > 0).long()
    return [{"x": a, "labels": b} for a, b in zip(x, y)]


class Classifier(Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)

    def predict_step(self, batch):
        logits = self(batch["x"])
        outputs = {"y_pred": logits.argmax(-1), "y_score": logits.softmax(-1)}
        if "labels" in batch:
            outputs["y_true"] = batch["labels"]
            outputs["loss"] = F.cross_entropy(logits, batch["labels"])
        return outputs


class TermsClassifier(Classifier):
    # logs the terms of a combined loss
    def training_step(self, batch):
        ce = F.cross_entropy(self(batch["x"]), batch["labels"])
        l2 = self.linear.weight.pow(2).sum()
        return {"loss": ce + 0.01 * l2, "ce": ce, "l2": l2}


class SomeTermsClassifier(Classifier):
    # logs "odd" only on odd batches of 16 examples of INDEXED_DATA
    def training_step(self, batch):
        logs = {"loss": F.cross_entropy(self(batch["x"]), batch["labels"])}
        if batch["index"][0] // 16 % 2 == 1:
            logs["odd"] = torch.tensor(1.0)
        return logs


class FloatCounter(torch.Tensor):
    # counts the conversions to float, which wait for the device on a GPU
    reads = 0

    def __float__(self):
        FloatCounter.reads += 1
        return super().__float__()


class ProbeClassifier(Classifier):
    # logs a "probe" term that counts its conversions to float
    def training_step(self, batch):
        loss = F.cross_entropy(self(batch["x"]), batch["labels"])
        return {
            "loss": loss,
            "probe": torch.tensor(1.0).as_subclass(FloatCounter),
        }


class ExampleLossClassifier(Classifier):
    # returns one loss per example instead of the mean over the batch
    def predict_step(self, batch):
        outputs = super().predict_step(batch)
        if "loss" in outputs:
            outputs["loss"] = F.cross_entropy(
                self(batch["x"]), batch["labels"], reduction="none"
            )
        return outputs


INDEXED_DATA = [dict(d, index=i) for i, d in enumerate(make_data())]


class ListDataset(Dataset):
    # a dataset object (unlike a list, it can be weakly referenced)
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class EmptyIterable(IterableDataset):
    # an iterable dataset without a length, like a filtered stream
    def __iter__(self):
        return iter(())


class Stream(IterableDataset):
    # the items of a list, as an iterable dataset without a length
    def __init__(self, items):
        self.items = items

    def __iter__(self):
        return iter(self.items)


class OutputModel(Classifier):
    """Returns ``outputs(batch)`` from predict_step."""

    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs

    def predict_step(self, batch):
        return self.outputs(batch)


class Recorder(Callback):
    def __init__(self):
        self.calls = []
        self.batch_logs = []

    def on_train_batch_end(self, batch, logs):
        self.calls.append("on_train_batch_end")
        self.batch_logs.append(logs)


def _record(name):
    def hook(self, *args):
        self.calls.append(name)

    return hook


for _name in HOOKS:
    if _name != "on_train_batch_end":
        setattr(Recorder, _name, _record(_name))


class StopAt(Callback):
    """Stops training on the `count`-th call of `hook`."""

    def __init__(self, hook, count):
        self.hook, self.count = hook, count

    def on_epoch_end(self, logs):
        self._tick("on_epoch_end")

    def on_train_batch_end(self, batch, logs):
        self._tick("on_train_batch_end")

    def _tick(self, hook):
        if hook == self.hook:
            self.count -= 1
            if self.count == 0:
                self.session.should_stop = True


def skip_steps(*skipped):
    """Report the given optimizer step attempts (one-based) as skipped.

    fp16 skips a step whose gradients overflowed; this fakes it on CPU,
    where the step still runs. The trainer reads `step_was_skipped` once
    per attempt.
    """
    attempts = itertools.count(1)
    return mock.patch.object(
        AcceleratedOptimizer,
        "step_was_skipped",
        new_callable=mock.PropertyMock,
        side_effect=lambda: next(attempts) in skipped,
    )


class TrainerTestCase(unittest.TestCase):
    def setUp(self):
        # accelerate keeps process-wide state; start every test afresh
        AcceleratorState._reset_state(reset_partial_state=True)
        GradientState._reset_state()
        self.data = make_data()

    def loader(self, data=None, batch_size=16, **kwargs):
        return DataLoader(
            self.data if data is None else data,
            batch_size=batch_size,
            **kwargs,
        )

    def trainer(
        self,
        model=None,
        lr=0.5,
        accumulation=1,
        mixed_precision="no",
        **kwargs,
    ):
        model = Classifier() if model is None else model
        return Trainer(
            model,
            functools.partial(torch.optim.SGD, lr=lr),
            cpu=True,
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=accumulation,
            **kwargs,
        )


def fit_session(trainer, *args, **kwargs):
    """Fit, and return the session of the fit, to read after it."""
    sessions = []

    def new_state(trainer):
        sessions.append(Session(trainer))
        return sessions[-1]

    with mock.patch("tklearn.nn.trainer.Session", side_effect=new_state):
        trainer.fit(*args, **kwargs)
    # the first: sessions that callbacks open during the fit come after
    return sessions[0]


class TestFit(TrainerTestCase):
    def test_learns(self):
        trainer = self.trainer(metrics={"acc": Accuracy()})
        session = fit_session(
            trainer, self.loader(shuffle=True), self.loader(), epochs=5
        )
        history = session.history
        self.assertEqual(len(history), 5)
        self.assertEqual(list(history[0]), ["loss", "valid_loss", "valid_acc"])
        self.assertLess(history[-1]["loss"], history[0]["loss"])
        self.assertGreater(history[-1]["valid_acc"], 0.9)
        self.assertEqual(session.global_step, 5 * 6)
        self.assertEqual(session.epoch, 4)

    def test_trains_with_a_loss_per_example(self):
        # the default training_step averages it, so this trains exactly
        # like the model returning the mean over the batch
        expected = self.trainer().fit(self.loader(), epochs=2)
        history = self.trainer(ExampleLossClassifier()).fit(
            self.loader(), epochs=2
        )
        for logs, expected_logs in zip(history, expected):
            self.assertAlmostEqual(logs["loss"], expected_logs["loss"])

    def test_epoch_loss_is_mean_of_batch_losses(self):
        recorder = Recorder()
        trainer = self.trainer(callbacks=[recorder])
        history = trainer.fit(self.loader(), epochs=1)
        losses = [logs["loss"] for logs in recorder.batch_logs]
        self.assertEqual(len(losses), 6)
        self.assertAlmostEqual(history[0]["loss"], np.mean(losses))

    def test_logs_loss_terms(self):
        recorder = Recorder()
        trainer = self.trainer(TermsClassifier(), callbacks=[recorder])
        history = trainer.fit(self.loader(), epochs=2)
        self.assertEqual(list(history[0]), ["loss", "ce", "l2"])
        for logs in recorder.batch_logs:
            self.assertAlmostEqual(
                logs["loss"], logs["ce"] + 0.01 * logs["l2"], places=5
            )

    def test_terms_logged_on_some_batches(self):
        # a term is averaged over the batches that logged it
        trainer = self.trainer(SomeTermsClassifier())
        history = trainer.fit(self.loader(INDEXED_DATA), epochs=1)
        self.assertEqual(list(history[0]), ["loss", "odd"])
        self.assertAlmostEqual(history[0]["odd"], 1.0)

    def test_batch_logs_are_converted_when_read(self):
        FloatCounter.reads = 0
        history = self.trainer(ProbeClassifier()).fit(self.loader())
        self.assertEqual(FloatCounter.reads, 0)
        self.assertAlmostEqual(history[0]["probe"], 1.0)

        recorder = Recorder()
        self.trainer(ProbeClassifier(), callbacks=[recorder]).fit(
            self.loader()
        )
        self.assertEqual(FloatCounter.reads, 0)
        for logs in recorder.batch_logs:
            self.assertIs(type(logs["probe"]), float)
            self.assertIs(type(logs["loss"]), float)
            logs["probe"]  # converted once
        self.assertEqual(FloatCounter.reads, 6)
        with self.assertRaises(TypeError):
            recorder.batch_logs[0]["probe"] = 2.0

    def test_invalid_training_losses(self):
        cases = [
            (
                {"ce": torch.tensor(1.0, requires_grad=True)},
                "without a 'loss'",
            ),
            (torch.ones(2, requires_grad=True), "scalar tensor"),
            (
                {"loss": torch.tensor(1.0, requires_grad=True), "v": [1, 2]},
                "'v'",
            ),
        ]
        for loss, message in cases:
            with self.subTest(message=message):
                model = Classifier()
                model.training_step = lambda batch, loss=loss: loss
                with self.assertRaisesRegex(ValueError, message):
                    self.trainer(model).fit(self.loader())

    def test_needs_a_training_loss(self):
        model = OutputModel(lambda batch: batch["x"])
        with self.assertRaisesRegex(NotImplementedError, "training_step"):
            self.trainer(model).fit(self.loader())
        trainer = Trainer(Module(), cpu=True)
        with self.assertRaisesRegex(NotImplementedError, "predict_step"):
            trainer.predict(self.loader())

    def test_each_fit_is_a_new_session(self):
        trainer = self.trainer()
        first = fit_session(trainer, self.loader(), epochs=3)
        second = fit_session(trainer, self.loader(), epochs=1)
        self.assertEqual(len(second.history), 1)
        self.assertEqual(second.global_step, 6)
        # with an optimizer of its own, built by the factory
        self.assertIsNot(second.optimizer, first.optimizer)
        self.assertEqual(len(first.history), 3)
        self.assertEqual(first.global_step, 18)

    def test_gradient_accumulation_matches_larger_batches(self):
        accumulated = Classifier()
        trainer = self.trainer(accumulated, accumulation=2)
        session = fit_session(trainer, self.loader(batch_size=8), epochs=2)
        self.assertEqual(session.global_step, 2 * 6)

        AcceleratorState._reset_state(reset_partial_state=True)
        GradientState._reset_state()
        full = Classifier()
        self.trainer(full).fit(self.loader(batch_size=16), epochs=2)
        for a, b in zip(accumulated.parameters(), full.parameters()):
            torch.testing.assert_close(a, b)

    def test_steps_at_end_of_epoch_when_accumulating(self):
        # 5 batches with 2 per step: the last step has a single batch
        trainer = self.trainer(accumulation=2)
        session = fit_session(
            trainer, self.loader(self.data[:40], batch_size=8), epochs=2
        )
        self.assertEqual(session.global_step, 2 * 3)

    def test_another_trainer_does_not_change_the_accumulation(self):
        # accelerate keeps the accumulation steps process-wide, and every
        # accelerator sets its own; another trainer evaluating during fit
        # must not change this one's
        other = self.trainer()
        inner = self.loader(self.data[:16])

        class EvaluateOther(Callback):
            def on_train_batch_begin(self, batch):
                other.evaluate(inner)

        expected = Classifier()
        self.trainer(expected, accumulation=2).fit(
            self.loader(batch_size=8), epochs=2
        )
        model = Classifier()
        trainer = self.trainer(
            model, accumulation=2, callbacks=[EvaluateOther()]
        )
        session = fit_session(trainer, self.loader(batch_size=8), epochs=2)
        self.assertEqual(session.global_step, 2 * 6)
        for a, b in zip(model.parameters(), expected.parameters()):
            torch.testing.assert_close(a, b)

    def test_skipped_steps_are_not_counted(self):
        built = []

        def schedule(optimizer, num_training_steps):
            built.append(LambdaLR(optimizer, lambda step: 1.0))
            return built[0]

        trainer = self.trainer(lr_scheduler=schedule)
        with skip_steps(1, 8):
            session = fit_session(trainer, self.loader(), epochs=2)
        self.assertEqual(session.global_step, 2 * 6 - 2)
        self.assertEqual(built[0].last_epoch, session.global_step)

    def test_clips_gradients(self):
        norms = []

        class GradNorm(Callback):
            def on_before_optimizer_step(self):
                params = self.session.model.parameters()
                grads = torch.cat([p.grad.flatten() for p in params])
                norms.append(torch.linalg.vector_norm(grads))

        trainer = self.trainer(max_grad_norm=1e-3, callbacks=[GradNorm()])
        trainer.fit(self.loader(), epochs=1)
        self.assertEqual(len(norms), 6)
        for norm in norms:
            self.assertLessEqual(float(norm), 1e-3 + 1e-6)

    def test_empty_dataloader(self):
        # a named schedule would otherwise be built for 0 steps
        recorder = Recorder()
        trainer = self.trainer(lr_scheduler="linear", callbacks=[recorder])
        with mock.patch.object(AcceleratedOptimizer, "step") as step:
            with self.assertRaisesRegex(ValueError, "empty"):
                trainer.fit(self.loader([]), self.loader(), epochs=3)
        step.assert_not_called()
        # raised before on_train_begin, so no hook but on_exception
        self.assertEqual(recorder.calls, ["on_exception"])

    def test_empty_dataloader_without_a_length(self):
        # empty only once iterated; Accelerate dispatches the batches of an
        # iterable dataset, and raises when there are none
        recorder = Recorder()
        trainer = self.trainer(callbacks=[recorder])
        with self.assertRaises(ValueError):
            trainer.fit(self.loader(EmptyIterable()), self.loader(), epochs=3)
        self.assertNotIn("on_test_begin", recorder.calls)
        self.assertNotIn("on_epoch_end", recorder.calls)

    def test_mixed_precision(self):
        trainer = self.trainer(
            mixed_precision="bf16", metrics={"f1": F1(average="macro")}
        )
        history = trainer.fit(self.loader(), self.loader(), epochs=2)
        self.assertGreater(history[-1]["valid_f1"], 0.5)
        outputs = trainer.predict(self.loader())
        self.assertEqual(outputs["y_score"].dtype, torch.float32)


class TestRunningMean(unittest.TestCase):
    def setUp(self):
        AcceleratorState._reset_state(reset_partial_state=True)
        GradientState._reset_state()
        self.accelerator = Accelerator(cpu=True)

    def test_means_of_the_batches_that_logged_each_value(self):
        means = RunningMean(self.accelerator)
        means.update({"loss": torch.tensor(1.0), "odd": 4.0})
        means.update({"loss": torch.tensor(2.0)})
        means.update({"loss": torch.tensor(6.0), "odd": torch.tensor(2.0)})
        self.assertEqual(means.compute(), {"loss": 3.0, "odd": 3.0})

    def test_without_values(self):
        self.assertEqual(RunningMean(self.accelerator).compute(), {})

    def test_does_not_convert_the_values(self):
        FloatCounter.reads = 0
        means = RunningMean(self.accelerator)
        for _ in range(3):
            value = torch.tensor(2.0).as_subclass(FloatCounter)
            means.update({"probe": value})
        self.assertEqual(FloatCounter.reads, 0)
        self.assertEqual(means.compute(), {"probe": 2.0})

    def test_reads_batch_logs_without_converting_them(self):
        FloatCounter.reads = 0
        value = torch.tensor(5.0).as_subclass(FloatCounter)
        means = RunningMean(self.accelerator)
        means.update(BatchLogs({"probe": value}))
        self.assertEqual(FloatCounter.reads, 0)
        self.assertEqual(means.compute(), {"probe": 5.0})


class TestStopping(TrainerTestCase):
    def test_stops_after_epoch(self):
        trainer = self.trainer(callbacks=[StopAt("on_epoch_end", 2)])
        history = trainer.fit(self.loader(), epochs=5)
        self.assertEqual(len(history), 2)

    def test_stops_within_epoch(self):
        recorder = Recorder()
        trainer = self.trainer(
            callbacks=[StopAt("on_train_batch_end", 2), recorder]
        )
        session = fit_session(trainer, self.loader(), self.loader(), epochs=5)
        history = session.history
        self.assertEqual(len(history), 1)
        self.assertIn("valid_loss", history[0])
        self.assertEqual(session.global_step, 2)
        self.assertEqual(recorder.calls.count("on_train_batch_end"), 2)
        self.assertEqual(recorder.calls[-2:], ["on_epoch_end", "on_train_end"])

    def test_next_fit_starts_a_new_accumulation_window(self):
        # a fit that ends after 1 of 4 accumulated batches must not carry
        # that batch's gradients, or its place in the window, into the next

        class Interrupt(Callback):
            def on_train_batch_end(self, batch, logs):
                raise KeyboardInterrupt

        def stop(model):
            callbacks = [StopAt("on_train_batch_end", 1)]
            trainer = self.trainer(model, accumulation=4, callbacks=callbacks)
            trainer.fit(self.loader())

        def interrupt(model):
            callbacks = [Interrupt()]
            trainer = self.trainer(model, accumulation=4, callbacks=callbacks)
            with self.assertRaises(KeyboardInterrupt):
                trainer.fit(self.loader())

        expected = Classifier()
        self.trainer(expected, accumulation=4).fit(self.loader())
        for end in (stop, interrupt):
            with self.subTest(end=end.__name__):
                model = Classifier()
                end(model)
                self.trainer(model, accumulation=4).fit(self.loader())
                for a, b in zip(model.parameters(), expected.parameters()):
                    torch.testing.assert_close(a, b)


class TestCallbacks(TrainerTestCase):
    def test_hook_order(self):
        recorder = Recorder()
        trainer = self.trainer(callbacks=[recorder])
        loader = self.loader(self.data[:32])
        trainer.fit(loader, self.loader(self.data[:16]), epochs=2)
        train_batch = [
            "on_train_batch_begin",
            "on_before_optimizer_step",
            "on_train_batch_end",
        ]
        evaluate = [
            "on_test_begin",
            "on_test_batch_begin",
            "on_test_batch_end",
            "on_test_end",
        ]
        epoch = ["on_epoch_begin", *train_batch * 2, *evaluate, "on_epoch_end"]
        self.assertEqual(
            recorder.calls, ["on_train_begin", *epoch * 2, "on_train_end"]
        )

        recorder.calls = []
        trainer.predict(loader)
        self.assertEqual(
            recorder.calls,
            [
                "on_predict_begin",
                *["on_predict_batch_begin", "on_predict_batch_end"] * 2,
                "on_predict_end",
            ],
        )

    def test_hooks_get_the_trainer(self):
        seen = []

        class Epochs(Callback):
            def on_epoch_end(self, logs):
                seen.append((
                    self.session.epoch,
                    self.session.global_step,
                    logs,
                ))

        trainer = self.trainer(callbacks=[Epochs()])
        history = trainer.fit(self.loader(), epochs=2)
        self.assertEqual([s[:2] for s in seen], [(0, 6), (1, 12)])
        self.assertEqual([s[2] for s in seen], history)

    def test_callback_in_two_sessions_at_once(self):
        # a callback keeps the state of its session (e.g. the patience of
        # EarlyStopping), which the session of another trainer would change
        shared = Recorder()
        other = self.trainer(callbacks=[shared])
        loader = self.loader(self.data[:16])

        class EvaluateOther(Callback):
            def on_epoch_begin(self):
                other.evaluate(loader)

        trainer = self.trainer(callbacks=[shared, EvaluateOther()])
        with self.assertRaisesRegex(
            RuntimeError, "Recorder is in another session"
        ):
            trainer.fit(loader)
        self.assertNotIn("on_test_begin", shared.calls)
        # each session releases its callbacks when it ends, even when it
        # raises
        other.evaluate(loader)
        self.trainer(callbacks=[shared]).fit(loader)

    def test_callback_session(self):
        seen = []

        class Note(Callback):
            def on_train_begin(self):
                seen.append(("train", self.session))

            def on_test_begin(self):
                seen.append(("test", self.session))

            def on_exception(self, exception):
                seen.append(("exception", self.session))

        note = Note()
        with self.assertRaisesRegex(RuntimeError, "Note has a session only"):
            note.session
        trainer = self.trainer(callbacks=[note])
        trainer.fit(self.loader(), self.loader(), epochs=1)
        # one session for the fit and its evaluation, of this trainer
        (_, session), (_, evaluation) = seen
        self.assertEqual([name for name, _ in seen], ["train", "test"])
        self.assertIs(evaluation, session)
        self.assertIs(session.trainer, trainer)
        with self.assertRaisesRegex(RuntimeError, "Note has a session only"):
            note.session

        # also given back after a session that raised
        seen.clear()
        trainer = self.trainer(
            RaisingClassifier("predict_step", 1), callbacks=[note]
        )
        with self.assertRaises(RuntimeError):
            trainer.evaluate(self.loader())
        self.assertEqual([name for name, _ in seen], ["test", "exception"])
        self.assertIs(seen[0][1], seen[1][1])
        with self.assertRaisesRegex(RuntimeError, "Note has a session only"):
            note.session

    def test_callback_added_during_a_session(self):
        epochs = []

        class Read(Callback):
            def on_epoch_end(self, logs):
                epochs.append(self.session.epoch)

        read = Read()

        class Add(Callback):
            def on_train_begin(self):
                self.session.callbacks.append(read)

        trainer = self.trainer(callbacks=[Add()])
        trainer.fit(self.loader(), epochs=2)
        self.assertEqual(epochs, [0, 1])
        # given back at the end of the session, though not there at its
        # start
        with self.assertRaisesRegex(RuntimeError, "Read has a session only"):
            read.session


class RaisingClassifier(Classifier):
    """Raises `exception` on the `count`-th call of the hook `step`;
    ``"predict_step"`` counts the calls of evaluate and predict."""

    def __init__(self, step, count, exception=None):
        super().__init__()
        self.step, self.count = step, count
        self.exception = exception or RuntimeError("raised by the model")

    def training_step(self, batch):
        self._tick("training_step")
        return super().training_step(batch)

    def predict_step(self, batch):
        # the default training_step calls it too
        if not self.training:
            self._tick("predict_step")
        return super().predict_step(batch)

    def _tick(self, step):
        if step == self.step:
            self.count -= 1
            if self.count == 0:
                raise self.exception


class ExceptionRecorder(Recorder):
    """Also keeps the exceptions and accelerators `on_exception` gets."""

    def __init__(self):
        super().__init__()
        self.exceptions = []
        self.accelerators = []

    def on_exception(self, exception):
        self.calls.append("on_exception")
        self.exceptions.append(exception)
        self.accelerators.append(self.session.accelerator)


class TestExceptions(TrainerTestCase):
    def run_raising(self, model, run):
        recorder = ExceptionRecorder()
        trainer = self.trainer(model, callbacks=[recorder])
        with self.assertRaises(BaseException) as raised:
            run(trainer)
        self.assertIs(raised.exception, model.exception)
        self.assertEqual(recorder.exceptions, [model.exception])
        self.assertEqual(recorder.calls[-1], "on_exception")
        self.assertIsNotNone(recorder.accelerators[0])
        return recorder.calls

    def test_fit_that_raises(self):
        calls = self.run_raising(
            RaisingClassifier("training_step", 3),
            lambda t: t.fit(self.loader(), epochs=2),
        )
        train_batch = [
            "on_train_batch_begin",
            "on_before_optimizer_step",
            "on_train_batch_end",
        ]
        self.assertEqual(
            calls,
            [
                "on_train_begin",
                "on_epoch_begin",
                *train_batch * 2,
                "on_train_batch_begin",
                "on_exception",
            ],
        )

    def test_evaluation_in_fit_that_raises(self):
        # once, from fit, and no on_test_end, on_epoch_end or on_train_end
        calls = self.run_raising(
            RaisingClassifier("predict_step", 2),
            lambda t: t.fit(self.loader(), self.loader(), epochs=2),
        )
        self.assertEqual(
            calls[-6:],
            [
                "on_train_batch_end",
                "on_test_begin",
                "on_test_batch_begin",
                "on_test_batch_end",
                "on_test_batch_begin",
                "on_exception",
            ],
        )
        self.assertEqual(calls.count("on_exception"), 1)

    def test_evaluate_and_predict_that_raise(self):
        for name in ("evaluate", "predict"):
            with self.subTest(name=name):
                calls = self.run_raising(
                    RaisingClassifier("predict_step", 2),
                    lambda t, name=name: getattr(t, name)(self.loader()),
                )
                kind = "test" if name == "evaluate" else "predict"
                self.assertEqual(
                    calls,
                    [
                        f"on_{kind}_begin",
                        f"on_{kind}_batch_begin",
                        f"on_{kind}_batch_end",
                        f"on_{kind}_batch_begin",
                        "on_exception",
                    ],
                )

    def test_keyboard_interrupt(self):
        calls = self.run_raising(
            RaisingClassifier("training_step", 2, KeyboardInterrupt()),
            lambda t: t.fit(self.loader()),
        )
        self.assertEqual(calls.count("on_exception"), 1)

    def test_raised_before_any_begin_hook(self):
        recorder = ExceptionRecorder()
        trainer = self.trainer(callbacks=[recorder])
        with self.assertRaisesRegex(ValueError, "empty"):
            trainer.fit(self.loader([]))
        self.assertEqual(recorder.calls, ["on_exception"])

    def test_on_exception_that_raises(self):
        # the other callbacks still get it, and the session's exception
        # goes on
        class Broken(Callback):
            def on_exception(self, exception):
                raise OSError("cannot clean up")

        recorder = ExceptionRecorder()
        model = RaisingClassifier("training_step", 1)
        trainer = self.trainer(model, callbacks=[Broken(), recorder])
        with (
            self.assertLogs("tklearn.nn.trainer", "ERROR") as logs,
            self.assertRaises(RuntimeError) as raised,
        ):
            trainer.fit(self.loader())
        self.assertIs(raised.exception, model.exception)
        self.assertEqual(recorder.exceptions, [model.exception])
        self.assertIn("Broken.on_exception raised", logs.output[0])
        self.assertIn("cannot clean up", logs.output[0])

    def test_a_finished_session_does_not_call_it(self):
        recorder = ExceptionRecorder()
        trainer = self.trainer(callbacks=[recorder])
        trainer.fit(self.loader(), self.loader())
        trainer.predict(self.loader())
        self.assertNotIn("on_exception", recorder.calls)


class TestEvaluate(TrainerTestCase):
    def test_matches_sklearn(self):
        trainer = self.trainer(
            metrics={"f1": F1(average="macro"), "acc": Accuracy()}
        )
        trainer.fit(self.loader(), epochs=1)
        results = trainer.evaluate(self.loader(), prefix="test_")
        self.assertEqual(list(results), ["test_loss", "test_f1", "test_acc"])

        outputs = trainer.predict(self.loader())
        y_true, y_pred = outputs["y_true"].numpy(), outputs["y_pred"].numpy()
        self.assertAlmostEqual(
            results["test_f1"], sk.f1_score(y_true, y_pred, average="macro")
        )
        self.assertAlmostEqual(
            results["test_acc"], sk.accuracy_score(y_true, y_pred)
        )
        with torch.no_grad():
            losses = [
                trainer.model.predict_step(batch)["loss"]
                for batch in self.loader()
            ]
        self.assertAlmostEqual(
            results["test_loss"], float(torch.stack(losses).mean()), places=6
        )

    def test_loss_is_mean_over_examples(self):
        # batches of 40, 40 and 16: each batch loss counts once per example
        trainer = self.trainer()
        loader = self.loader(batch_size=40)
        with torch.no_grad():
            outputs = [trainer.model.predict_step(batch) for batch in loader]
        expected = sum(float(o["loss"]) * len(o["y_true"]) for o in outputs)
        self.assertAlmostEqual(
            trainer.evaluate(loader)["loss"], expected / 96, places=6
        )

    def test_loss_per_example(self):
        trainer = self.trainer(ExampleLossClassifier())
        loader = self.loader(batch_size=40)
        with torch.no_grad():
            losses = [trainer.model.predict_step(b)["loss"] for b in loader]
        self.assertAlmostEqual(
            trainer.evaluate(loader)["loss"],
            float(torch.cat(losses).mean()),
            places=6,
        )

    def test_invalid_losses(self):
        model = OutputModel(lambda batch: {"loss": torch.zeros(2, 3)})
        with self.assertRaisesRegex(ValueError, "per example"):
            self.trainer(model).evaluate(self.loader())
        # a scalar loss counts per example, which need a tensor to be counted
        model = OutputModel(lambda batch: {"loss": torch.tensor(1.0)})
        with self.assertRaisesRegex(ValueError, "count the examples"):
            self.trainer(model).evaluate(DataLoader(["a"] * 8, batch_size=4))

    def test_without_loss_or_metrics(self):
        model = OutputModel(lambda batch: {"y_true": batch["labels"]})
        self.assertEqual(self.trainer(model).evaluate(self.loader()), {})

    def test_restores_training_mode(self):
        trainer = self.trainer()
        for training in (True, False):
            trainer.model.train(training)
            trainer.evaluate(self.loader())
            trainer.predict(self.loader())
            self.assertEqual(trainer.model.training, training)

    def test_needs_mapping_outputs(self):
        model = OutputModel(lambda batch: batch["x"])
        with self.assertRaisesRegex(TypeError, "mapping"):
            self.trainer(model).evaluate(self.loader())

    def test_missing_metric_inputs(self):
        trainer = self.trainer(metrics={"acc": Accuracy()})
        unlabeled = [{"x": example["x"]} for example in self.data]
        with self.assertRaisesRegex(KeyError, "y_true"):
            trainer.evaluate(self.loader(unlabeled))

    def test_empty_dataloader(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            self.trainer().evaluate(self.loader([]))

    def test_prepares_each_dataloader_once_per_session(self):
        # so that the evaluation after each epoch reuses the persistent
        # workers of the one before
        loader = self.loader()
        prepared = []

        class Capture(Callback):
            def on_test_begin(self):
                prepared.append(self.session._prepare_dataloader(loader))

        trainer = self.trainer(callbacks=[Capture()])
        trainer.fit(self.loader(), loader, epochs=2)
        self.assertIsNot(prepared[0], loader)
        self.assertIs(prepared[0], prepared[1])

    def test_releases_dataloaders(self):
        # a trainer used on many temporary dataloaders must not keep them
        trainer = self.trainer()
        loader = self.loader(ListDataset(self.data))
        dataset = weakref.ref(loader.dataset)
        trainer.evaluate(loader)
        del loader
        gc.collect()
        self.assertIsNone(dataset())

    def test_nested_evaluate(self):
        # an evaluation started during another must not reset its metrics
        metrics = {"cm": ConfusionMatrix()}
        trainer = self.trainer(metrics=metrics)
        outer = self.loader(self.data[:48])
        inner = self.loader(self.data[48:64])
        expected = trainer.evaluate(outer)["cm"]

        class Nested(Callback):
            def __init__(self):
                self.done = False

            def on_test_batch_end(self, batch, outputs):
                if not self.done:
                    self.done = True
                    self.session.evaluate(inner)

        trainer = self.trainer(
            trainer.model, metrics=metrics, callbacks=[Nested()]
        )
        np.testing.assert_array_equal(trainer.evaluate(outer)["cm"], expected)

    def test_leaves_given_metrics_untouched(self):
        # metrics shared with other code keep their state, and that state
        # does not leak into the results
        f1 = F1(average="macro")
        f1.update([0, 1, 1], [0, 1, 0])
        before = f1.compute()
        trainer = self.trainer(metrics={"f1": f1})
        trainer.fit(self.loader(), self.loader())
        results = trainer.evaluate(self.loader())
        self.assertEqual(f1.compute(), before)
        fresh = {"f1": F1(average="macro")}
        self.assertEqual(
            results, trainer.evaluate(self.loader(), metrics=fresh)
        )

    def test_metrics_for_one_call(self):
        trainer = self.trainer(metrics={"acc": Accuracy()})
        results = trainer.evaluate(
            self.loader(), metrics={"cm": ConfusionMatrix()}
        )
        self.assertEqual(list(results), ["loss", "cm"])
        results = trainer.evaluate(self.loader(), metrics={})
        self.assertEqual(list(results), ["loss"])
        self.assertEqual(
            list(trainer.evaluate(self.loader())), ["loss", "acc"]
        )

    def test_metrics_are_a_collection(self):
        trainer = self.trainer(metrics={"acc": Accuracy()})
        self.assertIsInstance(trainer.metrics, MetricCollection)
        self.assertEqual(
            list(trainer.evaluate(self.loader())), ["loss", "acc"]
        )
        trainer = self.trainer()
        self.assertEqual(list(trainer.evaluate(self.loader())), ["loss"])
        with self.assertRaisesRegex(TypeError, "Metric"):
            self.trainer(metrics={"acc": "accuracy"})


class TestPredict(TrainerTestCase):
    def test_keeps_order_and_drops_loss(self):
        trainer = self.trainer()
        outputs = trainer.predict(self.loader(batch_size=10))
        self.assertEqual(set(outputs), {"y_true", "y_pred", "y_score"})
        self.assertEqual(outputs["y_score"].shape, (96, 2))
        self.assertEqual(
            outputs["y_true"].tolist(), [e["labels"].item() for e in self.data]
        )

    def test_structures(self):
        def outputs(batch):
            x = batch["x"]
            return Pair(
                first=(x, x[:, 0].numpy()),
                second={"ids": x[:, 0].tolist(), "none": None},
            )

        trainer = self.trainer(OutputModel(outputs))
        result = trainer.predict(self.loader(batch_size=10))
        x = torch.stack([e["x"] for e in self.data])
        self.assertIsInstance(result, Pair)
        torch.testing.assert_close(result.first[0], x)
        np.testing.assert_allclose(result.first[1], x[:, 0].numpy())
        self.assertEqual(result.second["ids"], x[:, 0].tolist())
        self.assertIsNone(result.second["none"])

    def test_pads_outputs_of_different_lengths(self):
        # per-token outputs of batches padded to lengths 5, 7 and 6
        lengths = [5, 7, 6]

        def outputs(batch):
            n = len(batch["x"])
            length = lengths[batch["index"][0].item() // 32]
            return {
                "labels": torch.ones(n, length, dtype=torch.long),
                "logits": torch.ones(n, length, 3),
                "mask": torch.ones(n, length, dtype=torch.bool),
                "array": np.ones((n, length), dtype=np.int32),
                "width": torch.ones(n, 2),
            }

        trainer = self.trainer(OutputModel(outputs))
        result = trainer.predict(self.loader(INDEXED_DATA, batch_size=32))
        self.assertEqual(result["labels"].shape, (96, 7))
        self.assertEqual(result["logits"].shape, (96, 7, 3))
        self.assertEqual(result["width"].shape, (96, 2))
        # the first batch, of length 5, is padded with 2 columns
        self.assertEqual(result["labels"][0].tolist(), [1] * 5 + [-100] * 2)
        self.assertEqual(result["logits"][0, 5:].tolist(), [[0.0] * 3] * 2)
        self.assertEqual(result["mask"][0].tolist(), [True] * 5 + [False] * 2)
        self.assertEqual(result["array"][0].tolist(), [1] * 5 + [-100] * 2)
        self.assertEqual(result["labels"][40].tolist(), [1] * 7)

    def test_rejects_outputs_with_different_dimensions(self):
        def outputs(batch):
            x = batch["x"]
            return {"x": x[:, :, None] if batch["index"][0] >= 80 else x}

        model = OutputModel(outputs)
        with self.assertRaisesRegex(ValueError, "number of dimensions"):
            self.trainer(model).predict(self.loader(INDEXED_DATA))

    def test_rejects_scalars(self):
        model = OutputModel(lambda batch: {"mean": batch["x"].mean()})
        with self.assertRaisesRegex(ValueError, "one value per example"):
            self.trainer(model).predict(self.loader())

    def test_rejects_a_missing_key(self):
        # the last batch has no labels, so the Classifier returns no y_true
        data = self.data[:80] + [{"x": e["x"]} for e in self.data[80:]]
        with self.assertRaisesRegex(ValueError, r"lacks \['y_true'\]"):
            self.trainer().predict(self.loader(data))

    def test_rejects_an_extra_key(self):
        def outputs(batch):
            x = batch["x"]
            return {"x": x, "z": x} if batch["index"][0] >= 80 else {"x": x}

        model = OutputModel(outputs)
        with self.assertRaisesRegex(ValueError, r"extra \['z'\]"):
            self.trainer(model).predict(self.loader(INDEXED_DATA))

    def test_rejects_tuples_of_different_lengths(self):
        def outputs(batch):
            x = batch["x"]
            return (x, x) if batch["index"][0] >= 80 else (x,)

        model = OutputModel(outputs)
        with self.assertRaisesRegex(ValueError, r"lengths \[1, 2\]"):
            self.trainer(model).predict(self.loader(INDEXED_DATA))

    def test_empty_dataloader(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            self.trainer().predict(self.loader([]))

    def test_without_optimizer(self):
        AcceleratorState._reset_state(reset_partial_state=True)
        trainer = Trainer(Classifier(), cpu=True)
        self.assertEqual(len(trainer.predict(self.loader())["y_pred"]), 96)
        with self.assertRaisesRegex(ValueError, "optimizer"):
            trainer.fit(self.loader())


class TestSchedules(TrainerTestCase):
    def test_named_schedule_spans_training(self):
        lrs = []

        class LearningRate(Callback):
            def on_before_optimizer_step(self):
                lrs.append(self.session.optimizer.param_groups[0]["lr"])

        trainer = self.trainer(
            lr=1.0,
            accumulation=2,
            lr_scheduler="linear",
            warmup=2,
            callbacks=[LearningRate()],
        )
        session = fit_session(
            trainer, self.loader(self.data[:40], batch_size=8), epochs=2
        )
        # 3 steps per epoch: warmup to 1.0 over 2 steps, then decay to 0
        np.testing.assert_allclose(lrs, [0.0, 0.5, 1.0, 0.75, 0.5, 0.25])
        self.assertEqual(session.optimizer.param_groups[0]["lr"], 0.0)

    def test_callable_schedule_gets_step_count(self):
        built = []

        def schedule(optimizer, num_training_steps):
            built.append(LambdaLR(optimizer, lambda step: 1.0))
            built.append(num_training_steps)
            return built[0]

        trainer = self.trainer(accumulation=4, lr_scheduler=schedule)
        trainer.fit(self.loader(), epochs=3)  # 6 batches, 2 steps each
        scheduler, num_steps = built
        self.assertEqual(num_steps, 3 * 2)
        self.assertEqual(scheduler.last_epoch, num_steps)

    def test_named_schedule_needs_a_length(self):
        trainer = self.trainer(lr_scheduler="linear")
        with self.assertRaisesRegex(ValueError, "length"):
            trainer.fit(self.loader(Stream(self.data)))

    def test_overridden_schedule(self):
        # without an lr_scheduler, and without a step count when the
        # dataloader has no length
        num_steps = []

        class Custom(Trainer):
            def create_scheduler(self, optimizer, num_training_steps):
                num_steps.append(num_training_steps)
                return LambdaLR(optimizer, lambda step: 0.5)

        trainer = Custom(
            Classifier(),
            functools.partial(torch.optim.SGD, lr=1.0),
            cpu=True,
            mixed_precision="no",
        )
        session = fit_session(trainer, self.loader(), epochs=2)
        self.assertIsInstance(session.scheduler, LambdaLR)
        self.assertEqual(session.optimizer.param_groups[0]["lr"], 0.5)
        trainer.fit(self.loader(Stream(self.data)))
        self.assertEqual(num_steps, [2 * 6, None])

    def test_get_scheduler_warmup_fraction(self):
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=1.0)
        scheduler = get_scheduler("linear", optimizer, 10, warmup=0.25)
        lrs = []
        for _ in range(4):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        np.testing.assert_allclose(lrs, [0.0, 1 / 3, 2 / 3, 1.0])

    def test_warmup_steps(self):
        self.assertEqual(warmup_steps(5, 100), 5)
        self.assertEqual(warmup_steps(0.1, 95), 10)
        for warmup, error in [
            (True, TypeError),
            ("1", TypeError),
            (-1, ValueError),
            (1.5, ValueError),
        ]:
            with self.subTest(warmup=warmup):
                with self.assertRaises(error):
                    warmup_steps(warmup, 10)


class TestValidation(TrainerTestCase):
    def test_constructor(self):
        model = Classifier()
        optimizer = functools.partial(torch.optim.SGD, lr=0.1)
        # instances, which a session would change, instead of factories
        sgd = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = LambdaLR(sgd, lambda step: 1.0)
        cases = [
            (dict(model=torch.nn.Linear(1, 1)), TypeError, "Module"),
            (dict(optimizer=sgd), TypeError, "functools.partial"),
            (dict(optimizer=0.1), TypeError, "functools.partial"),
            (
                dict(optimizer=optimizer, lr_scheduler=scheduler),
                TypeError,
                "num_training_steps",
            ),
            (dict(lr_scheduler="linear"), ValueError, "optimizer"),
            (dict(optimizer=optimizer, warmup=2), ValueError, "named"),
            (dict(callbacks=[object()]), TypeError, "Callback"),
            (
                dict(gradient_accumulation_steps=0),
                ValueError,
                "gradient_accumulation_steps",
            ),
            (
                dict(gradient_accumulation_steps=2.0),
                ValueError,
                "gradient_accumulation_steps",
            ),
        ]
        for kwargs, error, message in cases:
            with self.subTest(kwargs=kwargs):
                kwargs.setdefault("model", model)
                with self.assertRaisesRegex(error, message):
                    Trainer(**kwargs)

    def test_accelerator_exists_during_a_session(self):
        seen = []

        class Capture(Callback):
            def on_train_begin(self):
                seen.append(self.session.accelerator)

            def on_test_begin(self):
                seen.append(self.session.accelerator)

        trainer = self.trainer(callbacks=[Capture()])
        session = fit_session(trainer, self.loader(), self.loader())
        trainer.evaluate(self.loader())
        with self.assertRaisesRegex(RuntimeError, "during a fit"):
            session.accelerator
        # the evaluation inside fit shares its accelerator; evaluate gets one
        self.assertIs(seen[0], seen[1])
        self.assertIsNot(seen[1], seen[2])

    def test_fit_needs_an_optimizer(self):
        trainer = Trainer(Classifier(), cpu=True, mixed_precision="no")
        with self.assertRaisesRegex(ValueError, "optimizer"):
            trainer.fit(self.loader())
        # evaluate and predict do not
        trainer.evaluate(self.loader())
        trainer.predict(self.loader())

    def test_overridden_optimizer(self):
        # parameter groups, without an optimizer factory

        class Grouped(Trainer):
            def create_optimizer(self):
                params = dict(self.model.named_parameters())
                return torch.optim.SGD(
                    [
                        {"params": [params["linear.weight"]]},
                        {"params": [params["linear.bias"]], "lr": 0.0},
                    ],
                    lr=0.5,
                )

        model = Classifier()
        weight, bias = model.linear.weight.clone(), model.linear.bias.clone()
        trainer = Grouped(model, cpu=True, mixed_precision="no")
        session = fit_session(trainer, self.loader())
        self.assertEqual(len(session.optimizer.param_groups), 2)
        self.assertFalse(torch.equal(model.linear.weight, weight))
        torch.testing.assert_close(model.linear.bias, bias)

    def test_sessions_use_create_accelerator(self):
        created, seen = [], []

        class Custom(Trainer):
            def create_accelerator(self):
                created.append(super().create_accelerator())
                return created[-1]

        class Capture(Callback):
            def on_train_begin(self):
                seen.append(self.session.accelerator)

            def on_test_begin(self):
                seen.append(self.session.accelerator)

            def on_predict_begin(self):
                seen.append(self.session.accelerator)

        trainer = Custom(
            Classifier(),
            functools.partial(torch.optim.SGD, lr=0.5),
            callbacks=[Capture()],
            cpu=True,
            mixed_precision="no",
        )
        trainer.fit(self.loader())
        trainer.evaluate(self.loader())
        trainer.predict(self.loader())
        self.assertEqual((len(created), len(seen)), (3, 3))
        for a, b in zip(seen, created):
            self.assertIs(a, b)

    def test_settings_are_fixed(self):
        # validated when the trainer is created
        trainer = self.trainer(callbacks=[Recorder()])
        replacements = {
            "model": Classifier(),
            "metrics": {"acc": Accuracy()},
            "callbacks": [Recorder()],
            "max_grad_norm": 1.0,
            "mixed_precision": "bf16",
            "gradient_accumulation_steps": 2,
            "cpu": False,
        }
        # a tuple, so that the list cannot be changed in place either
        self.assertIsInstance(trainer.callbacks, tuple)
        for used in (False, True):
            if used:
                trainer.fit(self.loader())
            for name, value in replacements.items():
                with self.subTest(name=name, used=used):
                    before = getattr(trainer, name)
                    with self.assertRaises(AttributeError):
                        setattr(trainer, name, value)
                    self.assertIs(getattr(trainer, name), before)

    def test_fit_arguments(self):
        trainer = self.trainer()
        with self.assertRaisesRegex(TypeError, "DataLoader"):
            trainer.fit(self.data)
        with self.assertRaisesRegex(ValueError, "epochs"):
            trainer.fit(self.loader(), epochs=0)

    def test_module_device(self):
        self.assertEqual(Module().device, torch.device("cpu"))
        self.assertEqual(Classifier().device, torch.device("cpu"))


class TestMultipleProcesses(unittest.TestCase):
    # runs this file as two CPU processes; see `_run_on_two_processes`
    def run_on_two_processes(self, scenario):
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            # --standalone hangs where the hostname does not resolve
            "--master_addr=127.0.0.1",
            f"--master_port={_free_port()}",
            "--nproc_per_node=2",
            __file__,
            f"--two-processes={scenario}",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )
        except subprocess.TimeoutExpired:
            self.fail(f"{scenario} on two processes did not finish")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [
            line.removeprefix("outputs: ")
            for line in result.stdout.splitlines()
            if line.startswith("outputs: ")
        ]
        self.assertEqual(len(lines), 1, result.stdout)
        outputs = json.loads(lines[0])
        self.assertEqual(len(outputs), 2, result.stdout)
        return outputs

    def test_terms_logged_on_some_processes(self):
        # batches 0, 2, 4 go to rank 0 and 1, 3, 5 to rank 1, so only rank 1
        # logs "odd"; both must reduce the same keys or one hangs
        histories = self.run_on_two_processes("fit")
        for history in histories:
            self.assertEqual(list(history[0]), ["loss", "odd"])
            self.assertAlmostEqual(history[0]["odd"], 1.0)
        self.assertEqual(histories[0], histories[1])

    def test_stop_on_one_process(self):
        # rank 0 stops after its first batch; rank 1 must stop with it, or
        # it trains alone and hangs in the reduce of the epoch's logs
        for steps in self.run_on_two_processes("stop"):
            self.assertEqual(steps, 1)

    def test_evaluate_leaves_out_padding(self):
        # 40 examples in batches of 16: Accelerate pads the last batch with
        # duplicates, which must not count in the loss
        AcceleratorState._reset_state(reset_partial_state=True)
        GradientState._reset_state()
        trainer = Trainer(ExampleLossClassifier(), cpu=True)
        loader = DataLoader(INDEXED_DATA[:40], batch_size=16)
        expected = trainer.evaluate(loader)["loss"]
        for loss in self.run_on_two_processes("evaluate"):
            self.assertAlmostEqual(loss, expected, places=6)


class StopOnMainProcess(Callback):
    def on_train_batch_end(self, batch, logs):
        if self.session.accelerator.is_main_process:
            self.session.should_stop = True


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_on_two_processes(scenario):
    state = PartialState(cpu=True)
    assert state.num_processes == 2, state.num_processes
    if scenario == "evaluate":
        trainer = Trainer(ExampleLossClassifier(), cpu=True)
        loader = DataLoader(INDEXED_DATA[:40], batch_size=16)
        output = trainer.evaluate(loader)["loss"]
    else:
        model = SomeTermsClassifier()
        optimizer = functools.partial(torch.optim.SGD, lr=0.5)
        callbacks = [StopOnMainProcess()] if scenario == "stop" else []
        trainer = Trainer(model, optimizer, callbacks=callbacks, cpu=True)
        loader = DataLoader(INDEXED_DATA, batch_size=16)
        session = fit_session(trainer, loader, epochs=2)
        output = (
            session.global_step if scenario == "stop" else session.history[:1]
        )
    # print from one process, as lines printed at once can interleave
    outputs = gather_object([output])
    if state.is_main_process:
        print("outputs: " + json.dumps(outputs), flush=True)


if __name__ == "__main__":
    scenarios = [a for a in sys.argv if a.startswith("--two-processes=")]
    if scenarios:
        _run_on_two_processes(scenarios[0].removeprefix("--two-processes="))
    else:
        unittest.main()
