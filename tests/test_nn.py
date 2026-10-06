import json
import socket
import subprocess
import sys
import unittest
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.state import AcceleratorState, GradientState
from accelerate.utils import gather_object
from sklearn import metrics as sk
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from tklearn.metrics import F1, Accuracy, ConfusionMatrix, MetricCollection
from tklearn.nn import Callback, Module, Trainer, get_scheduler
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


INDEXED_DATA = [dict(d, index=i) for i, d in enumerate(make_data())]


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

    def on_train_batch_end(self, trainer, batch, logs):
        self.calls.append("on_train_batch_end")
        self.batch_logs.append(logs)


def _record(name):
    def hook(self, trainer, *args):
        self.calls.append(name)

    return hook


for _name in HOOKS:
    if _name != "on_train_batch_end":
        setattr(Recorder, _name, _record(_name))


class StopAt(Callback):
    """Stops training on the `count`-th call of `hook`."""

    def __init__(self, hook, count):
        self.hook, self.count = hook, count

    def on_epoch_end(self, trainer, logs):
        self._tick(trainer, "on_epoch_end")

    def on_train_batch_end(self, trainer, batch, logs):
        self._tick(trainer, "on_train_batch_end")

    def _tick(self, trainer, hook):
        if hook == self.hook:
            self.count -= 1
            if self.count == 0:
                trainer.should_stop = True


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
        accelerator = Accelerator(
            cpu=True,
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=accumulation,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        return Trainer(model, optimizer, accelerator=accelerator, **kwargs)


class TestFit(TrainerTestCase):
    def test_learns(self):
        trainer = self.trainer(metrics={"acc": Accuracy()})
        history = trainer.fit(
            self.loader(shuffle=True), self.loader(), epochs=5
        )
        self.assertIs(history, trainer.history)
        self.assertEqual(len(history), 5)
        self.assertEqual(list(history[0]), ["loss", "valid_loss", "valid_acc"])
        self.assertLess(history[-1]["loss"], history[0]["loss"])
        self.assertGreater(history[-1]["valid_acc"], 0.9)
        self.assertEqual(trainer.global_step, 5 * 6)
        self.assertEqual(trainer.epoch, 4)

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
        trainer = Trainer(Module(), accelerator=Accelerator(cpu=True))
        with self.assertRaisesRegex(NotImplementedError, "predict_step"):
            trainer.predict(self.loader())

    def test_history_restarts(self):
        trainer = self.trainer()
        trainer.fit(self.loader(), epochs=3)
        trainer.fit(self.loader(), epochs=1)
        self.assertEqual(len(trainer.history), 1)
        self.assertEqual(trainer.global_step, 6)

    def test_gradient_accumulation_matches_larger_batches(self):
        accumulated = Classifier()
        trainer = self.trainer(accumulated, accumulation=2)
        trainer.fit(self.loader(batch_size=8), epochs=2)
        self.assertEqual(trainer.global_step, 2 * 6)

        AcceleratorState._reset_state(reset_partial_state=True)
        GradientState._reset_state()
        full = Classifier()
        self.trainer(full).fit(self.loader(batch_size=16), epochs=2)
        for a, b in zip(accumulated.parameters(), full.parameters()):
            torch.testing.assert_close(a, b)

    def test_steps_at_end_of_epoch_when_accumulating(self):
        # 5 batches with 2 per step: the last step has a single batch
        trainer = self.trainer(accumulation=2)
        trainer.fit(self.loader(self.data[:40], batch_size=8), epochs=2)
        self.assertEqual(trainer.global_step, 2 * 3)

    def test_clips_gradients(self):
        norms = []

        class GradNorm(Callback):
            def on_before_optimizer_step(self, trainer):
                params = trainer.model.parameters()
                grads = torch.cat([p.grad.flatten() for p in params])
                norms.append(torch.linalg.vector_norm(grads))

        trainer = self.trainer(max_grad_norm=1e-3, callbacks=[GradNorm()])
        trainer.fit(self.loader(), epochs=1)
        self.assertEqual(len(norms), 6)
        for norm in norms:
            self.assertLessEqual(float(norm), 1e-3 + 1e-6)

    def test_mixed_precision(self):
        trainer = self.trainer(
            mixed_precision="bf16", metrics={"f1": F1(average="macro")}
        )
        history = trainer.fit(self.loader(), self.loader(), epochs=2)
        self.assertGreater(history[-1]["valid_f1"], 0.5)
        outputs = trainer.predict(self.loader())
        self.assertEqual(outputs["y_score"].dtype, torch.float32)


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
        history = trainer.fit(self.loader(), self.loader(), epochs=5)
        self.assertEqual(len(history), 1)
        self.assertIn("valid_loss", history[0])
        self.assertEqual(trainer.global_step, 2)
        self.assertEqual(recorder.calls.count("on_train_batch_end"), 2)
        self.assertEqual(recorder.calls[-2:], ["on_epoch_end", "on_fit_end"])

    def test_next_fit_starts_a_new_accumulation_window(self):
        # a fit that ends after 1 of 4 accumulated batches must not carry
        # that batch's gradients, or its place in the window, into the next

        class Interrupt(Callback):
            def on_train_batch_end(self, trainer, batch, logs):
                raise KeyboardInterrupt

        def stop(trainer):
            trainer.callbacks = [StopAt("on_train_batch_end", 1)]
            trainer.fit(self.loader())

        def interrupt(trainer):
            trainer.callbacks = [Interrupt()]
            with self.assertRaises(KeyboardInterrupt):
                trainer.fit(self.loader())

        expected = Classifier()
        self.trainer(expected, accumulation=4).fit(self.loader())
        for end in (stop, interrupt):
            with self.subTest(end=end.__name__):
                model = Classifier()
                trainer = self.trainer(model, accumulation=4)
                end(trainer)
                trainer.callbacks = []
                trainer.fit(self.loader())
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
            "on_evaluate_begin",
            "on_evaluate_batch_begin",
            "on_evaluate_batch_end",
            "on_evaluate_end",
        ]
        epoch = ["on_epoch_begin", *train_batch * 2, *evaluate, "on_epoch_end"]
        self.assertEqual(
            recorder.calls, ["on_fit_begin", *epoch * 2, "on_fit_end"]
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
            def on_epoch_end(self, trainer, logs):
                seen.append((trainer.epoch, trainer.global_step, logs))

        trainer = self.trainer(callbacks=[Epochs()])
        trainer.fit(self.loader(), epochs=2)
        self.assertEqual([s[:2] for s in seen], [(0, 6), (1, 12)])
        self.assertEqual([s[2] for s in seen], trainer.history)


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

    def test_prepares_each_dataloader_once(self):
        trainer = self.trainer()
        loader = self.loader()
        trainer.evaluate(loader)
        trainer.evaluate(loader)
        self.assertEqual(len(trainer.accelerator._dataloaders), 1)

    def test_nested_evaluate(self):
        # an evaluation started during another must not reset its metrics
        trainer = self.trainer(metrics={"cm": ConfusionMatrix()})
        outer = self.loader(self.data[:48])
        inner = self.loader(self.data[48:64])
        expected = trainer.evaluate(outer)["cm"]

        class Nested(Callback):
            def __init__(self):
                self.done = False

            def on_evaluate_batch_end(self, trainer, batch, outputs):
                if not self.done:
                    self.done = True
                    trainer.evaluate(inner)

        trainer.callbacks = [Nested()]
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

    def test_assigning_metrics(self):
        trainer = self.trainer()
        trainer.metrics = {"acc": Accuracy()}
        self.assertIsInstance(trainer.metrics, MetricCollection)
        self.assertEqual(
            list(trainer.evaluate(self.loader())), ["loss", "acc"]
        )
        trainer.metrics = None
        self.assertEqual(list(trainer.evaluate(self.loader())), ["loss"])
        with self.assertRaisesRegex(TypeError, "Metric"):
            trainer.metrics = {"acc": "accuracy"}


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

    def test_rejects_scalars(self):
        model = OutputModel(lambda batch: {"mean": batch["x"].mean()})
        with self.assertRaisesRegex(ValueError, "one value per example"):
            self.trainer(model).predict(self.loader())

    def test_empty_dataloader(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            self.trainer().predict(self.loader([]))

    def test_without_optimizer(self):
        AcceleratorState._reset_state(reset_partial_state=True)
        trainer = Trainer(Classifier(), accelerator=Accelerator(cpu=True))
        self.assertEqual(len(trainer.predict(self.loader())["y_pred"]), 96)
        with self.assertRaisesRegex(ValueError, "optimizer"):
            trainer.fit(self.loader())


class TestSchedules(TrainerTestCase):
    def test_named_schedule_spans_training(self):
        lrs = []

        class LearningRate(Callback):
            def on_before_optimizer_step(self, trainer):
                lrs.append(trainer.optimizer.param_groups[0]["lr"])

        trainer = self.trainer(
            lr=1.0,
            accumulation=2,
            lr_scheduler="linear",
            warmup=2,
            callbacks=[LearningRate()],
        )
        trainer.fit(self.loader(self.data[:40], batch_size=8), epochs=2)
        # 3 steps per epoch: warmup to 1.0 over 2 steps, then decay to 0
        np.testing.assert_allclose(lrs, [0.0, 0.5, 1.0, 0.75, 0.5, 0.25])
        self.assertEqual(trainer.optimizer.param_groups[0]["lr"], 0.0)

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
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        accelerator = Accelerator(cpu=True)
        cases = [
            (dict(model=torch.nn.Linear(1, 1)), TypeError, "Module"),
            (dict(lr_scheduler="linear"), ValueError, "optimizer"),
            (dict(optimizer=optimizer, warmup=2), ValueError, "named"),
            (dict(callbacks=[object()]), TypeError, "Callback"),
            (
                dict(accelerator=accelerator, mixed_precision="bf16"),
                ValueError,
                "accelerator",
            ),
        ]
        for kwargs, error, message in cases:
            with self.subTest(kwargs=kwargs):
                kwargs.setdefault("model", model)
                with self.assertRaisesRegex(error, message):
                    Trainer(**kwargs)

    def test_model_optimizer_and_accelerator_are_fixed(self):
        # the accelerator wraps them on first use and the trainer keeps the
        # wrapped versions, so a replacement would be silently ignored
        trainer = self.trainer()
        model = Classifier()
        replacements = {
            "model": model,
            "optimizer": torch.optim.SGD(model.parameters(), lr=0.1),
            "accelerator": Accelerator(cpu=True),
        }
        for used in (False, True):
            if used:
                trainer.fit(self.loader())
            for name, value in replacements.items():
                with self.subTest(name=name, used=used):
                    before = getattr(trainer, name)
                    with self.assertRaisesRegex(AttributeError, "new Trainer"):
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


class StopOnMainProcess(Callback):
    def on_train_batch_end(self, trainer, batch, logs):
        if trainer.accelerator.is_main_process:
            trainer.should_stop = True


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_on_two_processes(scenario):
    accelerator = Accelerator(cpu=True)
    assert accelerator.num_processes == 2, accelerator.num_processes
    model = SomeTermsClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    callbacks = [StopOnMainProcess()] if scenario == "stop" else []
    trainer = Trainer(
        model, optimizer, callbacks=callbacks, accelerator=accelerator
    )
    history = trainer.fit(DataLoader(INDEXED_DATA, batch_size=16), epochs=2)
    output = trainer.global_step if scenario == "stop" else history[:1]
    # print from one process, as lines printed at once can interleave
    outputs = gather_object([output])
    if accelerator.is_main_process:
        print("outputs: " + json.dumps(outputs), flush=True)


if __name__ == "__main__":
    scenarios = [a for a in sys.argv if a.startswith("--two-processes=")]
    if scenarios:
        _run_on_two_processes(scenarios[0].removeprefix("--two-processes="))
    else:
        unittest.main()
