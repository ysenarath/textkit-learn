import json
import math
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from accelerate import Accelerator
from test_nn import Classifier, TrainerTestCase

from tklearn.metrics import Accuracy
from tklearn.nn import Trainer
from tklearn.nn.callbacks import (
    Callback,
    LambdaCallback,
    RunLogger,
    load_runs,
)
from tklearn.nn.callbacks.run_logger import default_run_name


class RunLoggerTestCase(TrainerTestCase):
    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def logger(self, name="run", **kwargs):
        return RunLogger(self.dir, name, **kwargs)

    def records(self, logger):
        lines = []
        for path in sorted(logger.run_dir.glob("events-*.jsonl")):
            lines += path.read_text().splitlines()
        return [json.loads(line) for line in lines]


class TestRunLogger(RunLoggerTestCase):
    def test_records_a_fit(self):
        logger = self.logger(log_every_n_steps=2, config={"batch_size": 16})
        trainer = self.trainer(metrics={"acc": Accuracy()}, callbacks=[logger])
        history = trainer.fit(self.loader(), self.loader(), epochs=2)
        records = self.records(logger)
        # 6 batches per epoch, one step each
        self.assertEqual(
            [r["event"] for r in records],
            [
                "fit_begin",
                *["step"] * 3,
                "epoch",
                *["step"] * 3,
                "epoch",
                "fit_end",
            ],
        )
        steps = [r for r in records if r["event"] == "step"]
        self.assertEqual([r["step"] for r in steps], [2, 4, 6, 8, 10, 12])
        self.assertEqual([r["epoch"] for r in steps], [0] * 3 + [1] * 3)
        epochs = [r for r in records if r["event"] == "epoch"]
        for record, logs in zip(epochs, history):
            self.assertEqual({k: record[k] for k in logs}, logs)
            self.assertGreater(record["epoch_time"], 0)
        begin, end = records[0], records[-1]
        self.assertEqual(begin["epochs"], 2)
        self.assertEqual(begin["num_batches"], 6)
        self.assertEqual(begin["attempt"], 0)
        self.assertEqual(end["step"], 12)
        self.assertFalse(end["stopped"])
        times = [r["time"] for r in records]
        self.assertEqual(times, sorted(times))

    def test_writes_config(self):
        logger = self.logger(config={"model": "linear", "batch_size": 16})
        trainer = self.trainer(
            lr_scheduler="linear", warmup=2, max_grad_norm=1.0
        )
        trainer.callbacks = [logger]
        trainer.fit(self.loader(), epochs=2)
        config = json.loads((logger.run_dir / "config.json").read_text())
        self.assertEqual(config["name"], "run")
        self.assertEqual(
            config["config"], {"model": "linear", "batch_size": 16}
        )
        self.assertEqual(config["model"]["parameters"], 4 * 2 + 2)
        self.assertEqual(config["model"]["trainable_parameters"], 10)
        self.assertEqual(config["optimizer"]["class"], "torch.optim.sgd.SGD")
        group = config["optimizer"]["param_groups"][0]
        # the warmup starts from 0
        self.assertEqual((group["initial_lr"], group["lr"]), (0.5, 0.0))
        self.assertEqual(config["trainer"]["epochs"], 2)
        self.assertEqual(config["trainer"]["lr_scheduler"], "linear")
        self.assertEqual(config["trainer"]["warmup"], 2)
        self.assertEqual(config["trainer"]["max_grad_norm"], 1.0)
        self.assertEqual(config["trainer"]["mixed_precision"], "no")
        self.assertEqual(
            config["trainer"]["callbacks"],
            ["tklearn.nn.callbacks.run_logger.RunLogger"],
        )
        self.assertIn("host", config["environment"])
        self.assertNotIn("SLURM_JOB_ID", config["environment"]["slurm"])
        # written by renaming a temporary file, which is gone
        self.assertEqual(
            sorted(
                p.name for p in logger.run_dir.iterdir() if "config" in p.name
            ),
            ["config.json"],
        )

    def test_step_records_mean_of_their_batches(self):
        losses = []
        record = LambdaCallback(
            on_train_batch_end=lambda trainer, batch, logs: losses.append(
                logs["loss"]
            )
        )
        logger = self.logger(log_every_n_steps=3)
        trainer = self.trainer(callbacks=[record, logger])
        trainer.fit(self.loader(), epochs=1)
        steps = [r for r in self.records(logger) if r["event"] == "step"]
        self.assertEqual(len(steps), 2)
        self.assertAlmostEqual(steps[0]["loss"], np.mean(losses[:3]))
        self.assertAlmostEqual(steps[1]["loss"], np.mean(losses[3:]))
        for step in steps:
            self.assertEqual(step["lr"], 0.5)
            self.assertGreater(step["step_time"], 0)
            self.assertGreater(step["memory_gb"], 0)

    def test_steps_with_gradient_accumulation(self):
        # 6 batches, 2 per step: each record covers 2 batches
        losses = []
        record = LambdaCallback(
            on_train_batch_end=lambda trainer, batch, logs: losses.append(
                logs["loss"]
            )
        )
        logger = self.logger(log_every_n_steps=1)
        trainer = self.trainer(accumulation=2, callbacks=[record, logger])
        trainer.fit(self.loader(), epochs=1)
        steps = [r for r in self.records(logger) if r["event"] == "step"]
        self.assertEqual([r["step"] for r in steps], [1, 2, 3])
        for i, step in enumerate(steps):
            self.assertAlmostEqual(
                step["loss"], np.mean(losses[2 * i : 2 * i + 2])
            )

    def test_learning_rate_of_each_group(self):
        model = Classifier()
        optimizer = torch.optim.SGD([
            {"params": [model.linear.weight], "lr": 0.1},
            {"params": [model.linear.bias], "lr": 0.2},
        ])
        logger = self.logger(log_every_n_steps=6)
        trainer = Trainer(
            model,
            optimizer,
            accelerator=Accelerator(cpu=True),
            callbacks=[logger],
        )
        trainer.fit(self.loader(), epochs=1)
        (step,) = [r for r in self.records(logger) if r["event"] == "step"]
        self.assertEqual((step["lr_0"], step["lr_1"]), (0.1, 0.2))
        self.assertNotIn("lr", step)

    def test_grad_norm(self):
        for max_grad_norm in (None, 1e-3):
            with self.subTest(max_grad_norm=max_grad_norm):
                expected = []

                def norm(trainer):
                    if max_grad_norm is not None:
                        # the trainer has clipped the gradients by now
                        expected.append(float(trainer.grad_norm))
                        return
                    grads = [p.grad for p in trainer.model.parameters()]
                    expected.append(
                        float(torch.cat([g.flatten() for g in grads]).norm())
                    )

                record = LambdaCallback(on_before_optimizer_step=norm)
                logger = self.logger(str(max_grad_norm), log_every_n_steps=2)
                trainer = self.trainer(
                    max_grad_norm=max_grad_norm, callbacks=[record, logger]
                )
                trainer.fit(self.loader(), epochs=1)
                steps = [
                    r for r in self.records(logger) if r["event"] == "step"
                ]
                for step, value in zip(steps, expected[1::2]):
                    self.assertAlmostEqual(step["grad_norm"], value, places=5)
                if max_grad_norm is not None:
                    # before clipping
                    for step in steps:
                        self.assertGreater(step["grad_norm"], max_grad_norm)

    def test_step_time_leaves_out_evaluation(self):
        logger = self.logger(log_every_n_steps=6)

        class SlowEvaluation(Callback):
            def on_test_end(self, trainer, logs):
                time.sleep(0.2)

        trainer = self.trainer(callbacks=[SlowEvaluation(), logger])
        trainer.fit(self.loader(), self.loader(), epochs=2)
        steps = [r for r in self.records(logger) if r["event"] == "step"]
        self.assertLess(steps[1]["step_time"] * 6, 0.2)

    def test_records_evaluate_outside_fit_only(self):
        logger = self.logger()
        trainer = self.trainer(metrics={"acc": Accuracy()}, callbacks=[logger])
        trainer.fit(self.loader(), self.loader(), epochs=1)
        results = trainer.evaluate(self.loader(), prefix="test_")
        events = [r for r in self.records(logger) if r["event"] == "evaluate"]
        self.assertEqual(len(events), 1)
        self.assertEqual({k: events[0][k] for k in results}, results)
        self.assertEqual(events[0]["step"], 6)

    def test_epochs_only(self):
        logger = self.logger(log_every_n_steps=0)
        self.trainer(callbacks=[logger]).fit(self.loader(), epochs=2)
        events = [r["event"] for r in self.records(logger)]
        self.assertNotIn("step", events)
        self.assertEqual(events.count("epoch"), 2)

    def test_non_finite_values(self):
        model = Classifier()
        model.training_step = lambda batch: torch.tensor(
            math.nan, requires_grad=True
        )
        logger = self.logger(log_every_n_steps=3)
        self.trainer(model, callbacks=[logger]).fit(self.loader())
        frame = load_runs(self.dir, event="step")
        self.assertTrue(frame["loss"].isna().all())

    def test_only_the_main_process_writes(self):
        logger = self.logger(log_every_n_steps=1)
        trainer = self.trainer(callbacks=[logger])
        with mock.patch.object(
            Accelerator, "is_main_process", new_callable=mock.PropertyMock
        ) as is_main_process:
            is_main_process.return_value = False
            trainer.fit(self.loader(), self.loader(), epochs=2)
            trainer.evaluate(self.loader())
        self.assertFalse(logger.run_dir.exists())

    def test_continues_a_run(self):
        # e.g. a requeued SLURM job with the same run name
        for epochs in (2, 1):
            logger = self.logger(log_every_n_steps=0)
            self.trainer(callbacks=[logger]).fit(self.loader(), epochs=epochs)
        events = [r["event"] for r in self.records(logger)]
        self.assertEqual(events.count("fit_begin"), 2)
        self.assertEqual(events.count("epoch"), 3)

    def test_records_the_slurm_job(self):
        env = {"SLURM_JOB_ID": "78", "SLURM_RESTART_COUNT": "2"}
        with mock.patch.dict(os.environ, env):
            logger = RunLogger(self.dir)
            self.trainer(callbacks=[logger]).fit(self.loader())
        self.assertEqual(logger.run_dir, self.dir / "78")
        config = json.loads((logger.run_dir / "config.json").read_text())
        slurm = config["environment"]["slurm"]
        self.assertEqual(slurm["SLURM_JOB_ID"], "78")
        self.assertEqual(self.records(logger)[0]["attempt"], 2)

    def test_validates_log_every_n_steps(self):
        for value in (-1, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "log_every_n_steps"):
                    RunLogger(self.dir, log_every_n_steps=value)


class TestRunName(unittest.TestCase):
    def test_default_names(self):
        cases = [
            (
                {
                    "SLURM_JOB_ID": "78",
                    "SLURM_ARRAY_JOB_ID": "77",
                    "SLURM_ARRAY_TASK_ID": "3",
                },
                "77_3",
            ),
            ({"SLURM_JOB_ID": "78"}, "78"),
        ]
        for env, name in cases:
            with self.subTest(name=name):
                with mock.patch.dict(os.environ, env):
                    self.assertEqual(default_run_name(), name)
        slurm = {k: v for k, v in os.environ.items() if k.startswith("SLURM")}
        with mock.patch.dict(os.environ, {}):
            for key in slurm:
                del os.environ[key]
            name = default_run_name()
        self.assertTrue(name.endswith(f"-{os.getpid()}"), name)


class TestLoadRuns(RunLoggerTestCase):
    def fit(self, name, lr, epochs=2):
        logger = self.logger(name, log_every_n_steps=3, config={"lr": lr})
        self.trainer(lr=lr, callbacks=[logger]).fit(
            self.loader(), epochs=epochs
        )
        return logger

    def test_reads_every_run(self):
        self.fit("sweep/a", 0.1)
        self.fit("sweep/b", 0.5, epochs=3)
        frame = load_runs(self.dir)
        self.assertEqual(list(frame["run"]), ["sweep/a"] * 2 + ["sweep/b"] * 3)
        self.assertEqual(list(frame["epoch"]), [0, 1, 0, 1, 2])
        self.assertEqual(list(frame["config.lr"]), [0.1] * 2 + [0.5] * 3)
        self.assertTrue((frame["event"] == "epoch").all())

        steps = load_runs(self.dir / "sweep", event="step", config=False)
        self.assertEqual(list(steps["run"]), ["a"] * 4 + ["b"] * 6)
        self.assertNotIn("config.lr", steps)
        everything = load_runs(self.dir, event=None)
        self.assertEqual(
            set(everything["event"]), {"fit_begin", "step", "epoch", "fit_end"}
        )

    def test_skips_a_cut_off_line(self):
        logger = self.fit("run", 0.1)
        (path,) = logger.run_dir.glob("events-*.jsonl")
        with open(path, "a") as f:
            f.write('{"event": "epoch", "epo')
        self.assertEqual(len(load_runs(self.dir)), 2)

    def test_empty_directory(self):
        self.assertTrue(load_runs(self.dir).empty)


if __name__ == "__main__":
    unittest.main()
