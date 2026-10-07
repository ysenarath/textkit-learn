import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from click.testing import CliRunner
from opentelemetry import trace
from opentelemetry.trace import StatusCode
from test_nn import TrainerTestCase

from tklearn.cli import main
from tklearn.metrics import Accuracy
from tklearn.nn.callbacks import OpenTelemetryCallback
from tklearn.tracing import FileTracerProvider
from tklearn.tracing.monitor import RunMonitor, snapshot


def without_slurm():
    patch = mock.patch.dict(os.environ, {})
    patch.start()
    for key in list(os.environ):
        if key.startswith("SLURM"):
            del os.environ[key]
    return patch


class MonitorTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.addCleanup(without_slurm().stop)

    def fit(self, name, *, end=None, epochs=4, batches=10, accumulation=2):
        """A fit as OpenTelemetryCallback records it: two epochs started,
        the first ended, and a steps span; `end` ends the fit: "ok",
        "stopped" or "error"."""
        provider = FileTracerProvider(self.dir, name)
        tracer = provider.get_tracer("test")
        fit = tracer.start_span(
            "fit",
            attributes={
                "trainer.epochs": epochs,
                "trainer.num_batches": batches,
                "trainer.gradient_accumulation_steps": accumulation,
            },
        )
        context = trace.set_span_in_context(fit)
        epoch = tracer.start_span("epoch", context, attributes={"epoch": 0})
        epoch.set_attributes({
            "step": 5,
            "loss": 0.5,
            "valid_f1": 0.7,
            "valid_cm": "[[1, 0], [0, 1]]",
        })
        epoch.end()
        tracer.start_span("epoch", context, attributes={"epoch": 1})
        steps = tracer.start_span("steps", context)
        steps.set_attributes({"step": 7, "loss": 0.4, "step_time": 0.5})
        steps.end()
        if end is not None:
            fit.set_attributes({"step": 7, "stopped": end == "stopped"})
            if end == "error":
                fit.set_status(StatusCode.ERROR, "failed")
            fit.end()
        return provider, fit


class TestRunMonitor(MonitorTestCase):
    def test_states(self):
        self.fit("set/running")
        self.fit("set/finished", end="ok")
        self.fit("stopped", end="stopped")
        self.fit("failed", end="error")
        provider = FileTracerProvider(self.dir, "evaluate-only")
        with provider.get_tracer("test").start_as_current_span("evaluate"):
            pass
        statuses = {s.run: s for s in snapshot(self.dir)}
        self.assertEqual(
            {run: s.status for run, s in statuses.items()},
            {
                "evaluate-only": "unknown",
                "failed": "failed",
                "set/finished": "finished",
                "set/running": "running",
                "stopped": "stopped",
            },
        )
        running = statuses["set/running"]
        self.assertEqual((running.epoch, running.epochs), (1, 4))
        # 4 epochs of 10 batches, 2 accumulated per step
        self.assertEqual((running.step, running.total_steps), (7, 20))
        self.assertAlmostEqual(running.progress, 0.35)
        self.assertEqual(running.loss, 0.4)
        self.assertEqual(running.step_time, 0.5)
        self.assertAlmostEqual(running.eta, (20 - 7) * 0.5)
        self.assertEqual(
            running.metrics,
            {"loss": 0.5, "valid_f1": 0.7, "valid_cm": "[[1, 0], [0, 1]]"},
        )
        self.assertLess(time.time() - running.last_time, 60)
        self.assertLessEqual(running.start_time, running.last_time)
        self.assertIsNone(statuses["set/finished"].eta)

    def test_stalled(self):
        self.fit("running")
        self.fit("finished", end="ok")
        monitor = RunMonitor(self.dir, stale_after=300)
        later = time.time() + 301
        statuses = {s.run: s for s in monitor.refresh(now=later)}
        self.assertEqual(statuses["running"].status, "stalled")
        self.assertIsNone(statuses["running"].eta)
        self.assertEqual(statuses["finished"].status, "finished")

    def test_reads_what_was_added(self):
        provider, fit = self.fit("run")
        monitor = RunMonitor(self.dir)
        (status,) = monitor.refresh()
        self.assertEqual(status.status, "running")
        # a record still being written is read once it is complete
        (path,) = provider.dir.glob("spans-*.jsonl")
        record = json.dumps({
            "type": "end",
            "trace_id": f"{fit.get_span_context().trace_id:032x}",
            "span_id": f"{fit.get_span_context().span_id:016x}",
            "name": "fit",
            "end_time": time.time_ns(),
            "status": "UNSET",
            "status_description": None,
            "attributes": {"step": 20, "stopped": False},
            "links": [],
        })
        with open(path, "a") as f:
            f.write(record[:40])
        (status,) = monitor.refresh()
        self.assertEqual(status.status, "running")
        with open(path, "a") as f:
            f.write(record[40:] + "\n")
        (status,) = monitor.refresh()
        self.assertEqual((status.status, status.step), ("finished", 20))

    def test_new_runs_and_files(self):
        self.fit("first")
        monitor = RunMonitor(self.dir, rescan_interval=3600)
        self.assertEqual([s.run for s in monitor.refresh()], ["first"])
        self.fit("second")
        # not looked for until the rescan interval has passed
        self.assertEqual([s.run for s in monitor.refresh()], ["first"])
        monitor.rescan_interval = 0
        self.assertEqual(
            [s.run for s in monitor.refresh()], ["first", "second"]
        )
        # a requeued job adds a file and a new fit, which is the latest
        self.fit("first", end="ok")
        statuses = {s.run: s.status for s in monitor.refresh()}
        self.assertEqual(statuses["first"], "finished")

    def test_pattern(self):
        for name in ("task-set-1/task-1-v1", "task-set-1/task-1-v2", "task-2"):
            self.fit(name)
        runs = [s.run for s in snapshot(self.dir, "task-set-1/*")]
        self.assertEqual(
            runs, ["task-set-1/task-1-v1", "task-set-1/task-1-v2"]
        )

    def test_slurm(self):
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "41"}):
            self.fit("timed-out")
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "42"}):
            self.fit("alive")
        output = "41|TIMEOUT\n41.batch|CANCELLED by 0\n42|RUNNING\n"
        result = subprocess.CompletedProcess([], 0, stdout=output)
        with mock.patch("subprocess.run", return_value=result) as run:
            statuses = {s.run: s for s in snapshot(self.dir, slurm=True)}
        self.assertIn("41,42", run.call_args.args[0])
        self.assertEqual(statuses["timed-out"].status, "failed")
        self.assertEqual(statuses["timed-out"].slurm_state, "TIMEOUT")
        self.assertEqual(statuses["alive"].status, "running")
        self.assertEqual(statuses["alive"].slurm_state, "RUNNING")
        # without sacct, the files alone decide
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            statuses = {s.run: s for s in snapshot(self.dir, slurm=True)}
        self.assertEqual(statuses["timed-out"].status, "running")
        self.assertIsNone(statuses["timed-out"].slurm_state)


class TestMonitorTraining(TrainerTestCase, MonitorTestCase):
    def setUp(self):
        TrainerTestCase.setUp(self)
        MonitorTestCase.setUp(self)

    def test_a_fit(self):
        provider = FileTracerProvider(self.dir, "run")
        callback = OpenTelemetryCallback(
            tracer_provider=provider, log_every_n_steps=4
        )
        trainer = self.trainer(
            metrics={"acc": Accuracy()}, callbacks=[callback]
        )
        history = trainer.fit(self.loader(), self.loader(), epochs=2)
        (status,) = snapshot(self.dir)
        self.assertEqual(status.status, "finished")
        self.assertEqual((status.epoch, status.epochs), (1, 2))
        self.assertEqual((status.step, status.total_steps), (12, 12))
        self.assertEqual(status.progress, 1.0)
        for key, value in history[-1].items():
            self.assertAlmostEqual(status.metrics[key], value)


class TestCommand(MonitorTestCase):
    def invoke(self, *args):
        runner = CliRunner()
        # wide enough that rich does not wrap the table
        return runner.invoke(main, list(args), env={"COLUMNS": "200"})

    def test_monitor_once(self):
        self.fit("task-set-1/task-1-v1")
        self.fit("task-2", end="ok")
        result = self.invoke("runs", "monitor", str(self.dir), "--once")
        self.assertEqual(result.exit_code, 0, result.output)
        output = result.output
        self.assertIn("2 runs: 1 finished, 1 running", output)
        self.assertIn(f"{self.dir} at", output)
        for text in ("task-set-1/task-1-v1", "task-2", "valid_f1", "2/4"):
            self.assertIn(text, output)
        # valid_cm is not a number, so not shown by default
        self.assertNotIn("valid_cm", output)
        self.assertIn("7/20", output)
        self.assertIn("35%", output)
        self.assertIn("6s", output)  # ETA of 13 steps of 0.5 s

    def test_options(self):
        self.fit("task-set-1/task-1-v1")
        self.fit("task-2")
        result = self.invoke(
            "runs",
            "monitor",
            str(self.dir),
            "--once",
            "--filter",
            "task-2",
            "-m",
            "loss",
            "--stale-after",
            "0",
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn("task-set-1", result.output)
        self.assertIn("stalled", result.output)
        self.assertNotIn("valid_f1", result.output)

    def test_live(self):
        self.fit("run")
        with mock.patch(
            "tklearn.cli.runs.time.sleep", side_effect=KeyboardInterrupt
        ):
            result = self.invoke("runs", "monitor", str(self.dir))
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("running", result.output)

    def test_errors_and_help(self):
        result = self.invoke("runs", "monitor", str(self.dir / "missing"))
        self.assertEqual(result.exit_code, 2)
        self.assertIn("does not exist", result.output)
        result = self.invoke("--version")
        self.assertIn("tklearn, version", result.output)

    def test_python_m_tklearn(self):
        result = subprocess.run(
            [sys.executable, "-m", "tklearn", "runs", "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("monitor", result.stdout)

    def test_help_does_not_import_tracing(self):
        code = (
            "import sys; from tklearn.cli import main; "
            "assert 'tklearn.tracing' not in sys.modules; "
            "assert 'torch' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    unittest.main()
