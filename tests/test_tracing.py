import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from opentelemetry import trace
from opentelemetry.trace import Link, StatusCode
from test_nn import TrainerTestCase

from tklearn.metrics import Accuracy, ConfusionMatrix
from tklearn.nn.callbacks import OpenTelemetryCallback
from tklearn.tracing import (
    FileTracerProvider,
    default_run_name,
    load_events,
    load_spans,
)


def without_slurm():
    """Patch the environment to look like no SLURM job."""
    patch = mock.patch.dict(os.environ, {})
    patch.start()
    for key in list(os.environ):
        if key.startswith("SLURM"):
            del os.environ[key]
    return patch


class TracingTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.addCleanup(without_slurm().stop)

    def provider(self, name="run", **kwargs):
        return FileTracerProvider(self.dir, name, **kwargs)

    def records(self, provider):
        lines = []
        for path in sorted(provider.dir.glob("spans-*.jsonl")):
            lines += path.read_text().splitlines()
        return [json.loads(line) for line in lines]


class TestFileTracerProvider(TracingTestCase):
    def test_implements_the_opentelemetry_api(self):
        provider = self.provider()
        self.assertIsInstance(provider, trace.TracerProvider)
        tracer = trace.get_tracer("test", "1.0", tracer_provider=provider)
        with tracer.start_as_current_span("work") as span:
            self.assertIs(trace.get_current_span(), span)
            self.assertTrue(span.is_recording())
        self.assertFalse(span.is_recording())
        (row,) = load_spans(self.dir).itertuples()
        self.assertEqual(row.name, "work")
        self.assertEqual(row.status, "UNSET")
        self.assertGreater(row.duration, 0)

    def test_writes_records_as_they_happen(self):
        provider = self.provider(resource={"lr": 0.1})
        tracer = provider.get_tracer("test")
        span = tracer.start_span("fit", attributes={"epochs": 3})
        types = [r["type"] for r in self.records(provider)]
        self.assertEqual(types, ["resource", "start"])
        span.add_event("checkpoint", {"path": "a.pt"})
        self.assertEqual(self.records(provider)[-1]["type"], "event")
        span.set_attribute("step", 30)
        span.end()
        resource, start, event, end = self.records(provider)
        self.assertEqual(resource["attributes"]["lr"], 0.1)
        self.assertEqual(start["attributes"], {"epochs": 3})
        self.assertEqual(start["scope"], "test")
        self.assertEqual(event["name"], "checkpoint")
        self.assertEqual(end["attributes"], {"epochs": 3, "step": 30})
        self.assertGreaterEqual(end["end_time"], start["start_time"])
        for record in (start, event, end):
            self.assertEqual(record["span_id"], start["span_id"])

    def test_nesting(self):
        tracer = self.provider().get_tracer("test")
        with tracer.start_as_current_span("fit") as fit:
            with tracer.start_as_current_span("epoch"):
                pass
            explicit = tracer.start_span(
                "steps", context=trace.set_span_in_context(fit)
            )
            explicit.end()
        with tracer.start_as_current_span("other"):
            pass
        spans = load_spans(self.dir).set_index("name")
        for child in ("epoch", "steps"):
            self.assertEqual(
                spans.loc[child, "parent_id"], spans.loc["fit", "span_id"]
            )
            self.assertEqual(
                spans.loc[child, "trace_id"], spans.loc["fit", "trace_id"]
            )
        self.assertIsNone(spans.loc["other", "parent_id"])
        self.assertNotEqual(
            spans.loc["other", "trace_id"], spans.loc["fit", "trace_id"]
        )
        self.assertEqual(len(spans.loc["fit", "trace_id"]), 32)
        self.assertEqual(len(spans.loc["fit", "span_id"]), 16)

    def test_keeps_values_opentelemetry_does_not_cover(self):
        tracer = self.provider().get_tracer("test")
        with tracer.start_as_current_span("epoch") as span:
            span.set_attributes({
                "cm": np.eye(2, dtype=int),
                "f1": np.array([0.5, 0.7]),
                "optim": {"lr": 0.1, "betas": [0.9, 0.99]},
                "sched": {},
                "nan": math.nan,
                "path": Path("data"),
            })
        span = load_spans(self.dir).iloc[0]
        self.assertEqual(span["cm"], ((1, 0), (0, 1)))
        self.assertEqual(span["f1"], (0.5, 0.7))
        self.assertEqual(span["optim.lr"], 0.1)
        self.assertEqual(span["optim.betas"], (0.9, 0.99))
        self.assertEqual(span["sched"], {})
        self.assertTrue(math.isnan(span["nan"]))
        self.assertEqual(span["path"], os.path.abspath("data"))

    def test_status_and_exceptions(self):
        tracer = self.provider().get_tracer("test")
        with self.assertRaises(ValueError):
            with tracer.start_as_current_span("current"):
                raise ValueError("bad")
        with self.assertRaises(KeyError):
            with tracer.start_span("plain"):
                raise KeyError("k")
        with tracer.start_span("ok") as span:
            span.set_status(StatusCode.OK)
            span.set_status(StatusCode.ERROR, "ignored, as OK is final")
        with tracer.start_span("unset") as span:
            span.set_status(StatusCode.ERROR, "failed")
            span.set_status(StatusCode.UNSET)
        spans = load_spans(self.dir).set_index("name")
        self.assertEqual(spans.loc["current", "status"], "ERROR")
        self.assertEqual(
            spans.loc["current", "status_description"], "ValueError: bad"
        )
        self.assertEqual(spans.loc["plain", "status"], "ERROR")
        self.assertEqual(spans.loc["ok", "status"], "OK")
        self.assertEqual(spans.loc["unset", "status"], "ERROR")
        events = load_events(self.dir, "exception").set_index("span")
        self.assertEqual(events.loc["current", "exception.type"], "ValueError")
        self.assertEqual(events.loc["current", "exception.message"], "bad")
        self.assertTrue(events.loc["current", "exception.escaped"])
        self.assertIn(
            "raise ValueError", events.loc["current", "exception.stacktrace"]
        )

    def test_ended_spans_do_not_change(self):
        provider = self.provider()
        span = provider.get_tracer("test").start_span("a")
        span.update_name("b")
        span.end()
        span.end()
        span.set_attribute("late", 1)
        span.add_event("late")
        span.update_name("c")
        types = [r["type"] for r in self.records(provider)]
        self.assertEqual(types, ["resource", "start", "end"])
        (row,) = load_spans(self.dir).itertuples()
        self.assertEqual(row.name, "b")
        self.assertNotIn("late", load_spans(self.dir))

    def test_links(self):
        tracer = self.provider().get_tracer("test")
        first = tracer.start_span("first")
        first.end()
        second = tracer.start_span(
            "second", links=[Link(first.get_span_context(), {"why": "a"})]
        )
        second.add_link(first.get_span_context())
        second.end()
        end = self.records(second._provider)[-1]
        span_id = f"{first.get_span_context().span_id:016x}"
        self.assertEqual(
            end["links"],
            [
                {
                    "trace_id": f"{first.get_span_context().trace_id:032x}",
                    "span_id": span_id,
                    "attributes": {"why": "a"},
                },
                {
                    "trace_id": f"{first.get_span_context().trace_id:032x}",
                    "span_id": span_id,
                    "attributes": {},
                },
            ],
        )

    def test_spans_of_a_killed_job(self):
        provider = self.provider()
        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("done"):
            pass
        tracer.start_span("open", attributes={"epoch": 2})
        # a line cut off when the job was killed
        (path,) = provider.dir.glob("spans-*.jsonl")
        with open(path, "a") as f:
            f.write('{"type": "start", "span_id": "ab')
        spans = load_spans(self.dir).set_index("name")
        self.assertEqual(list(spans.index), ["done", "open"])
        self.assertTrue(math.isnan(spans.loc["open", "duration"]))
        self.assertIsNone(spans.loc["open", "status"])
        self.assertEqual(spans.loc["open", "epoch"], 2)

    def test_each_provider_has_its_own_file(self):
        # e.g. two processes of a distributed job, or a requeued job
        first = self.provider(resource={"rank": 0})
        second = self.provider(resource={"rank": 1})
        for provider in (first, second):
            with provider.get_tracer("test").start_as_current_span("fit"):
                pass
        self.assertEqual(len(list(first.dir.glob("spans-*.jsonl"))), 2)
        spans = load_spans(self.dir)
        self.assertEqual(list(spans["run"]), ["run", "run"])
        self.assertEqual(sorted(spans["resource.rank"]), [0, 1])

    def test_resource(self):
        env = {"SLURM_JOB_ID": "78", "SLURM_RESTART_COUNT": "2"}
        with mock.patch.dict(os.environ, env):
            provider = FileTracerProvider(self.dir, resource={"lr": 0.1})
        self.assertEqual(provider.dir, self.dir / "78")
        resource = provider.resource
        self.assertEqual(resource["lr"], 0.1)
        self.assertEqual(resource["slurm.job.id"], "78")
        self.assertEqual(resource["slurm.restart_count"], "2")
        self.assertEqual(resource["process.pid"], os.getpid())
        self.assertEqual(resource["telemetry.sdk.name"], "tklearn")
        self.assertIn("host.name", resource)


class TestDefaultRunName(TracingTestCase):
    def test_names(self):
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
        self.assertTrue(default_run_name().endswith(f"-{os.getpid()}"))


class TestLoad(TracingTestCase):
    def test_runs_names_and_columns(self):
        for name, lr in [("sweep/a", 0.1), ("sweep/b", 0.5)]:
            tracer = self.provider(name, resource={"lr": lr}).get_tracer("t")
            with tracer.start_as_current_span("fit"):
                for epoch in range(2):
                    attributes = {"epoch": epoch, "name": "clash"}
                    with tracer.start_as_current_span(
                        "epoch", attributes=attributes
                    ):
                        pass
        epochs = load_spans(self.dir, "epoch")
        self.assertEqual(
            list(epochs["run"]), ["sweep/a"] * 2 + ["sweep/b"] * 2
        )
        self.assertEqual(list(epochs["epoch"]), [0, 1, 0, 1])
        self.assertEqual(list(epochs["resource.lr"]), [0.1, 0.1, 0.5, 0.5])
        self.assertEqual(set(epochs["name"]), {"epoch"})
        self.assertEqual(set(epochs["attributes.name"]), {"clash"})
        self.assertEqual(
            str(epochs["start_time"].dtype), "datetime64[ns, UTC]"
        )
        bare = load_spans(self.dir / "sweep", resource=False)
        self.assertEqual(list(bare["run"]), ["a"] * 3 + ["b"] * 3)
        self.assertFalse(any(c.startswith("resource.") for c in bare))

    def test_empty_directory(self):
        spans = load_spans(self.dir)
        self.assertTrue(spans.empty)
        self.assertIn("duration", spans)
        self.assertTrue(load_events(self.dir).empty)

    def test_does_not_import_torch(self):
        code = (
            "import sys, tklearn.tracing; "
            "assert 'torch' not in sys.modules, 'torch imported'"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


class TestTraining(TrainerTestCase, TracingTestCase):
    def setUp(self):
        TrainerTestCase.setUp(self)
        TracingTestCase.setUp(self)

    def test_records_a_fit(self):
        provider = self.provider(resource={"lr": 0.5})
        callback = OpenTelemetryCallback(
            tracer_provider=provider, log_every_n_steps=3
        )
        trainer = self.trainer(
            metrics={"acc": Accuracy(), "cm": ConfusionMatrix()},
            callbacks=[callback],
        )
        history = trainer.fit(self.loader(), self.loader(), epochs=2)
        trainer.evaluate(self.loader(), prefix="test_")
        spans = load_spans(self.dir)
        self.assertEqual(
            list(spans["name"]),
            ["fit", "epoch", "steps", "steps", "evaluate"] * 1
            + ["epoch", "steps", "steps", "evaluate", "evaluate"],
        )
        epochs = load_spans(self.dir, "epoch")
        for row, logs in zip(epochs.to_dict("records"), history):
            self.assertAlmostEqual(row["loss"], logs["loss"])
            self.assertAlmostEqual(row["valid_acc"], logs["valid_acc"])
            # JSON text from the callback, which any SDK can take
            self.assertEqual(
                json.loads(row["valid_cm"]), logs["valid_cm"].tolist()
            )
        self.assertEqual(list(epochs["resource.lr"]), [0.5, 0.5])
        fit = load_spans(self.dir, "fit").iloc[0]
        self.assertEqual(fit["step"], 12)
        self.assertEqual(fit["model.parameters"], 10)
        steps = load_spans(self.dir, "steps")
        self.assertEqual(list(steps["step"]), [3, 6, 9, 12])
        self.assertTrue((steps["parent_id"] == fit["span_id"]).all())


if __name__ == "__main__":
    unittest.main()
