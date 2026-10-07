import json
import math
import unittest
from unittest import mock

from accelerate import Accelerator
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode
from test_nn import Classifier, TrainerTestCase

from tklearn.metrics import Accuracy, ConfusionMatrix
from tklearn.nn.callbacks import LambdaCallback, OpenTelemetryCallback
from tklearn.nn.callbacks.otel import _attribute_value


class TestOpenTelemetryCallback(TrainerTestCase):
    def setUp(self):
        super().setUp()
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))

    def callback(self, **kwargs):
        return OpenTelemetryCallback(tracer_provider=self.provider, **kwargs)

    def spans(self, name=None):
        spans = self.exporter.get_finished_spans()
        return [s for s in spans if name is None or s.name == name]

    def assert_parent(self, span, parent):
        self.assertEqual(span.parent.span_id, parent.context.span_id)
        self.assertEqual(span.context.trace_id, parent.context.trace_id)

    def test_traces_a_fit(self):
        callback = self.callback(log_every_n_steps=2)
        trainer = self.trainer(
            metrics={"acc": Accuracy()}, callbacks=[callback]
        )
        history = trainer.fit(self.loader(), self.loader(), epochs=2)
        # in the order they end; 6 batches per epoch, one step each
        per_epoch = [*["steps"] * 3, "evaluate", "epoch"]
        self.assertEqual(
            [s.name for s in self.spans()], [*per_epoch * 2, "fit"]
        )
        (fit,) = self.spans("fit")
        self.assertIsNone(fit.parent)
        for span in self.spans("epoch") + self.spans("steps"):
            self.assert_parent(span, fit)
        for evaluate, epoch in zip(
            self.spans("evaluate"), self.spans("epoch")
        ):
            self.assert_parent(evaluate, epoch)
            self.assertEqual(evaluate.attributes["num_batches"], 6)
        for i, (epoch, logs) in enumerate(zip(self.spans("epoch"), history)):
            self.assertEqual(epoch.attributes["epoch"], i)
            self.assertEqual(epoch.attributes["step"], 6 * (i + 1))
            for key, value in logs.items():
                self.assertAlmostEqual(epoch.attributes[key], value)

    def test_describes_the_fit(self):
        callback = self.callback()
        trainer = self.trainer(
            metrics={"acc": Accuracy()},
            callbacks=[callback],
            lr_scheduler="linear",
            warmup=2,
            max_grad_norm=1.0,
        )
        trainer.fit(self.loader(), epochs=2)
        attributes = self.spans("fit")[0].attributes
        self.assertTrue(attributes["model.class"].endswith("Classifier"))
        self.assertEqual(attributes["model.parameters"], 4 * 2 + 2)
        self.assertEqual(attributes["optimizer.class"], "torch.optim.sgd.SGD")
        self.assertEqual(
            attributes["optimizer.param_groups.*.initial_lr"], (0.5,)
        )
        self.assertEqual(attributes["trainer.epochs"], 2)
        self.assertEqual(attributes["trainer.num_batches"], 6)
        self.assertEqual(attributes["trainer.lr_scheduler"], "linear")
        self.assertEqual(attributes["trainer.max_grad_norm"], 1.0)
        self.assertEqual(attributes["trainer.metrics"], ("acc",))
        self.assertEqual(
            attributes["trainer.callbacks"],
            ("tklearn.nn.callbacks.otel.OpenTelemetryCallback",),
        )
        # set when fit ends
        self.assertEqual(attributes["step"], 12)
        self.assertIs(attributes["stopped"], False)

    def test_steps_hold_the_mean_of_their_batches(self):
        losses = []
        record = LambdaCallback(
            on_train_batch_end=lambda trainer, batch, logs: losses.append(
                logs["loss"]
            )
        )
        callback = self.callback(log_every_n_steps=4)
        trainer = self.trainer(callbacks=[record, callback])
        trainer.fit(self.loader(), epochs=2)
        steps = self.spans("steps")
        self.assertEqual([s.attributes["step"] for s in steps], [4, 8, 12])
        self.assertEqual([s.attributes["epoch"] for s in steps], [0, 1, 1])
        for span in steps:
            step = span.attributes["step"]
            self.assertAlmostEqual(
                span.attributes["loss"], sum(losses[step - 4 : step]) / 4
            )
            self.assertEqual(span.attributes["lr"], 0.5)
            self.assertGreater(span.attributes["grad_norm"], 0)
            self.assertGreater(span.attributes["step_time"], 0)
            self.assertLess(span.start_time, span.end_time)
            # no CUDA or MPS memory on the CPU
            self.assertNotIn("memory_gb", span.attributes)

    def test_steps_with_gradient_accumulation(self):
        callback = self.callback(log_every_n_steps=1)
        trainer = self.trainer(accumulation=2, callbacks=[callback])
        trainer.fit(self.loader(), epochs=1)
        steps = self.spans("steps")
        self.assertEqual([s.attributes["step"] for s in steps], [1, 2, 3])

    def test_grad_norm_before_clipping(self):
        norms = []

        def record(trainer, batch, logs):
            if trainer.global_step % 2 == 0:
                norms.append(float(trainer.grad_norm))

        callback = self.callback(log_every_n_steps=2)
        trainer = self.trainer(
            max_grad_norm=1e-3,
            callbacks=[LambdaCallback(on_train_batch_end=record), callback],
        )
        trainer.fit(self.loader(), epochs=1)
        steps = self.spans("steps")
        self.assertEqual(len(steps), 3)
        for span, norm in zip(steps, norms):
            self.assertAlmostEqual(span.attributes["grad_norm"], norm)
            self.assertGreater(span.attributes["grad_norm"], 1e-3)

    def test_evaluate_and_predict_in_the_current_span(self):
        trainer = self.trainer(
            metrics={"acc": Accuracy()}, callbacks=[self.callback()]
        )
        tracer = self.provider.get_tracer("test")
        with tracer.start_as_current_span("experiment"):
            results = trainer.evaluate(self.loader(), prefix="test_")
            trainer.predict(self.loader())
        (experiment,) = self.spans("experiment")
        (evaluate,) = self.spans("evaluate")
        (predict,) = self.spans("predict")
        self.assert_parent(evaluate, experiment)
        self.assert_parent(predict, experiment)
        for key, value in results.items():
            self.assertAlmostEqual(evaluate.attributes[key], value)
        self.assertEqual(predict.attributes["num_batches"], 6)

    def test_values_attributes_cannot_hold(self):
        callback = self.callback()
        trainer = self.trainer(
            metrics={"cm": ConfusionMatrix()}, callbacks=[callback]
        )
        history = trainer.fit(self.loader(), self.loader(), epochs=1)
        attributes = self.spans("epoch")[0].attributes
        # a confusion matrix, as JSON text
        self.assertEqual(
            json.loads(attributes["valid_cm"]), history[0]["valid_cm"].tolist()
        )

    def test_attribute_values(self):
        cases = [
            (0.5, 0.5),
            ((1, 2), (1, 2)),
            ((1, 2.5), (1.0, 2.5)),
            ((None, "relu"), (None, "relu")),
            ((True, None), (True, None)),
            (((1, 0), (0, 1)), "[[1, 0], [0, 1]]"),
            ((1, "a"), '[1, "a"]'),
            ({}, "{}"),
            (math.inf, math.inf),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(_attribute_value(value), expected)

    def test_spans_of_a_fit_that_raised_end_with_an_error(self):
        def fail(trainer, batch, logs):
            if trainer.global_step == 3:
                raise RuntimeError("stop")

        callback = self.callback()
        trainer = self.trainer(
            callbacks=[callback, LambdaCallback(on_train_batch_end=fail)]
        )
        with self.assertRaises(RuntimeError):
            trainer.fit(self.loader(), epochs=2)
        self.assertEqual(self.spans(), [])
        # still current, as no hook ran after the error
        self.assertTrue(trace.get_current_span().is_recording())
        trainer.callbacks = [callback]
        trainer.fit(self.loader(), epochs=1)
        self.assertFalse(trace.get_current_span().get_span_context().is_valid)
        statuses = [(s.name, s.status.status_code) for s in self.spans()]
        self.assertEqual(
            statuses,
            [
                ("epoch", StatusCode.ERROR),
                ("fit", StatusCode.ERROR),
                ("epoch", StatusCode.UNSET),
                ("fit", StatusCode.UNSET),
            ],
        )

    def test_epochs_only(self):
        trainer = self.trainer(callbacks=[self.callback(log_every_n_steps=0)])
        trainer.fit(self.loader(), epochs=2)
        self.assertEqual(
            [s.name for s in self.spans()], ["epoch", "epoch", "fit"]
        )

    def test_only_the_main_process_records(self):
        trainer = self.trainer(callbacks=[self.callback(log_every_n_steps=2)])
        with mock.patch.object(
            Accelerator, "is_main_process", new_callable=mock.PropertyMock
        ) as is_main_process:
            is_main_process.return_value = False
            trainer.fit(self.loader(), self.loader(), epochs=1)
            trainer.predict(self.loader())
        self.assertEqual(self.spans(), [])

    def test_without_an_sdk(self):
        # the global provider records nothing until an SDK sets one
        trainer = self.trainer(callbacks=[OpenTelemetryCallback()])
        trainer.fit(self.loader(), self.loader(), epochs=1)
        trainer.evaluate(self.loader())
        self.assertEqual(self.spans(), [])

    def test_spans_are_current_whatever_the_order(self):
        seen = []

        def note(hook):
            def record(trainer, *args):
                span = trace.get_current_span()
                span.add_event(hook)
                seen.append((hook, span))

            return record

        # listed before the OpenTelemetry callback, which wraps it anyway
        early = LambdaCallback(
            on_train_begin=note("train_begin"),
            on_epoch_begin=note("epoch_begin"),
            on_test_end=note("test_end"),
            on_epoch_end=note("epoch_end"),
            on_train_end=note("train_end"),
        )
        trainer = self.trainer(callbacks=[early, self.callback()])
        trainer.fit(self.loader(), self.loader(), epochs=1)
        (fit,) = self.spans("fit")
        (epoch,) = self.spans("epoch")
        (evaluate,) = self.spans("evaluate")
        expected = {
            "train_begin": fit,
            "epoch_begin": epoch,
            "test_end": evaluate,
            "epoch_end": epoch,
            "train_end": fit,
        }
        for hook, span in seen:
            with self.subTest(hook=hook):
                self.assertEqual(
                    span.get_span_context().span_id,
                    expected[hook].context.span_id,
                )
                self.assertIn(hook, [e.name for e in expected[hook].events])
        # the context before fit is back
        self.assertFalse(trace.get_current_span().get_span_context().is_valid)

    def test_model_code_adds_child_spans(self):
        tracer = self.provider.get_tracer("model")

        class Traced(Classifier):
            def training_step(self, batch):
                with tracer.start_as_current_span("forward"):
                    return super().training_step(batch)

        trainer = self.trainer(Traced(), callbacks=[self.callback()])
        trainer.fit(self.loader(), epochs=1)
        (epoch,) = self.spans("epoch")
        forwards = self.spans("forward")
        self.assertEqual(len(forwards), 6)
        for span in forwards:
            self.assert_parent(span, epoch)

    def test_validates_log_every_n_steps(self):
        for value in (-1, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    OpenTelemetryCallback(log_every_n_steps=value)


if __name__ == "__main__":
    unittest.main()
