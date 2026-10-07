from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import Span, Status, StatusCode, TracerProvider
from torch.optim.lr_scheduler import LRScheduler

import tklearn
from tklearn.nn.callbacks.base import Callback
from tklearn.utils.flatten import flatten, freeze

if TYPE_CHECKING:
    from tklearn.nn.trainer import Trainer

__all__ = [
    "OpenTelemetryCallback",
]


class OpenTelemetryCallback(Callback):
    """Trace `Trainer.fit`, `evaluate` and `predict` with OpenTelemetry.

    The callback uses only the OpenTelemetry API: the application decides
    where the spans go by configuring an SDK, e.g. ``opentelemetry-sdk``
    with an OTLP exporter for Jaeger or a collector. Without one, the
    spans are not recorded.

    `fit` gives a ``"fit"`` span, holding

    - an ``"epoch"`` span per epoch, with ``epoch``, ``step`` and the
      epoch's logs, as in `Trainer.history`; it holds the ``"evaluate"``
      span of the epoch;
    - a ``"steps"`` span every `log_every_n_steps` optimizer steps,
      covering those steps, with ``step``, ``epoch``, the mean training
      ``loss`` and logged terms over them and over all processes, ``lr``
      (``lr_0``, ``lr_1``, ... with several parameter groups),
      ``grad_norm`` (before clipping), ``step_time`` (seconds per step,
      without evaluation) and, on CUDA or MPS, ``memory_gb``.

    The ``"fit"`` span describes the model, optimizer and trainer
    (``model.parameters``, ``optimizer.class``, ``trainer.epochs``, ...),
    and gets ``step`` and ``stopped`` when `fit` ends. `evaluate` and
    `predict` outside `fit` give ``"evaluate"`` (with the results) and
    ``"predict"`` spans in the current span, e.g. one that groups the
    runs of an experiment.

    The callback is a wrapper (`Callback.wrapper`): its spans enclose
    the hooks of the other callbacks, whatever their order. The
    ``"fit"``, ``"epoch"``, ``"evaluate"`` and ``"predict"`` spans are
    current while they run, so the other callbacks and the model can add
    events and spans to them through ``trace.get_current_span()``.

    Nested values become dotted attributes, as by
    `tklearn.utils.flatten.flatten`; values that attributes cannot hold,
    such as a confusion matrix, are recorded as JSON text.

    Only the main process records spans.

    Parameters
    ----------
    tracer_provider : TracerProvider, optional
        Provider of the tracer; defaults to the global one, set with
        ``opentelemetry.trace.set_tracer_provider``.
    log_every_n_steps : int, default=50
        Optimizer steps per ``"steps"`` span; 0 records no steps.

    Notes
    -----
    A span is ended by the hook that closes it. When `fit`, `evaluate`
    or `predict` raises, the spans it opened stay open, and current,
    until the next `fit` begins, which ends them with an error status.

    Computing ``grad_norm`` without `Trainer.max_grad_norm` reads every
    gradient on the steps that are recorded. Like `ModelCheckpoint`, it
    needs the full gradients on the main process, so it does not support
    FSDP or DeepSpeed ZeRO-3.

    Examples
    --------
    >>> from opentelemetry import trace
    >>> from opentelemetry.sdk.trace import TracerProvider
    >>> from opentelemetry.sdk.trace.export import BatchSpanProcessor
    >>> from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    ...     OTLPSpanExporter,
    ... )
    >>> provider = TracerProvider()
    >>> provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    >>> trace.set_tracer_provider(provider)
    >>> callback = OpenTelemetryCallback()
    >>> trainer = Trainer(model, optimizer, callbacks=[callback])
    >>> with trace.get_tracer(__name__).start_as_current_span("experiment"):
    ...     trainer.fit(train_loader, valid_loader, epochs=3)
    ...     trainer.evaluate(test_loader, prefix="test_")
    """

    wrapper = True

    def __init__(
        self,
        *,
        tracer_provider: TracerProvider | None = None,
        log_every_n_steps: int = 50,
    ) -> None:
        if isinstance(log_every_n_steps, bool) or log_every_n_steps < 0:
            msg = (
                "log_every_n_steps must be a non-negative number of steps, "
                f"got {log_every_n_steps!r}"
            )
            raise ValueError(msg)
        self.tracer = trace.get_tracer(
            "tklearn.nn",
            tklearn.__version__,
            tracer_provider=tracer_provider,
        )
        self.log_every_n_steps = log_every_n_steps
        self._fit_span: Span | None = None
        self._epoch_span: Span | None = None
        # open evaluate and predict spans, innermost last
        self._spans: list[Span] = []
        # tokens restoring the context before each current span was made
        # current, innermost last
        self._tokens: list[object] = []
        self._test_depth = 0
        self._grad_norm: torch.Tensor | None = None
        self._reset_window(0)

    # --- fit ---------------------------------------------------------------

    def on_train_begin(self, trainer: Trainer) -> None:
        self._end_open_spans()
        self._reset_window(0)
        self._grad_norm = None
        if not trainer.accelerator.is_main_process:
            return
        device = trainer.accelerator.device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        attributes = _attributes({
            "model": _describe_model(trainer.model),
            "optimizer": _describe_optimizer(trainer.optimizer),
            "trainer": _describe_trainer(trainer),
        })
        self._fit_span = self.tracer.start_span("fit", attributes=attributes)
        self._enter(self._fit_span)

    def on_epoch_begin(self, trainer: Trainer) -> None:
        if self._fit_span is None:
            return
        self._epoch_span = self.tracer.start_span(
            "epoch",
            context=trace.set_span_in_context(self._fit_span),
            attributes={"epoch": trainer.epoch},
        )
        self._enter(self._epoch_span)

    def on_before_optimizer_step(self, trainer: Trainer) -> None:
        # the step about to be taken is global_step + 1
        if not self._is_recorded_step(trainer.global_step + 1):
            return
        if trainer.max_grad_norm is not None and trainer.grad_norm is not None:
            self._grad_norm = trainer.grad_norm
        else:
            # fp16 gradients are scaled until the optimizer steps; unscaling
            # them here keeps the optimizer from doing it again
            trainer.accelerator.unscale_gradients()
            self._grad_norm = _grad_norm(trainer.model)

    def on_train_batch_end(
        self, trainer: Trainer, batch: Any, logs: dict[str, float]
    ) -> None:
        for key, value in logs.items():
            self._totals[key] = self._totals.get(key, 0.0) + value
            self._counts[key] = self._counts.get(key, 0) + 1
        step = trainer.global_step
        if (
            trainer.accelerator.sync_gradients
            and step > self._window_step
            and self._is_recorded_step(step)
        ):
            self._record_steps(trainer)

    def on_epoch_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:
        if self._epoch_span is None:
            return
        self._epoch_span.set_attributes(
            _attributes({**logs, "step": trainer.global_step})
        )
        self._exit()
        self._epoch_span.end()
        self._epoch_span = None

    def on_train_end(self, trainer: Trainer) -> None:
        if self._fit_span is None:
            return
        self._fit_span.set_attributes({
            "step": trainer.global_step,
            "stopped": trainer.should_stop,
        })
        self._exit()
        self._fit_span.end()
        self._fit_span = None

    # --- evaluate ----------------------------------------------------------

    def on_test_begin(self, trainer: Trainer) -> None:
        if self._test_depth == 0:
            self._test_start = time.perf_counter()
        self._test_depth += 1
        self._start_span("evaluate", trainer)

    def on_test_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:
        self._test_depth -= 1
        if self._test_depth == 0:
            # step_time leaves out evaluation during fit
            self._paused += time.perf_counter() - self._test_start
        self._end_span(_attributes(logs))

    # --- predict -----------------------------------------------------------

    def on_predict_begin(self, trainer: Trainer) -> None:
        self._start_span("predict", trainer)

    def on_predict_end(self, trainer: Trainer) -> None:
        self._end_span({})

    # --- helpers -----------------------------------------------------------

    def _context(self) -> Context | None:
        # the parent of new spans: the open epoch or fit span, or else the
        # current span
        for span in (self._epoch_span, self._fit_span):
            if span is not None:
                return trace.set_span_in_context(span)
        return None

    def _start_span(self, name: str, trainer: Trainer) -> None:
        if not trainer.accelerator.is_main_process:
            return
        attributes = {}
        if trainer.num_batches is not None:
            attributes["num_batches"] = trainer.num_batches
        span = self.tracer.start_span(
            name, context=self._context(), attributes=attributes
        )
        self._spans.append(span)
        self._enter(span)

    def _end_span(self, attributes: Mapping[str, Any]) -> None:
        if not self._spans:  # not the main process
            return
        span = self._spans.pop()
        span.set_attributes(attributes)
        self._exit()
        span.end()

    def _enter(self, span: Span) -> None:
        # current until it ends, for the other callbacks and the model
        context = trace.set_span_in_context(span)
        self._tokens.append(otel_context.attach(context))

    def _exit(self) -> None:
        otel_context.detach(self._tokens.pop())

    def _end_open_spans(self) -> None:
        # spans of a fit, evaluate or predict that raised, innermost first
        while self._tokens:
            self._exit()
        open_spans = [*reversed(self._spans), self._epoch_span, self._fit_span]
        for span in open_spans:
            if span is not None and span.is_recording():
                span.set_status(
                    Status(StatusCode.ERROR, "ended without finishing")
                )
                span.end()
        self._spans = []
        self._epoch_span = self._fit_span = None
        self._test_depth = 0

    def _is_recorded_step(self, step: int) -> bool:
        n = self.log_every_n_steps
        return n > 0 and step % n == 0

    def _reset_window(self, step: int) -> None:
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._window_step = step
        self._window_start = time.perf_counter()
        self._window_start_ns = time.time_ns()
        self._paused = 0.0

    def _record_steps(self, trainer: Trainer) -> None:
        step = trainer.global_step
        # reduces over processes, so every process takes part
        means = trainer._mean(self._totals, self._counts)
        step_time = (
            time.perf_counter() - self._window_start - self._paused
        ) / (step - self._window_step)
        start_ns, grad_norm = self._window_start_ns, self._grad_norm
        self._reset_window(step)
        self._grad_norm = None
        if self._fit_span is None:  # not the main process
            return
        groups = trainer.optimizer.param_groups
        if len(groups) == 1:
            lrs = {"lr": float(groups[0]["lr"])}
        else:
            lrs = {f"lr_{i}": float(g["lr"]) for i, g in enumerate(groups)}
        values = {
            **means,
            **lrs,
            "step": step,
            "epoch": trainer.epoch,
            "grad_norm": None if grad_norm is None else float(grad_norm),
            "step_time": step_time,
            "memory_gb": _memory_gb(trainer.accelerator.device),
        }
        span = self.tracer.start_span(
            "steps",
            context=trace.set_span_in_context(self._fit_span),
            attributes=_attributes(values),
            start_time=start_ns,
        )
        span.end()


def _attributes(values: Mapping[str, Any]) -> dict[str, Any]:
    """`values` as span attributes: flat, with each value one that
    OpenTelemetry takes, or else JSON text; None is left out."""
    try:
        flat = flatten(values)
    except ValueError:
        # nesting that flatten rejects, kept per key as text
        flat = {str(key): freeze(value) for key, value in values.items()}
    attributes = {}
    for key, value in flat.items():
        value = _attribute_value(value)
        if value is not None:
            attributes[key] = value
    return attributes


def _attribute_value(value: Any) -> Any:
    # attributes hold str, bool, int or float, or a sequence of one of
    # these types, which may contain None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, tuple):
        items = [item for item in value if item is not None]
        for kind in (bool, str):
            if all(isinstance(item, kind) for item in items):
                return value
        if all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in items
        ):
            if all(isinstance(item, int) for item in items):
                return value
            return tuple(None if v is None else float(v) for v in value)
    return json.dumps(value)


def _grad_norm(model: torch.nn.Module) -> torch.Tensor | None:
    norms = [
        torch.linalg.vector_norm(p.grad.detach(), dtype=torch.float32)
        for p in model.parameters()
        if p.grad is not None
    ]
    if not norms:
        return None
    return torch.linalg.vector_norm(torch.stack(norms))


def _memory_gb(device: torch.device) -> float | None:
    # the peak allocated CUDA memory since the last record, or the
    # allocated MPS memory
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        return peak / 1e9
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1e9
    return None


def _describe_model(model: torch.nn.Module) -> dict[str, Any]:
    params = list(model.parameters())
    return {
        "class": _qualname(type(model)),
        "parameters": sum(p.numel() for p in params),
        "trainable_parameters": sum(
            p.numel() for p in params if p.requires_grad
        ),
    }


def _describe_optimizer(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    groups = [
        {k: v for k, v in group.items() if k != "params"}
        for group in optimizer.param_groups
    ]
    return {"class": _qualname(type(optimizer)), "param_groups": groups}


def _describe_trainer(trainer: Trainer) -> dict[str, Any]:
    scheduler = trainer.lr_scheduler
    if isinstance(scheduler, LRScheduler):
        scheduler = _qualname(type(scheduler))
    elif callable(scheduler):
        scheduler = _qualname(scheduler)
    accelerator = trainer.accelerator
    return {
        "epochs": trainer.epochs,
        "num_batches": trainer.num_batches,
        "lr_scheduler": scheduler,
        "warmup": trainer.warmup,
        "max_grad_norm": trainer.max_grad_norm,
        "mixed_precision": accelerator.mixed_precision,
        "gradient_accumulation_steps": accelerator.gradient_accumulation_steps,
        "num_processes": accelerator.num_processes,
        "device": str(accelerator.device),
        "metrics": list(trainer.metrics),
        "callbacks": [_qualname(type(c)) for c in trainer.callbacks],
        "torch": torch.__version__,
    }


def _qualname(obj: Any) -> str:
    module = getattr(obj, "__module__", None)
    name = getattr(obj, "__qualname__", None) or repr(obj)
    return f"{module}.{name}" if module else name
