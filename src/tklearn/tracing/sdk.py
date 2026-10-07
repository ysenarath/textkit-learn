from __future__ import annotations

import contextlib
import json
import os
import platform
import secrets
import socket
import subprocess
import threading
import time
import traceback
from collections.abc import Generator, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import (
    Link,
    SpanContext,
    SpanKind,
    Status,
    StatusCode,
    TraceFlags,
)

import tklearn
from tklearn.utils.flatten import flatten, freeze

__all__ = [
    "FileTracerProvider",
    "default_run_name",
]

# resource attributes from SLURM's environment variables, when set
_SLURM_VARIABLES = {
    "SLURM_JOB_ID": "slurm.job.id",
    "SLURM_ARRAY_JOB_ID": "slurm.array_job.id",
    "SLURM_ARRAY_TASK_ID": "slurm.array_task.id",
    "SLURM_JOB_NAME": "slurm.job.name",
    "SLURM_JOB_NODELIST": "slurm.job.nodelist",
    "SLURM_NNODES": "slurm.job.num_nodes",
    "SLURM_PROCID": "slurm.procid",
    "SLURM_RESTART_COUNT": "slurm.restart_count",
    "SLURM_SUBMIT_DIR": "slurm.submit_dir",
}


class FileTracerProvider(trace.TracerProvider):
    """An OpenTelemetry tracer provider that writes spans to plain files.

    It implements the OpenTelemetry tracing API, so code instrumented with
    that API, such as `tklearn.nn.callbacks.OpenTelemetryCallback`,
    records into it. Its files suit shared storage without a database,
    such as NFS used by SLURM jobs on several machines. A run is a
    directory, ``root_dir/name``, of JSON-lines files
    ``spans-<time>-<host>-<pid>-<id>.jsonl``, one per provider object, so
    processes never write to the same file: each process of a
    distributed job, a requeued job, or another job given the same name,
    adds a file to the run. A file holds, one per line:

    - a ``"resource"`` record, first: the `resource` attributes;
    - a ``"start"`` record when a span starts: its name, ids, parent,
      kind, start time and attributes;
    - an ``"event"`` record when a span event is added, e.g. an
      exception;
    - an ``"end"`` record when a span ends: its end time, status, and
      final name, attributes and links.

    Each record is written when it happens, by opening, appending to and
    closing the file, which makes it visible to other machines (NFS
    close-to-open consistency). A job that is killed loses no record, and
    the spans it left open have a start but no end. Avoid many spans a
    second. `load_spans` and `load_events` read the files.

    Unlike the OpenTelemetry SDK, it has no sampling, batching or limits,
    and keeps attribute values that OpenTelemetry's types do not cover:
    nested mappings become dotted keys, and lists, tuples and arrays
    tuples (JSON arrays), as by `tklearn.utils.flatten.flatten`. It does
    not export to other backends.

    Parameters
    ----------
    root_dir : str or path-like
        Directory of the runs.
    name : str, optional
        Name of the run's directory; may contain ``/`` to group runs.
        Defaults to `default_run_name`: the SLURM job, so that a requeued
        job continues its run.
    resource : Mapping, optional
        Attributes of the run, e.g. its settings, to compare runs by
        (``resource.<key>`` columns of `load_spans`). They extend the
        ``host.name``, ``process.pid``, ``process.runtime.version``,
        ``vcs.ref.head.revision`` (the git commit of the working
        directory) and ``slurm.*`` attributes recorded by default.

    Examples
    --------
    >>> provider = FileTracerProvider(
    ...     "/nfs/project/runs", resource={"model": "roberta-base", "lr": 2e-5}
    ... )
    >>> callback = OpenTelemetryCallback(tracer_provider=provider)
    >>> trainer = Trainer(model, optimizer, callbacks=[callback])
    >>> trainer.fit(train_loader, valid_loader, epochs=3)
    >>> load_spans("/nfs/project/runs", name="epoch")  # one row per epoch
    """

    def __init__(
        self,
        root_dir: str | os.PathLike[str],
        name: str | None = None,
        resource: Mapping[str, Any] | None = None,
    ) -> None:
        self._root_dir = Path(root_dir)
        self._name = default_run_name() if name is None else name
        self._lock = threading.Lock()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # unique to this object; starts with the time, to sort by it
        token = secrets.token_hex(3)
        filename = f"spans-{stamp}-{_short_host()}-{os.getpid()}-{token}.jsonl"
        self._path = self.dir / filename
        self._resource = {**_default_resource(), **_clean(resource or {})}
        self.dir.mkdir(parents=True, exist_ok=True)
        self._write({
            "type": "resource",
            "time": time.time_ns(),
            "attributes": self._resource,
        })

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self.dir)!r})"

    @property
    def name(self) -> str:
        """Name of the run's directory."""
        return self._name

    @property
    def dir(self) -> Path:
        """Directory of the run."""
        return self._root_dir / self._name

    @property
    def resource(self) -> dict[str, Any]:
        """Attributes of the run, written first to its file."""
        return dict(self._resource)

    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: str | None = None,
        schema_url: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> trace.Tracer:
        return _Tracer(
            self, instrumenting_module_name, instrumenting_library_version
        )

    def _write(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(record) + "\n"
        # closing after each record publishes it to other NFS clients
        with self._lock, open(self._path, "a", encoding="utf-8") as f:
            f.write(line)


class _Tracer(trace.Tracer):
    def __init__(
        self, provider: FileTracerProvider, name: str, version: str | None
    ) -> None:
        self._provider = provider
        self._scope = name if version is None else f"{name}=={version}"

    def start_span(
        self,
        name: str,
        context: Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Mapping[str, Any] | None = None,
        links: Sequence[Link] | None = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> trace.Span:
        parent = trace.get_current_span(context).get_span_context()
        if parent.is_valid:
            trace_id, parent_id = parent.trace_id, parent.span_id
        else:
            trace_id, parent_id = _random_id(128), None
        span_context = SpanContext(
            trace_id,
            _random_id(64),
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
        return _Span(
            self._provider,
            name,
            span_context,
            parent_id=parent_id,
            kind=kind,
            scope=self._scope,
            attributes=attributes,
            links=links,
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
        )

    @contextlib.contextmanager
    def start_as_current_span(
        self,
        name: str,
        context: Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Mapping[str, Any] | None = None,
        links: Sequence[Link] | None = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> Generator[trace.Span]:
        span = self.start_span(
            name,
            context,
            kind,
            attributes,
            links,
            start_time,
            record_exception,
            set_status_on_exception,
        )
        # records an escaping exception itself, as use_span would record it
        # without exception.escaped
        with trace.use_span(
            span, record_exception=False, set_status_on_exception=False
        ):
            try:
                yield span
            except BaseException as e:
                span._record_escaped(e)
                raise
            finally:
                if end_on_exit:
                    span.end()


class _Span(trace.Span):
    def __init__(
        self,
        provider: FileTracerProvider,
        name: str,
        context: SpanContext,
        *,
        parent_id: int | None,
        kind: SpanKind,
        scope: str,
        attributes: Mapping[str, Any] | None,
        links: Sequence[Link] | None,
        start_time: int | None,
        record_exception: bool,
        set_status_on_exception: bool,
    ) -> None:
        self._provider = provider
        self._name = name
        self._context = context
        self._record_exception = record_exception
        self._set_status_on_exception = set_status_on_exception
        self._attributes = _clean(attributes or {})
        self._links = [
            _link(link.context, link.attributes) for link in links or ()
        ]
        self._status = Status(StatusCode.UNSET)
        self._end_time: int | None = None
        self._lock = threading.Lock()
        self._start_time = time.time_ns() if start_time is None else start_time
        self._write(
            "start",
            {
                "parent_id": None
                if parent_id is None
                else _span_id(parent_id),
                "name": name,
                "kind": kind.name,
                "scope": scope,
                "start_time": self._start_time,
                "attributes": self._attributes,
                "links": self._links,
            },
        )

    def __repr__(self) -> str:
        span_id = _span_id(self._context.span_id)
        return f"_Span({self._name!r}, span_id={span_id!r})"

    def get_span_context(self) -> SpanContext:
        return self._context

    def is_recording(self) -> bool:
        return self._end_time is None

    def set_attribute(self, key: str, value: Any) -> None:
        self.set_attributes({key: value})

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        with self._lock:
            if self.is_recording():
                self._attributes.update(_clean(attributes))

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
        timestamp: int | None = None,
    ) -> None:
        if not self.is_recording():
            return
        self._write(
            "event",
            {
                "name": name,
                "time": time.time_ns() if timestamp is None else timestamp,
                "attributes": _clean(attributes or {}),
            },
        )

    def add_link(
        self, context: SpanContext, attributes: Mapping[str, Any] | None = None
    ) -> None:
        with self._lock:
            if self.is_recording():
                self._links.append(_link(context, attributes))

    def update_name(self, name: str) -> None:
        with self._lock:
            if self.is_recording():
                self._name = name

    def set_status(
        self, status: Status | StatusCode, description: str | None = None
    ) -> None:
        if isinstance(status, StatusCode):
            status = Status(status, description)
        with self._lock:
            # OK is final, and UNSET does not override another status
            # (Status.is_ok is also true for UNSET)
            ok = self._status.status_code is StatusCode.OK
            if not self.is_recording() or ok:
                return
            if status.status_code is not StatusCode.UNSET:
                self._status = status

    def record_exception(
        self,
        exception: BaseException,
        attributes: Mapping[str, Any] | None = None,
        timestamp: int | None = None,
        escaped: bool = False,
    ) -> None:
        stacktrace = "".join(
            traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
        )
        self.add_event(
            "exception",
            {
                "exception.type": type(exception).__qualname__,
                "exception.message": str(exception),
                "exception.stacktrace": stacktrace,
                "exception.escaped": escaped,
                **(attributes or {}),
            },
            timestamp,
        )

    def end(self, end_time: int | None = None) -> None:
        with self._lock:
            if not self.is_recording():
                return
            self._end_time = time.time_ns() if end_time is None else end_time
            record = {
                "name": self._name,
                "end_time": self._end_time,
                "status": self._status.status_code.name,
                "status_description": self._status.description,
                "attributes": self._attributes,
                "links": self._links,
            }
        self._write("end", record)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_val is not None:
            self._record_escaped(exc_val)
        self.end()

    def _record_escaped(self, exception: BaseException) -> None:
        """Record an exception that leaves the span's scope, as set by the
        tracer's ``record_exception`` and ``set_status_on_exception``."""
        if not self.is_recording():
            return
        if self._record_exception:
            self.record_exception(exception, escaped=True)
        if self._set_status_on_exception:
            description = f"{type(exception).__name__}: {exception}"
            self.set_status(StatusCode.ERROR, description)

    def _write(self, record_type: str, fields: Mapping[str, Any]) -> None:
        self._provider._write({
            "type": record_type,
            "trace_id": _trace_id(self._context.trace_id),
            "span_id": _span_id(self._context.span_id),
            **fields,
        })


def default_run_name() -> str:
    """The SLURM job (``<array job>_<task>`` in a job array), or else
    ``<time>-<host>-<pid>``."""
    env = os.environ
    if "SLURM_ARRAY_JOB_ID" in env and "SLURM_ARRAY_TASK_ID" in env:
        return f"{env['SLURM_ARRAY_JOB_ID']}_{env['SLURM_ARRAY_TASK_ID']}"
    if "SLURM_JOB_ID" in env:
        return env["SLURM_JOB_ID"]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{_short_host()}-{os.getpid()}"


def _clean(values: Mapping[str, Any]) -> dict[str, Any]:
    # attributes as written: a mapping value flattened into dotted keys,
    # and every value frozen, e.g. arrays as tuples
    clean: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, Mapping) and value:
            try:
                clean.update(flatten({key: value}))
                continue
            except ValueError:
                pass
        clean[str(key)] = freeze(value)
    return clean


def _link(context: SpanContext, attributes: Mapping[str, Any] | None) -> dict:
    return {
        "trace_id": _trace_id(context.trace_id),
        "span_id": _span_id(context.span_id),
        "attributes": _clean(attributes or {}),
    }


def _random_id(bits: int) -> int:
    # secrets is unaffected by forks and by seeding `random`
    while True:
        value = secrets.randbits(bits)
        if value:  # 0 is the invalid id
            return value


def _trace_id(value: int) -> str:
    return f"{value:032x}"


def _span_id(value: int) -> str:
    return f"{value:016x}"


def _short_host() -> str:
    return socket.gethostname().split(".")[0]


def _default_resource() -> dict[str, Any]:
    resource = {
        "host.name": socket.gethostname(),
        "process.pid": os.getpid(),
        "process.runtime.version": platform.python_version(),
        "telemetry.sdk.name": "tklearn",
        "telemetry.sdk.version": tklearn.__version__,
        "vcs.ref.head.revision": _git_commit(),
    }
    for variable, key in _SLURM_VARIABLES.items():
        if variable in os.environ:
            resource[key] = os.environ[variable]
    return {key: value for key, value in resource.items() if value is not None}


def _git_commit() -> str | None:
    """Commit checked out in the working directory, if it is a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None
