from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from tklearn.tracing._records import (
    RunRecords,
    find_run_dirs,
    parse_line,
    span_files,
)

__all__ = [
    "load_events",
    "load_spans",
]

# columns of load_spans before the attributes
_SPAN_COLUMNS = (
    "run",
    "name",
    "trace_id",
    "span_id",
    "parent_id",
    "start_time",
    "end_time",
    "duration",
    "status",
    "status_description",
)
_EVENT_COLUMNS = ("run", "span", "trace_id", "span_id", "name", "time")


def load_spans(
    root_dir: str | os.PathLike[str],
    name: str | None = None,
    *,
    resource: bool = True,
) -> pd.DataFrame:
    """Read the spans of every run under a directory.

    Parameters
    ----------
    root_dir : str or path-like
        The `FileTracerProvider` ``root_dir``.
    name : str, optional
        Keep only spans of this name, e.g. ``"epoch"``; None keeps all.
    resource : bool, default=True
        Add the resource attributes of each span's process as
        ``resource.<key>`` columns, e.g. to compare the runs of a sweep.

    Returns
    -------
    DataFrame
        One row per span, in the order they started within each run:
        ``run`` (the run's directory under `root_dir`), ``name``,
        ``trace_id``, ``span_id``, ``parent_id``, ``start_time`` and
        ``end_time`` (UTC timestamps), ``duration`` (seconds), ``status``
        and ``status_description``, then a column per attribute (named
        ``attributes.<key>`` if it clashes with one of these). A span that
        never ended, e.g. in a job that was killed, has no end time,
        duration or status, and the attributes it started with. A line
        cut off by a killed job is skipped.
    """
    rows = []
    for run, spans, _ in _read_runs(root_dir):
        for span in spans.values():
            if name is not None and span["name"] != name:
                continue
            row = {column: span.get(column) for column in _SPAN_COLUMNS}
            row["run"] = run
            for key, value in span["attributes"].items():
                column = f"attributes.{key}" if key in row else key
                row[column] = value
            if resource:
                for key, value in span["resource"].items():
                    row[f"resource.{key}"] = value
            rows.append(row)
    frame = pd.DataFrame(rows, columns=_columns(rows, _SPAN_COLUMNS))
    for column in ("start_time", "end_time"):
        frame[column] = pd.to_datetime(frame[column], unit="ns", utc=True)
    frame["duration"] = pd.to_numeric(frame["duration"])
    return frame


def load_events(
    root_dir: str | os.PathLike[str], name: str | None = None
) -> pd.DataFrame:
    """Read the span events of every run under a directory.

    Parameters
    ----------
    root_dir : str or path-like
        The `FileTracerProvider` ``root_dir``.
    name : str, optional
        Keep only events of this name, e.g. ``"exception"``; None keeps
        all.

    Returns
    -------
    DataFrame
        One row per event, in the order they were added within each run:
        ``run``, ``span`` (the name of its span), ``trace_id``,
        ``span_id``, ``name``, ``time`` (a UTC timestamp), then a column
        per attribute.
    """
    rows = []
    for run, spans, events in _read_runs(root_dir):
        for event in events:
            if name is not None and event["name"] != name:
                continue
            span = spans.get(event["span_id"], {})
            row = {
                "run": run,
                "span": span.get("name"),
                "trace_id": event["trace_id"],
                "span_id": event["span_id"],
                "name": event["name"],
                "time": event["time"],
            }
            for key, value in event.get("attributes", {}).items():
                row[f"attributes.{key}" if key in row else key] = value
            rows.append(row)
    frame = pd.DataFrame(rows, columns=_columns(rows, _EVENT_COLUMNS))
    frame["time"] = pd.to_datetime(frame["time"], unit="ns", utc=True)
    return frame


def _read_runs(
    root_dir: str | os.PathLike[str],
) -> Iterator[tuple[str, dict[str, dict], list[dict]]]:
    """The name, spans by id and events of each run; each span holds the
    resource of the process (file) that recorded it."""
    root_dir = Path(root_dir)
    for run_dir in find_run_dirs(root_dir):
        records = RunRecords()
        for path in span_files(run_dir):
            resource: dict[str, Any] = {}
            with open(path, encoding="utf-8") as f:
                for line in f:
                    record = parse_line(line)
                    if record is None:
                        continue
                    if record.get("type") == "resource":
                        resource = record.get("attributes", {})
                    else:
                        records.add(record, resource)
        run = run_dir.relative_to(root_dir).as_posix()
        yield run, records.spans, records.events


def _columns(rows: list[dict], first: tuple[str, ...]) -> list[str]:
    # the fixed columns, then the others in the order they appear
    return list(dict.fromkeys([*first, *(k for row in rows for k in row)]))
