"""Spans and events built from the records of `FileTracerProvider`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tklearn.utils.flatten import freeze

__all__ = [
    "RunRecords",
    "find_run_dirs",
    "parse_line",
    "span_files",
]


class RunRecords:
    """The spans and events of a run, built record by record.

    Each span is a dict of its ``name``, ``trace_id``, ``span_id``,
    ``parent_id``, ``start_time``, ``attributes`` and the ``resource`` of
    the process that recorded it, and once it ends, its ``end_time``,
    ``duration`` (seconds), ``status`` and ``status_description``.
    """

    def __init__(self) -> None:
        self.spans: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        #: time of the latest record, in nanoseconds since the epoch
        self.last_time: int | None = None

    def add(self, record: dict[str, Any], resource: dict[str, Any]) -> None:
        """Add a ``start``, ``end`` or ``event`` record of a process with
        the given resource."""
        kind = record.get("type")
        if kind == "start":
            self.spans[record["span_id"]] = {
                "name": record["name"],
                "trace_id": record["trace_id"],
                "span_id": record["span_id"],
                "parent_id": record.get("parent_id"),
                "start_time": record["start_time"],
                "attributes": record.get("attributes", {}),
                "resource": resource,
            }
            self._seen(record["start_time"])
        elif kind == "end" and record["span_id"] in self.spans:
            span = self.spans[record["span_id"]]
            span.update(
                name=record["name"],
                end_time=record["end_time"],
                duration=(record["end_time"] - span["start_time"]) / 1e9,
                status=record["status"],
                status_description=record.get("status_description"),
                attributes=record.get("attributes", {}),
            )
            self._seen(record["end_time"])
        elif kind == "event":
            self.events.append(record)
            self._seen(record["time"])

    def _seen(self, time: Any) -> None:
        if isinstance(time, int) and (
            self.last_time is None or time > self.last_time
        ):
            self.last_time = time


def parse_line(line: str) -> dict[str, Any] | None:
    """The record of a line, or None for a line cut off by a job that was
    killed while writing."""
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict):
        return None
    if "attributes" in record:
        # JSON arrays as tuples, as the provider kept them
        record["attributes"] = {
            k: freeze(v) for k, v in record["attributes"].items()
        }
    return record


def find_run_dirs(root_dir: Path) -> list[Path]:
    """The directories under `root_dir`, at any depth, that hold span
    files, sorted."""
    return sorted({p.parent for p in root_dir.rglob("spans-*.jsonl")})


def span_files(run_dir: Path) -> list[Path]:
    """The span files of a run, oldest first (their names start with the
    time they were created)."""
    return sorted(run_dir.glob("spans-*.jsonl"))
