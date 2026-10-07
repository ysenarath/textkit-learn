from __future__ import annotations

import fnmatch
import math
import os
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tklearn.tracing._records import (
    RunRecords,
    find_run_dirs,
    parse_line,
    span_files,
)

__all__ = [
    "RunMonitor",
    "RunStatus",
    "snapshot",
]

# SLURM job states in which a job is still alive
_ALIVE_SLURM_STATES = frozenset({
    "PENDING",
    "CONFIGURING",
    "RUNNING",
    "COMPLETING",
    "REQUEUED",
    "RESIZING",
    "SUSPENDED",
    "STAGE_OUT",
})


@dataclass
class RunStatus:
    """The state of a run's latest `fit`, as its spans show it.

    Attributes
    ----------
    run : str
        The run's directory, relative to the monitored directory.
    status : str
        ``"running"``; ``"stalled"`` (no record for `stale_after`
        seconds); ``"finished"``; ``"stopped"`` (early, by a callback);
        ``"failed"`` (the fit span ended with an error, or, when SLURM is
        asked, its job ended without ending it, see `slurm_state`); or
        ``"unknown"`` (no fit span yet).
    epoch : int or None
        Zero-based epoch in progress, or the last one when finished.
    epochs : int or None
        Maximum number of epochs of the fit.
    step : int or None
        Optimizer steps of the latest ``"steps"`` span, or of the fit
        when it ended.
    total_steps : int or None
        Optimizer steps of all `epochs`, when the batches per epoch are
        known.
    loss : float or None
        Mean training loss of the latest ``"steps"`` span.
    step_time : float or None
        Seconds per optimizer step of the latest ``"steps"`` span,
        without evaluation.
    eta : float or None
        Seconds until the last epoch ends while running, from
        `step_time`, without evaluation.
    metrics : dict
        Logs of the latest epoch that ended, e.g. ``valid_loss``.
    start_time : float or None
        Unix time the fit started.
    last_time : float or None
        Unix time of the run's latest record.
    slurm_job_id : str or None
        SLURM job of the process that runs the fit.
    slurm_state : str or None
        Its ``sacct`` state, when asked for.
    """

    run: str
    status: str = "unknown"
    epoch: int | None = None
    epochs: int | None = None
    step: int | None = None
    total_steps: int | None = None
    loss: float | None = None
    step_time: float | None = None
    eta: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    start_time: float | None = None
    last_time: float | None = None
    slurm_job_id: str | None = None
    slurm_state: str | None = None

    @property
    def progress(self) -> float | None:
        """Fraction of `total_steps` done, from 0 to 1."""
        if self.step is None or not self.total_steps:
            return None
        return min(self.step / self.total_steps, 1.0)


class RunMonitor:
    """Follow the runs under a directory as `FileTracerProvider` writes.

    Each `refresh` reads only the lines added to the span files since the
    last one; a line still being written is read once it is complete.
    Runs are those directories, at any depth, that hold span files;
    new ones are looked for every `rescan_interval` seconds. Reading the
    files from another process or machine is safe: the providers only
    append to them.

    Parameters
    ----------
    root_dir : str or path-like
        Directory of the runs, e.g. the `FileTracerProvider` ``root_dir``.
    pattern : str, optional
        Follow only runs whose path relative to `root_dir` matches this
        glob, e.g. ``"task-set-1/*"``.
    stale_after : float, default=300
        Seconds without a record after which a running fit is
        ``"stalled"``.
    rescan_interval : float, default=30
        Seconds between looks for new runs; 0 looks on every refresh.
    slurm : bool, default=False
        Ask ``sacct`` for the state of the SLURM jobs of open fits; a fit
        whose job is no longer alive, e.g. after a ``TIMEOUT``, is
        ``"failed"``.

    Examples
    --------
    >>> monitor = RunMonitor("exp/outputs")
    >>> for status in monitor.refresh():
    ...     print(status.run, status.status, status.progress)
    """

    def __init__(
        self,
        root_dir: str | os.PathLike[str],
        pattern: str | None = None,
        *,
        stale_after: float = 300.0,
        rescan_interval: float = 30.0,
        slurm: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.pattern = pattern
        self.stale_after = stale_after
        self.rescan_interval = rescan_interval
        self.slurm = slurm
        self._runs: dict[str, _FollowedRun] = {}
        self._last_scan: float | None = None

    def refresh(self, now: float | None = None) -> list[RunStatus]:
        """Read what was added and give the state of each run, sorted by
        run.

        Parameters
        ----------
        now : float, optional
            Unix time to judge staleness against; defaults to the
            current time.
        """
        clock = time.monotonic()
        if (
            self._last_scan is None
            or clock - self._last_scan >= self.rescan_interval
        ):
            self._scan()
            self._last_scan = clock
        for run in self._runs.values():
            run.read()
        statuses = [
            _status(name, run.records) for name, run in self._runs.items()
        ]
        now = time.time() if now is None else now
        for status in statuses:
            if (
                status.status == "running"
                and status.last_time is not None
                and now - status.last_time > self.stale_after
            ):
                status.status = "stalled"
                status.eta = None
        if self.slurm:
            _check_slurm(statuses)
        return sorted(statuses, key=lambda s: s.run)

    def _scan(self) -> None:
        for run_dir in find_run_dirs(self.root_dir):
            name = run_dir.relative_to(self.root_dir).as_posix()
            if self.pattern is not None and not fnmatch.fnmatchcase(
                name, self.pattern
            ):
                continue
            if name not in self._runs:
                self._runs[name] = _FollowedRun(run_dir)
            self._runs[name].scan()


def snapshot(
    root_dir: str | os.PathLike[str],
    pattern: str | None = None,
    *,
    stale_after: float = 300.0,
    slurm: bool = False,
) -> list[RunStatus]:
    """The state of each run under a directory, once.

    Parameters are those of `RunMonitor`.
    """
    monitor = RunMonitor(
        root_dir, pattern, stale_after=stale_after, slurm=slurm
    )
    return monitor.refresh()


class _FollowedRun:
    """A run whose span files are read as they grow."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.records = RunRecords()
        self._files: dict[Path, _FollowedFile] = {}

    def scan(self) -> None:
        # a requeued job, or another process, adds a file
        for path in span_files(self.run_dir):
            if path not in self._files:
                self._files[path] = _FollowedFile(path)

    def read(self) -> None:
        for file in self._files.values():
            file.read(self.records)


class _FollowedFile:
    """A span file, read from where the last read stopped."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.resource: dict[str, Any] = {}

    def read(self, records: RunRecords) -> None:
        try:
            # opening the file again shows other NFS clients' writes
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                data = f.read()
        except FileNotFoundError:
            return
        # a line without its newline is still being written
        end = data.rfind(b"\n") + 1
        self.offset += end
        for line in data[:end].decode("utf-8").splitlines():
            record = parse_line(line)
            if record is None:
                continue
            if record.get("type") == "resource":
                self.resource = record.get("attributes", {})
            else:
                records.add(record, self.resource)


def _status(run: str, records: RunRecords) -> RunStatus:
    status = RunStatus(run)
    if records.last_time is not None:
        status.last_time = records.last_time / 1e9
    fits = [s for s in records.spans.values() if s["name"] == "fit"]
    if not fits:
        return status
    fit = max(fits, key=lambda s: s["start_time"])
    attributes = fit["attributes"]
    status.start_time = fit["start_time"] / 1e9
    status.slurm_job_id = fit["resource"].get("slurm.job.id")
    status.epochs = _int(attributes.get("trainer.epochs"))
    batches = _int(attributes.get("trainer.num_batches"))
    accumulation = _int(attributes.get("trainer.gradient_accumulation_steps"))
    if status.epochs is not None and batches is not None:
        steps_per_epoch = math.ceil(batches / (accumulation or 1))
        status.total_steps = status.epochs * steps_per_epoch
    children = [
        s for s in records.spans.values() if s["parent_id"] == fit["span_id"]
    ]
    epochs = sorted(
        (s for s in children if s["name"] == "epoch"),
        key=lambda s: s["start_time"],
    )
    if epochs:
        status.epoch = _int(epochs[-1]["attributes"].get("epoch"))
        ended = [s for s in epochs if "end_time" in s]
        if ended:
            status.metrics = {
                k: v
                for k, v in ended[-1]["attributes"].items()
                if k not in ("epoch", "step")
            }
    steps = sorted(
        (s for s in children if s["name"] == "steps" and "end_time" in s),
        key=lambda s: s["end_time"],
    )
    if steps:
        latest = steps[-1]["attributes"]
        status.step = _int(latest.get("step"))
        status.loss = _float(latest.get("loss"))
        status.step_time = _float(latest.get("step_time"))
    if "end_time" in fit:
        status.step = _int(fit["attributes"].get("step", status.step))
        if fit["status"] == "ERROR":
            status.status = "failed"
        elif fit["attributes"].get("stopped"):
            status.status = "stopped"
        else:
            status.status = "finished"
        return status
    status.status = "running"
    if (
        status.total_steps is not None
        and status.step is not None
        and status.step_time is not None
    ):
        remaining = max(status.total_steps - status.step, 0)
        status.eta = remaining * status.step_time
    return status


def _check_slurm(statuses: list[RunStatus]) -> None:
    open_fits = [
        s
        for s in statuses
        if s.status in ("running", "stalled") and s.slurm_job_id is not None
    ]
    states = _slurm_states(s.slurm_job_id for s in open_fits)
    for status in open_fits:
        state = states.get(status.slurm_job_id)
        status.slurm_state = state
        if state is not None and state not in _ALIVE_SLURM_STATES:
            # e.g. TIMEOUT, CANCELLED or OUT_OF_MEMORY: it will not finish
            status.status = "failed"
            status.eta = None


def _slurm_states(job_ids: Iterable[str]) -> dict[str, str]:
    """The ``sacct`` state of each job; empty without ``sacct``."""
    job_ids = sorted(set(job_ids))
    if not job_ids:
        return {}
    try:
        result = subprocess.run(
            [
                "sacct",
                "--noheader",
                "--parsable2",
                "--format=JobIDRaw,State",
                "--jobs",
                ",".join(job_ids),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    states = {}
    for line in result.stdout.splitlines():
        job_id, _, state = line.partition("|")
        # job steps, e.g. "123.batch", have their own lines
        if job_id in job_ids and state:
            # e.g. "CANCELLED by 1000"
            states[job_id] = state.split()[0]
    return states


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
