"""``tklearn runs``: follow the runs that
`tklearn.tracing.FileTracerProvider` records."""

from __future__ import annotations

import numbers
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from tklearn.tracing.monitor import RunStatus

__all__ = [
    "runs",
]

_STATUS_STYLES = {
    "running": "green",
    "stalled": "yellow",
    "finished": "blue",
    "stopped": "cyan",
    "failed": "bold red",
    "unknown": "dim",
}


@click.group()
def runs() -> None:
    """Follow the runs recorded by tklearn.tracing.FileTracerProvider."""


@runs.command()
@click.argument(
    "root_dir",
    default=".",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--filter",
    "pattern",
    metavar="GLOB",
    help="Only runs whose path under ROOT_DIR matches, e.g. 'task-set-1/*'.",
)
@click.option(
    "-m",
    "--metric",
    "metrics",
    multiple=True,
    metavar="NAME",
    help="Epoch metric to show; repeat for more. "
    "Defaults to the valid_* metrics.",
)
@click.option(
    "-n",
    "--interval",
    default=5.0,
    show_default=True,
    type=click.FloatRange(min=0.1),
    help="Seconds between refreshes.",
)
@click.option(
    "--stale-after",
    default=300.0,
    show_default=True,
    type=click.FloatRange(min=0),
    help="Seconds without a record after which a running fit is stalled.",
)
@click.option(
    "--slurm",
    is_flag=True,
    help="Ask sacct for the jobs' states; a fit whose job died is failed.",
)
@click.option("--once", is_flag=True, help="Print the table once and exit.")
def monitor(
    root_dir: Path,
    pattern: str | None,
    metrics: tuple[str, ...],
    interval: float,
    stale_after: float,
    slurm: bool,
    once: bool,
) -> None:
    """Show the progress of the runs under ROOT_DIR, at any depth.

    A run is a directory of span files, e.g. exp/outputs/task-set-1/task-1.
    Its latest fit gives the status, epoch, step, training loss, the
    metrics of the latest epoch and an ETA. Other processes and machines
    may write while it reads.
    """
    # imported here, so that `tklearn --help` does not load tracing
    from tklearn.tracing.monitor import RunMonitor

    follower = RunMonitor(
        root_dir, pattern, stale_after=stale_after, slurm=slurm
    )
    console = Console()
    if once:
        console.print(_table(follower.refresh(), root_dir, metrics, slurm))
        return
    with Live(console=console, auto_refresh=False) as live:
        try:
            while True:
                table = _table(follower.refresh(), root_dir, metrics, slurm)
                live.update(table, refresh=True)
                time.sleep(interval)
        except KeyboardInterrupt:
            pass


def _table(
    statuses: list[RunStatus],
    root_dir: Path,
    metrics: tuple[str, ...],
    slurm: bool,
) -> Table:
    now = time.time()
    metrics = metrics or _default_metrics(statuses)
    counts: dict[str, int] = {}
    for status in statuses:
        counts[status.status] = counts.get(status.status, 0) + 1
    summary = ", ".join(f"{n} {name}" for name, n in sorted(counts.items()))
    noun = "run" if len(statuses) == 1 else "runs"
    table = Table(
        title=f"{len(statuses)} {noun}" + (f": {summary}" if summary else ""),
        caption=f"{root_dir} at {time.strftime('%H:%M:%S')}",
    )
    # run paths wrap rather than being cut; values are never broken
    table.add_column("run", overflow="fold", min_width=12)
    table.add_column("status", no_wrap=True)
    if slurm:
        table.add_column("slurm", no_wrap=True)
    for name in ("epoch", "step", "done", "loss", *metrics, "ETA", "updated"):
        table.add_column(name, justify="right", no_wrap=True)
    group = None
    for status in statuses:
        # a section per top-level folder, e.g. task-set-1/...
        top = status.run.split("/")[0]
        if group is not None and top != group:
            table.add_section()
        group = top
        style = _STATUS_STYLES.get(status.status, "")
        row = [status.run, Text(status.status, style=style)]
        if slurm:
            row.append(status.slurm_state or "-")
        row += [
            _fraction(
                None if status.epoch is None else status.epoch + 1,
                status.epochs,
            ),
            _fraction(status.step, status.total_steps),
            "-" if status.progress is None else f"{status.progress:.0%}",
            _number(status.loss),
            *(_number(status.metrics.get(m)) for m in metrics),
            _duration(status.eta),
            "-"
            if status.last_time is None
            else f"{_duration(now - status.last_time)} ago",
        ]
        table.add_row(*row)
    return table


def _default_metrics(statuses: list[RunStatus]) -> tuple[str, ...]:
    # the numeric valid_* metrics, in the order they first appear
    names: dict[str, None] = {}
    for status in statuses:
        for key, value in status.metrics.items():
            if key.startswith("valid_") and _is_number(value):
                names[key] = None
    return tuple(names)


def _fraction(value: int | None, total: int | None) -> str:
    if value is None:
        return "-"
    return f"{value}/{total}" if total is not None else str(value)


def _number(value: Any) -> str:
    if value is None:
        return "-"
    if _is_number(value):
        return f"{value:.4g}"
    return str(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = max(int(round(seconds)), 0)
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"
