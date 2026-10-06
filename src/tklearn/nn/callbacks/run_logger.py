from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import psutil
import torch
from torch.optim.lr_scheduler import LRScheduler

import tklearn
from tklearn.nn.callbacks.base import Callback

if TYPE_CHECKING:
    from tklearn.nn.trainer import Trainer

__all__ = [
    "RunLogger",
    "load_runs",
]

# recorded in config.json, when set
_SLURM_VARIABLES = (
    "SLURM_JOB_ID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_NODELIST",
    "SLURM_NNODES",
    "SLURM_RESTART_COUNT",
    "SLURM_SUBMIT_DIR",
)


class RunLogger(Callback):
    """Record a run as plain files: its configuration and a log of events.

    Each run is a directory, ``log_dir/name``, holding

    - ``config.json``: the given `config`, the model, optimizer, trainer
      settings and the environment (host, versions, git commit, SLURM
      job), rewritten at the start of every `fit`;
    - ``events-<time>-<host>-<pid>.jsonl``: one JSON record per line,
      with an ``event`` and a Unix ``time``:

      - ``"fit_begin"``: ``epochs``, ``num_batches``, ``host``,
        ``attempt`` (``SLURM_RESTART_COUNT``) and ``num_processes``;
      - ``"step"``, every `log_every_n_steps` optimizer steps: ``step``,
        ``epoch``, the mean training ``loss`` and logged terms since the
        last record, ``lr`` (``lr_0``, ``lr_1``, ... with several
        parameter groups), ``grad_norm`` (before clipping),
        ``step_time`` (seconds per step, without evaluation) and
        ``memory_gb``;
      - ``"epoch"``: ``epoch``, ``step``, ``epoch_time`` and the epoch's
        logs, as in `Trainer.history`;
      - ``"evaluate"``: the results of `evaluate` called outside `fit`;
      - ``"fit_end"``: ``step``, ``epoch``, ``stopped`` and ``elapsed``.

    `load_runs` reads them into a DataFrame.

    The files suit shared storage without a database, such as NFS used
    by SLURM jobs on several machines. Only the main process writes, and
    each writing process has its own events file, so jobs never write to
    the same file: a requeued job, or another job given the same name,
    adds a file to the run instead. Each record is appended by opening,
    writing and closing the file, which makes it visible to other
    machines (NFS close-to-open consistency); avoid logging every few
    milliseconds.

    Parameters
    ----------
    log_dir : str or path-like, default="runs"
        Directory of the runs.
    name : str, optional
        Name of the run's directory; may contain ``/`` to group runs.
        Defaults to the SLURM job (``<array job>_<task>`` in a job array,
        so a requeued job continues its run), or else
        ``<time>-<host>-<pid>``.
    log_every_n_steps : int, default=50
        Optimizer steps between ``"step"`` records; 0 records epochs only.
    config : Mapping, optional
        Settings to record in ``config.json``, e.g. the model name,
        dataset and batch size. Values must be JSON serializable; others
        are recorded as strings.

    Notes
    -----
    ``memory_gb`` is the main process's peak allocated CUDA memory since
    the last record, the allocated MPS memory, or on the CPU the
    resident memory of the process.

    Computing ``grad_norm`` without `Trainer.max_grad_norm` reads every
    gradient on the steps that are recorded. Like `ModelCheckpoint`, it
    needs the full gradients on the main process, so it does not support
    FSDP or DeepSpeed ZeRO-3.

    Examples
    --------
    >>> logger = RunLogger(
    ...     "/nfs/project/runs", config={"model": "roberta-base", "lr": 2e-5}
    ... )
    >>> trainer = Trainer(model, optimizer, callbacks=[logger])
    >>> trainer.fit(train_loader, valid_loader, epochs=3)
    >>> load_runs("/nfs/project/runs", event="epoch")
    """

    def __init__(
        self,
        log_dir: str | os.PathLike[str] = "runs",
        name: str | None = None,
        *,
        log_every_n_steps: int = 50,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(log_every_n_steps, bool) or log_every_n_steps < 0:
            msg = (
                "log_every_n_steps must be a non-negative number of steps, "
                f"got {log_every_n_steps!r}"
            )
            raise ValueError(msg)
        self.log_dir = os.fspath(log_dir)
        self.name = default_run_name() if name is None else name
        self.log_every_n_steps = log_every_n_steps
        self.config = dict(config or {})
        # chosen on the first write
        self._events_path: Path | None = None
        self._in_fit = False
        self._test_depth = 0
        self._grad_norm: torch.Tensor | None = None
        self._reset_window(time.perf_counter(), 0)

    @property
    def run_dir(self) -> Path:
        """Directory of the run."""
        return Path(self.log_dir) / self.name

    # --- fit ---------------------------------------------------------------

    def on_train_begin(self, trainer: Trainer) -> None:
        self._in_fit = True
        now = time.perf_counter()
        self._fit_start = now
        self._reset_window(now, 0)
        self._grad_norm = None
        if not trainer.accelerator.is_main_process:
            return
        device = trainer.accelerator.device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        self._write_config(trainer)
        self._write(
            "fit_begin",
            epochs=trainer.epochs,
            num_batches=trainer.num_batches,
            host=socket.gethostname(),
            attempt=int(os.environ.get("SLURM_RESTART_COUNT", 0)),
            num_processes=trainer.accelerator.num_processes,
        )

    def on_epoch_begin(self, trainer: Trainer) -> None:
        self._epoch_start = time.perf_counter()

    def on_before_optimizer_step(self, trainer: Trainer) -> None:
        # the step about to be taken is global_step + 1
        if not self._is_logged_step(trainer.global_step + 1):
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
            and self._is_logged_step(step)
        ):
            self._log_step(trainer)

    def on_epoch_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:
        if trainer.accelerator.is_main_process:
            self._write(
                "epoch",
                logs,
                epoch=trainer.epoch,
                step=trainer.global_step,
                epoch_time=time.perf_counter() - self._epoch_start,
            )

    def on_train_end(self, trainer: Trainer) -> None:
        self._in_fit = False
        if trainer.accelerator.is_main_process:
            self._write(
                "fit_end",
                step=trainer.global_step,
                epoch=trainer.epoch,
                stopped=trainer.should_stop,
                elapsed=time.perf_counter() - self._fit_start,
            )

    # --- evaluate ----------------------------------------------------------

    def on_test_begin(self, trainer: Trainer) -> None:
        if self._test_depth == 0:
            self._test_start = time.perf_counter()
        self._test_depth += 1

    def on_test_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:
        self._test_depth -= 1
        if self._test_depth > 0:
            return
        # step_time leaves out evaluation during fit
        self._paused += time.perf_counter() - self._test_start
        if not self._in_fit and trainer.accelerator.is_main_process:
            self._write("evaluate", logs, step=trainer.global_step)

    # --- helpers -----------------------------------------------------------

    def _is_logged_step(self, step: int) -> bool:
        n = self.log_every_n_steps
        return n > 0 and step % n == 0

    def _reset_window(self, start: float, step: int) -> None:
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._window_start = start
        self._window_step = step
        self._paused = 0.0

    def _log_step(self, trainer: Trainer) -> None:
        now = time.perf_counter()
        step = trainer.global_step
        # reduces over processes, so every process takes part
        means = trainer._mean(self._totals, self._counts)
        step_time = (now - self._window_start - self._paused) / (
            step - self._window_step
        )
        grad_norm = self._grad_norm
        self._reset_window(now, step)
        self._grad_norm = None
        if not trainer.accelerator.is_main_process:
            return
        groups = trainer.optimizer.param_groups
        if len(groups) == 1:
            lrs = {"lr": float(groups[0]["lr"])}
        else:
            lrs = {f"lr_{i}": float(g["lr"]) for i, g in enumerate(groups)}
        self._write(
            "step",
            means,
            lrs,
            step=step,
            epoch=trainer.epoch,
            grad_norm=None if grad_norm is None else float(grad_norm),
            step_time=step_time,
            memory_gb=_memory_gb(trainer.accelerator.device),
        )

    def _write(
        self, event: str, *values: Mapping[str, Any], **fields: Any
    ) -> None:
        if self._events_path is None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            host = socket.gethostname().split(".")[0]
            filename = f"events-{stamp}-{host}-{os.getpid()}.jsonl"
            self._events_path = self.run_dir / filename
        record: dict[str, Any] = {}
        for value in values:
            record.update(value)
        # the event's own fields win over logs of the same name
        record.update(fields, event=event, time=time.time())
        line = json.dumps(record, default=_to_json) + "\n"
        # closing after each record publishes it to other NFS clients
        with open(self._events_path, "a", encoding="utf-8") as f:
            f.write(line)

    def _write_config(self, trainer: Trainer) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "name": self.name,
            "updated": datetime.now(timezone.utc).isoformat(),
            "config": self.config,
            "model": _describe_model(trainer.model),
            "optimizer": _describe_optimizer(trainer.optimizer),
            "trainer": _describe_trainer(trainer),
            "environment": _describe_environment(),
        }
        text = json.dumps(config, indent=2, default=_to_json)
        # write a temporary file and rename it, which is atomic, so that
        # readers never see a partial config
        path = self.run_dir / "config.json"
        tmp = path.with_name(f".config-{socket.gethostname()}-{os.getpid()}")
        tmp.write_text(text + "\n", encoding="utf-8")
        os.replace(tmp, path)


def default_run_name() -> str:
    """The SLURM job, ``<array job>_<task>`` in an array, or else
    ``<time>-<host>-<pid>``."""
    env = os.environ
    if "SLURM_ARRAY_JOB_ID" in env and "SLURM_ARRAY_TASK_ID" in env:
        return f"{env['SLURM_ARRAY_JOB_ID']}_{env['SLURM_ARRAY_TASK_ID']}"
    if "SLURM_JOB_ID" in env:
        return env["SLURM_JOB_ID"]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    host = socket.gethostname().split(".")[0]
    return f"{stamp}-{host}-{os.getpid()}"


def load_runs(
    log_dir: str | os.PathLike[str],
    event: str | None = "epoch",
    *,
    config: bool = True,
) -> pd.DataFrame:
    """Read the records of every run under a directory.

    Parameters
    ----------
    log_dir : str or path-like
        The `RunLogger` ``log_dir``.
    event : str, optional
        Keep only records of this event (``"step"``, ``"epoch"``,
        ``"evaluate"``, ...); None keeps all.
    config : bool, default=True
        Add the `RunLogger` ``config`` of each run as ``config.<key>``
        columns, e.g. to compare the runs of a sweep.

    Returns
    -------
    DataFrame
        One row per record, with the run's name in a ``run`` column, in
        the order they were written within each run. A line cut off by a
        job that was killed while writing is skipped.
    """
    log_dir = Path(log_dir)
    rows = []
    run_dirs = sorted({p.parent for p in log_dir.rglob("events-*.jsonl")})
    for run_dir in run_dirs:
        run = run_dir.relative_to(log_dir).as_posix()
        settings = _read_settings(run_dir) if config else {}
        # file names start with the time they were created
        for path in sorted(run_dir.glob("events-*.jsonl")):
            for record in _read_records(path):
                if event is None or record.get("event") == event:
                    rows.append({"run": run, **settings, **record})
    return pd.DataFrame(rows)


def _read_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _read_settings(run_dir: Path) -> dict[str, Any]:
    try:
        text = (run_dir / "config.json").read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    settings = json.loads(text).get("config", {})
    return {f"config.{k}": v for k, v in settings.items()}


def _grad_norm(model: torch.nn.Module) -> torch.Tensor | None:
    norms = [
        torch.linalg.vector_norm(p.grad.detach(), dtype=torch.float32)
        for p in model.parameters()
        if p.grad is not None
    ]
    if not norms:
        return None
    return torch.linalg.vector_norm(torch.stack(norms))


def _memory_gb(device: torch.device) -> float:
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        return peak / 1e9
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1e9
    return psutil.Process().memory_info().rss / 1e9


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
        "lr_scheduler": scheduler,
        "warmup": trainer.warmup,
        "max_grad_norm": trainer.max_grad_norm,
        "mixed_precision": accelerator.mixed_precision,
        "gradient_accumulation_steps": accelerator.gradient_accumulation_steps,
        "num_processes": accelerator.num_processes,
        "device": str(accelerator.device),
        "metrics": list(trainer.metrics),
        "callbacks": [_qualname(type(c)) for c in trainer.callbacks],
    }


def _describe_environment() -> dict[str, Any]:
    slurm = {
        name: os.environ[name]
        for name in _SLURM_VARIABLES
        if name in os.environ
    }
    return {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "tklearn": tklearn.__version__,
        "git_commit": _git_commit(),
        "slurm": slurm,
    }


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


def _qualname(obj: Any) -> str:
    module = getattr(obj, "__module__", None)
    name = getattr(obj, "__qualname__", None) or repr(obj)
    return f"{module}.{name}" if module else name


def _to_json(value: Any) -> Any:
    # tensors and numpy values as numbers and lists; anything else as text
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)
