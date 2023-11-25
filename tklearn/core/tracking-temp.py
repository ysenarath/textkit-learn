from __future__ import annotations
from ast import Tuple
import hashlib
from typing import Any, Dict, Mapping, Optional, Union
from pathlib import Path
import uuid

import tensorflow as tf

__all__ = [
    "Run",
    "Experiment",
    "ViewType",
]


def flatten(
    data: Dict[str, Any],
    parent_key: str = "",
    separator: str = ".",
):
    items = []
    for key, value in data.items():
        # escape dots
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, Mapping):
            items.extend(
                flatten(
                    value,
                    parent_key=new_key,
                    separator=separator,
                ).items()
            )
        else:
            items.append((new_key, value))
    return dict(items)


class Run(object):
    def __init__(
        self,
        experiment: Experiment,
        name: str = None,
        parent_ids: Optional[Tuple[str]] = None,
        id: str = None,
    ) -> None:
        self.experiment = experiment
        self.parent_ids = parent_ids or tuple()
        if id is None:
            self.id = f"run-{uuid.uuid4().hex}"
            name = "default" if name is None else name
        else:
            self.id = id

    @property
    def run_path(self) -> Path:
        path = self.experiment.tracking_uri
        for parent_id in self.parent_ids:
            path = path / parent_id
        return path / self.id

    @property
    def writer(self) -> tf.summary.SummaryWriter:
        if not hasattr(self, "_writer") or self._writer is None:
            self._writer = tf.summary.create_file_writer(str(self.run_path))
        return self._writer

    def log_metric(
        self,
        key: str,
        value: Any,
        *,
        step: Optional[int] = 0,
    ) -> None:
        """Log a metric to MLFlow."""
        for key, value in flatten({key: value}).items():
            with self.writer.as_default():
                tf.summary.scalar(f"metric.{key}", value, step=step)
            self.writer.flush()

    def log_param(
        self,
        key: str,
        value: Any,
        step: Optional[int] = 0,
    ) -> None:
        """Log a parameter to MLFlow."""
        for key, value in flatten({key: value}).items():
            with self.writer.as_default():
                tf.summary.scalar(f"param.{key}", value, step=step)
            self.writer.flush()

    def set_tag(
        self,
        key: str,
        value: Any,
    ) -> None:
        """Set tags for the run."""
        for key, value in flatten({key: value}).items():
            with self.writer.as_default():
                tf.summary.scalar(f"tag.{key}", value, step=0)
            self.writer.flush()

    def log_metrics(
        self,
        metrics: dict,
        *,
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to MLFlow."""
        for key, value in metrics.items():
            self.log_metric(key, value, step=step)

    def log_params(
        self,
        params: dict,
        step: Optional[int] = None,
    ) -> None:
        """Log parameters to MLFlow."""
        for key, value in params.items():
            self.log_param(key, value, step=step)

    def set_tags(self, tags: dict) -> None:
        """Set tags for the run."""
        for key, value in tags.items():
            self.set_tag(key, value)

    def start_run(
        self,
        run_name: Optional[str] = None,
    ):
        """Start a new run."""
        return Run(self.experiment, run_name, parent_ids=self.parent_ids + (self.id,))

    def log_text(
        self,
        name: str,
        data: str,
        step: Optional[int] = None,
    ):
        with self.writer.as_default():
            tf.summary.text(name, data, step=step)
        self.writer.flush()


class Experiment(object):
    """A run of an experiment."""

    def __init__(
        self,
        path: Union[str, Path],
        name: Union[str, None] = None,
    ) -> None:
        if name is None:
            name = "main"
        self.path = Path(path)
        self.experiment_id = hashlib.sha256(name.encode()).hexdigest()

    @property
    def tracking_uri(self) -> Path:
        return self.path / f"expr-{self.experiment_id}"

    def start_run(self, run_name: str):
        """Start a new run."""
        return Run(self, run_name)
