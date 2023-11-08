from __future__ import annotations
from ast import Tuple
import enum
from collections.abc import Mapping
import contextlib
from datetime import datetime
import errno
import hashlib
import json
import math
import os
from typing import Any, Optional, Union, Dict
from pathlib import Path
import uuid
import filelock
import pyarrow as pa

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from plotly import graph_objects as go


__all__ = [
    "Run",
    "Experiment",
    "ViewType",
]


def flatten(data: Dict[str, Any], parent_key="", separator="."):
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


def create_dir_if_not_exist(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            # File exists, and it's a directory,
            # another process beat us to creating this dir, that's OK.
            pass
        else:
            # Our target dir exists as a file, or different error,
            # reraise the error!
            raise


def explore_recursively(
    run_path: Path,
    namespace: str = "",
):
    data = []
    run_id = str(run_path).split("/")[-1]
    subpath = run_path / "metadata.json"
    with open(subpath, "r", encoding="utf-8") as fp:
        metadata = json.load(fp)
        run_name = metadata["name"]
    namespace = namespace + "." + run_name if namespace else run_name
    for file in os.listdir(run_path):
        subpath = run_path / file
        if file == "metadata.json":
            continue
        elif subpath.is_dir() and file.startswith("run-"):
            data += explore_recursively(
                subpath,
                namespace=namespace,
            )
        elif file.endswith(".json"):
            with open(subpath, "r", encoding="utf-8") as fp:
                d = json.load(fp)
                d["metadata.run.id"] = run_id
                d["metadata.run.namespace"] = namespace
                data.append(d)
    return data


class ArtifactType(enum.Enum):
    PATH = 0
    JSON = 1
    TABLE = 2
    FIGURE = 3
    IMAGE = 4
    HTML = 5
    TEXT = 6


class Artifact(object):
    def __init__(self, obj, type: Optional[ArtifactType] = None) -> None:
        self.obj = obj
        if type is None:
            try:
                objp = Path(obj).exists()
            except Exception as ex:
                objp = False
            if isinstance(obj, Path) or objp:
                type = ArtifactType.PATH
            elif isinstance(obj, str):
                type = ArtifactType.TEXT
            elif isinstance(obj, dict):
                type = ArtifactType.JSON
            elif isinstance(obj, pd.DataFrame):
                type = ArtifactType.TABLE
            elif isinstance(obj, (Figure, go.Figure)):
                type = ArtifactType.FIGURE
            elif isinstance(obj, (np.ndarray,)):
                type = ArtifactType.IMAGE
            elif isinstance(obj, (np.ndarray,)):
                type = ArtifactType.IMAGE
            else:
                type = None
        self.type = type


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
            create_dir_if_not_exist(self.run_path)
            with self.experiment.lock():
                with open(
                    self.run_path / "metadata.json",
                    "w",
                    encoding="utf-8",
                ) as fp:
                    json.dump(
                        {
                            "name": name,
                        },
                        fp,
                    )
        else:
            self.id = id

    @property
    def run_path(self) -> Path:
        path = self.experiment.tracking_uri
        for parent_id in self.parent_ids:
            path = path / parent_id
        return path / self.id

    def _log_data(self, **kwargs):
        if "timestamp" not in kwargs or kwargs["timestamp"] is None:
            kwargs["timestamp"] = math.ceil(datetime.now().timestamp())
        with self.experiment.lock():
            path = self.run_path / f"{uuid.uuid4().hex}.json"
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(kwargs, fp)

    def get_logs(self):
        data = explore_recursively(self.run_path)
        return pd.DataFrame(data)

    def log_metric(
        self,
        key: str,
        value: Any,
        *,
        timestamp: Optional[int] = None,
        step: Optional[int] = None,
    ) -> None:
        """Log a metric to MLFlow."""
        for key, value in flatten({key: value}).items():
            self._log_data(
                type="metric",
                key=key,
                value=value,
                step=step,
                timestamp=timestamp,
            )

    def log_param(
        self,
        key: str,
        value: Any,
        step: Optional[int] = None,
    ) -> None:
        """Log a parameter to MLFlow."""
        for key, value in flatten({key: value}).items():
            self._log_data(
                type="parameter",
                key=key,
                value=value,
            )

    def set_tag(self, key: str, value: Any) -> None:
        """Set tags for the run."""
        for key, value in flatten({key: value}).items():
            self._log_data(
                type="tag",
                key=key,
                value=value,
            )

    def log_metrics(
        self,
        metrics: dict,
        *,
        timestamp: Optional[int] = None,
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to MLFlow."""
        for key, value in metrics.items():
            self.log_metric(key, value, timestamp=timestamp, step=step)

    def log_params(
        self,
        params: dict,
        step: Optional[int] = None,
    ) -> None:
        """Log parameters to MLFlow."""
        for key, value in params.items():
            self.log_param(key, value)

    def set_tags(self, tags: dict) -> None:
        """Set tags for the run."""
        for key, value in tags.items():
            self.set_tag(key, value)

    def log_artifact(
        self,
        artifact: Union[Artifact, Any],
        artifact_path: Optional[str] = None,
        artifact_type: Optional[ArtifactType] = None,
    ):
        # if not isinstance(artifact, Artifact):
        #     artifact = Artifact(artifact, type=artifact_type)
        # obj = artifact.obj
        raise NotImplementedError

    def start_run(self, run_name: Optional[str] = None):
        """Start a new run."""
        return Run(self.experiment, run_name, parent_ids=self.parent_ids + (self.id,))


class Experiment(object):
    """A run of an experiment."""

    def __init__(
        self,
        path: Union[str, Path],
        name: Union[str, None] = None,
        version: Optional[str] = None,
    ) -> None:
        if name is None:
            name = "main"
        self.path = Path(path)
        self.experiment_id = hashlib.sha256(name.encode()).hexdigest()
        create_dir_if_not_exist(self.tracking_uri)
        with self.lock():
            with open(
                self.tracking_uri / "metadata.json",
                "w",
                encoding="utf-8",
            ) as fp:
                json.dump(
                    {
                        "name": name,
                        "version": version,
                    },
                    fp,
                )

    @property
    def tracking_uri(self) -> Path:
        return self.path / f"expr-{self.experiment_id}"

    def start_run(self, run_name: Optional[str] = None):
        """Start a new run."""
        return Run(self, run_name)

    @contextlib.contextmanager
    def lock(self):
        yield filelock.FileLock(self.tracking_uri / "experiment.lock")

    def start_run(self, run_name: Optional[str] = None):
        """Start a new run."""
        return Run(self, run_name)

    def list_runs(self):
        return [
            Run(self, id=run_id)
            for run_id in os.listdir(self.tracking_uri)
            if run_id.startswith("run-")
        ]
