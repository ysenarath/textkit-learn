from __future__ import annotations
from collections.abc import Mapping
from typing import Any, Optional, Union, Dict, List, Generator
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
from matplotlib.figure import Figure
from plotly import graph_objects as go
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.entities import (
    Experiment as MLflowExperiment,
    Run as MLflowRun,
    ViewType,
)

__all__ = [
    "Run",
    "Experiment",
    "ViewType",
]


def flatten(data: Dict[str, Any], parent_key="", separator="__"):
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


import enum


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
    def __init__(self, experiment: Experiment, mlflow_run: MLflowRun) -> None:
        self.experiment = experiment
        self._mlflow_run: MLflowRun = mlflow_run

    @property
    def id(self) -> str:
        """Get the run ID."""
        return self._mlflow_run.info.run_id

    @property
    def name(self) -> Optional[str]:
        """Get the run name."""
        return self._mlflow_run.info.run_name

    @property
    def _mlflow_client(self) -> mlflow.tracking.MlflowClient:
        return self.experiment._mlflow_client

    @property
    def _mlflow_experiment(self) -> MLflowExperiment:
        return self.experiment._mlflow_experiment

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
            self._mlflow_client.log_metric(
                self._mlflow_run.info.run_id, key, value, timestamp, step
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

    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter to MLFlow."""
        for key, value in flatten({key: value}).items():
            self._mlflow_client.log_param(
                self._mlflow_run.info.run_id,
                key,
                value,
            )

    def log_params(self, params: dict) -> None:
        """Log parameters to MLFlow."""
        for key, value in params.items():
            self.log_param(key, value)

    def set_tag(self, key: str, value: Any) -> None:
        """Set tags for the run."""
        self._mlflow_client.set_tag(self._mlflow_run.info.run_id, key, value)

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
        if not isinstance(artifact, Artifact):
            artifact = Artifact(artifact, type=artifact_type)
        obj = artifact.obj
        if artifact.type == ArtifactType.PATH:
            self._mlflow_client.log_artifact(
                self._mlflow_run.info.run_id, obj, artifact_path=artifact_path
            )
        elif artifact.type == ArtifactType.JSON:
            self._mlflow_client.log_dict(
                self._mlflow_run.info.run_id, obj, artifact_file=artifact_path
            )
        elif artifact.type == ArtifactType.TABLE:
            self._mlflow_client.log_table(
                self._mlflow_run.info.run_id, obj, artifact_file=artifact_path
            )
        elif artifact.type == ArtifactType.FIGURE:
            self._mlflow_client.log_figure(
                self._mlflow_run.info.run_id, obj, artifact_file=artifact_path
            )
        elif artifact.type == ArtifactType.IMAGE:
            self._mlflow_client.log_image(
                self._mlflow_run.info.run_id, obj, artifact_file=artifact_path
            )
        elif artifact.type == ArtifactType.HTML:
            if isinstance(obj, pd.DataFrame):
                obj = obj.to_html()
            self._mlflow_client.log_text(
                self._mlflow_run.info.run_id, obj, artifact_file=artifact_path
            )
        elif artifact.type == ArtifactType.TEXT:
            if isinstance(obj, pd.DataFrame):
                obj = obj.to_html()
            self._mlflow_client.log_text(
                self._mlflow_run.info.run_id, obj, artifact_file=artifact_path
            )
        else:
            raise NotImplementedError(f"unsupported artifact type '{artifact.type}'")

    def set_terminated(self) -> None:
        """Close the run."""
        self._mlflow_client.set_terminated(self._mlflow_run.info.run_id)

    def start_run(self, run_name: Optional[str] = None):
        """Start a new run."""
        mlflow_run = self._mlflow_client.create_run(
            self._mlflow_experiment.experiment_id,
            tags={"mlflow.parentRunId": self._mlflow_run.info.run_id},
            run_name=run_name,
        )
        return Run(self.experiment, mlflow_run)


class Experiment(object):
    """A run of an experiment."""

    def __init__(
        self,
        name: Union[str, None] = None,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ) -> None:
        self._mlflow_client = mlflow.tracking.MlflowClient(
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
        )
        # self._mlflow_client._get_registry_client()
        self._mlflow_experiment: MLflowExperiment = (
            self._mlflow_get_or_create_experiment(
                name,
                artifact_location=artifact_location,
            )
        )

    @property
    def name(self):
        return self._mlflow_experiment.name

    @property
    def tracking_uri(self) -> str:
        return self._mlflow_client.tracking_uri

    @property
    def registry_uri(self) -> str:
        return self._mlflow_client._registry_uri

    @property
    def artifact_location(self):
        return self._mlflow_experiment.artifact_location

    def _mlflow_get_or_create_experiment(
        self,
        experiment_name: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ) -> MLflowExperiment:
        """Get or create an MLFlow experiment."""
        if experiment_name is None:
            experiment_name = "Default"
        try:
            experiment_id = self._mlflow_client.create_experiment(
                experiment_name, artifact_location=artifact_location
            )
            experiment = self._mlflow_client.get_experiment(experiment_id)
        except:
            experiment = self._mlflow_client.get_experiment_by_name(experiment_name)
        return experiment

    @property
    def experiment_id(self) -> str:
        return self._mlflow_experiment.experiment_id

    def start_run(self, run_name: Optional[str] = None):
        """Start a new run."""
        mlflow_run = self._mlflow_client.create_run(
            self._mlflow_experiment.experiment_id, run_name=run_name
        )
        return Run(self, mlflow_run)

    def search_runs(
        self,
        filter_string: str,
        run_view_type: int = ViewType.ACTIVE_ONLY,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: Union[List[str], None] = None,
    ) -> Generator[Run, None, None]:
        result = self._mlflow_client.search_runs(
            experiment_ids=[self._mlflow_experiment.experiment_id],
            filter_string=filter_string,
            run_view_type=run_view_type,
            max_results=max_results,
            order_by=order_by,
        )
        total_results = 0
        while result.token is not None:
            for mlflow_run in result:
                yield Run(self, mlflow_run)
                total_results += 1
            if total_results >= max_results:
                break
            result = self._mlflow_client.search_runs(
                experiment_ids=[self._mlflow_experiment.experiment_id],
                filter_string=filter_string,
                run_view_type=run_view_type,
                max_results=max_results,
                order_by=order_by,
                page_token=result.token,
            )
