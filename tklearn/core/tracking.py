from __future__ import annotations
from typing import Any, Optional, Union
import mlflow
from mlflow.entities import (
    Experiment as MLflowExperiment,
    Run as MLflowRun,
)


class Run:
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
        value: float,
        *,
        timestamp: Optional[int] = None,
        step: Optional[int] = None,
    ) -> None:
        """Log a metric to MLFlow."""
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

    def log_param(self, key: str, value: str) -> None:
        """Log a parameter to MLFlow."""
        self._mlflow_client.log_param(self._mlflow_run.info.run_id, key, value)

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

    def terminate(self) -> None:
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
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
        experiment_name: Union[str, None] = None,
    ) -> None:
        self._mlflow_client = mlflow.tracking.MlflowClient(
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
        )
        self._mlflow_client._get_registry_client()
        self._mlflow_experiment: MLflowExperiment = (
            self._mlflow_get_or_create_experiment(
                experiment_name,
                artifact_location=artifact_location,
            )
        )

    def _mlflow_get_or_create_experiment(
        self,
        experiment_name: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ) -> MLflowExperiment:
        """Get or create an MLFlow experiment."""
        if experiment_name is None:
            experiment_name = "Default"
        experiment = self._mlflow_client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = self._mlflow_client.create_experiment(
                experiment_name, artifact_location=artifact_location
            )
            experiment = self._mlflow_client.get_experiment(experiment_id)
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
