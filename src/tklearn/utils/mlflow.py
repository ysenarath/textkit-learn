from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, TypeVar

import yaml
from mlflow.entities import RunStatus
from tqdm import auto as tqdm

T = TypeVar("T")


@dataclass
class Value:
    value: float
    timestamp: Optional[float] = None
    step: Optional[int] = None


@dataclass
class Run:
    id: str
    info: dict[str, Any] = field(default_factory=dict)
    values: dict[str, Value] = field(default_factory=dict)
    children: dict[str, Run] = field(default_factory=dict)


class MLFlowRunLoader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.meta_path = self.path / "meta.yaml"
        self.params_path = self.path / "params"
        self.metrics_path = self.path / "metrics"
        self.tags_path = self.path / "tags"
        self.artifacts_path = self.path / "artifacts"

    def _read_yaml(self, path: Path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _read_value(self, path: Path, type=None) -> Value:
        with open(path, "r") as f:
            values = []
            all_lines = f.read().strip()
            if type == "metric":
                # if it's a metric, it is either a float or tuple
                for i, line in enumerate(all_lines.split("\n")):
                    # line must have 3 parts or it is a float
                    if " " in line:
                        try:
                            timestamp, value, step = line.split(" ")
                        except Exception:
                            raise ValueError(
                                f"error in '{path}' at line {i}: {line}"
                            )
                        values += [
                            Value(float(value), float(timestamp), int(step))
                        ]
                    else:
                        return Value(float(line))
            else:
                line = all_lines.strip()
                if path.name.endswith("mlflow.loggedArtifacts"):
                    return Value(json.loads(line))
                try:
                    # do opposite of str(...) to get the original value
                    return Value(ast.literal_eval(line))
                except Exception:
                    pass
                try:
                    # try to load as json
                    # - (everything except a string should be json compatible)
                    return Value(json.loads(line))
                except json.JSONDecodeError:
                    return Value(line)

    def load(self, pbar: tqdm.tqdm):
        info = self._read_yaml(self.meta_path)
        values = {}
        for param in self.params_path.iterdir():
            # read each file and append to the list
            pbar.set_postfix_str(param.name)
            values[f"param.{param.name}"] = self._read_value(param)
        for metric in self.metrics_path.iterdir():
            # read each file and append to the list
            pbar.set_postfix_str(metric.name)
            values[f"metric.{metric.name}"] = self._read_value(
                metric, type="metric"
            )
        for tag in self.tags_path.iterdir():
            pbar.set_postfix_str(tag.name)
            # read each file and append to the list
            values[f"tag.{tag.name}"] = self._read_value(tag)
        return Run(
            id=self.path.name,
            info=info,
            values=values,
        )


class MLFlowClient:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _is_run(self, run: Path):
        return run.is_dir() and (run / "meta.yaml").exists()

    def load(self):
        total = 0
        run_paths = []
        for possible_run in self.path.iterdir():
            if self._is_run(possible_run):
                total += 1
                run_paths.append(possible_run)
        runs: Dict[str, Run] = {}
        pbar: Iterable[Path] = tqdm.tqdm(run_paths)
        for run_path in pbar:
            pbar.set_description_str(f"Loading {run_path.name}")
            r = MLFlowRunLoader(run_path).load(pbar)
            runs[r.id] = r
        root = {}
        # add children
        for run in runs.values():
            if RunStatus.to_string(run.info["status"]) in ["FINISHED"]:
                continue
            if "tag.mlflow.parentRunId" not in run.values:
                root[run.id] = run
                continue
            parent_run_id = run.values["tag.mlflow.parentRunId"].value
            if parent_run_id in runs:
                runs[parent_run_id].children[run.id] = run
        return root
