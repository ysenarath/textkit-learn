from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable

import orjson
from tqdm import auto as tqdm

from tklearn import config
from tklearn.kb.conceptnet import csv
from tklearn.kb.loader import KnowledgeLoader, KnowledgeLoaderConfig
from tklearn.logging import get_logger
from tklearn.utils import download

__all__ = [
    "ConceptNetLoader",
]

logger = get_logger(__name__)


class ConceptNetLoaderConfig(KnowledgeLoaderConfig):
    identifier: ClassVar[str] = "conceptnet"
    force: bool = False
    namespace: str = "http://conceptnet.io/"


class ConceptNetLoader(KnowledgeLoader):
    config: ConceptNetLoaderConfig

    def __post_init__(self) -> None:
        path = (
            config.cache_dir
            / "kb"
            / "conceptnet"
            / "conceptnet-assertions-5.7.0.csv.gz"
        )
        if not path.name.endswith(".csv.gz"):
            raise ValueError("path must be a '.csv.gz' file")
        self.path = Path(path)
        self.csv_path = self.path.with_suffix("")
        self.jsonl_path = self.csv_path.with_suffix(".jsonl")
        self.jsonld_path = self.csv_path.with_suffix(".jsonld")
        self.verbose = self.config.verbose
        self.force = self.config.force
        self.exist_ok = True
        self.unzip = True
        super().__post_init__()

    def download(self) -> None:
        """Download resource."""
        url = config.external.conceptnet.download_url
        download(
            url,
            self.path,
            verbose=self.verbose,
            force=self.force,
            exist_ok=self.exist_ok,
            unzip=self.unzip,
        )
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"'{self.csv_path}' does not exist, make sure to unzip the downloaded file"
            )
        with open(self.csv_path, "rb") as fp:
            n_rows_csv = sum(1 for _ in fp)
        # clean up the jsonl file if it exists or the row count differs
        jsonl_path = self.jsonl_path
        if jsonl_path.exists():
            with open(jsonl_path, "rb") as fp:
                n_rows_jsonl = sum(1 for _ in fp)
            if self.force or n_rows_csv != n_rows_jsonl:
                if not self.force:
                    logger.warning(
                        "Overwriting '%s' because the row count differs (CSV: %d, JSONL: %d)",
                        jsonl_path,
                        n_rows_csv,
                        n_rows_jsonl,
                    )
                jsonl_path.unlink()
            elif self.exist_ok:
                return
            else:
                raise FileExistsError(f"'{jsonl_path}' already exists")
        progress_bar = None
        if self.verbose:
            progress_bar = tqdm.tqdm(
                total=n_rows_csv,
                desc="Parsing the CSV file and writing to JSONL",
            )
        with open(jsonl_path, "w", encoding="utf-8") as fw:
            with open(self.csv_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    edge = csv.read_line(line)
                    fw.write(
                        json.dumps(edge, ensure_ascii=False).replace(
                            "\\u0000", ""
                        )
                        + "\n"
                    )
                    if progress_bar:
                        progress_bar.update(1)

    def iterrows(self) -> Iterable[Dict[str, Any]]:
        """Iterate over edges."""
        with open(self.jsonl_path, "rb") as f:
            n_rows = sum(1 for _ in f)
        progress_bar = None
        if self.verbose:
            progress_bar = tqdm.tqdm(total=n_rows, desc="Loading edges")
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                yield orjson.loads(line)
                if progress_bar:
                    progress_bar.update(1)
