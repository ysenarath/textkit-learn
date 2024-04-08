import json
from pathlib import Path
from typing import Generator, Union

import orjson
from tqdm import auto as tqdm

from tklearn import config
from tklearn.base.resource import ResourceIO
from tklearn.kb.conceptnet.uri import conjunction_uri, to_json_ld
from tklearn.utils import download

__all__ = [
    "ConceptNetIO",
]


def parse_edge(line: str) -> dict:
    parts = line.strip().split("\t")
    edge_id, relation_uri, subject_uri, object_uri, metadata = parts
    metadata = orjson.loads(metadata)
    edge = {
        "@id": edge_id,
        "rel": to_json_ld(relation_uri),
        "start": to_json_ld(subject_uri),
        "end": to_json_ld(object_uri),
    }
    if "surfaceText" in metadata:
        edge["surfaceText"] = metadata["surfaceText"]
    if "sources" in metadata:
        sources = []
        source: dict
        for source in metadata["sources"]:
            source["@id"] = conjunction_uri(*source.values())
            sources.append(source)
        edge["sources"] = sources
    if "license" in metadata:
        edge["license"] = metadata["license"]
    if "weight" in metadata:
        edge["weight"] = metadata["weight"]
    if "dataset" in metadata:
        edge["dataset"] = metadata["dataset"]
    return edge


class ConceptNetIO(ResourceIO):
    def __init__(self, path: Union[str, Path] = None):
        if path is None:
            path = (
                config.cache_dir
                / "resources"
                / "conceptnet"
                / "conceptnet-assertions-5.7.0.csv.gz"
            )
        if not str(path).endswith(".csv.gz"):
            raise ValueError("path must be a '.csv.gz' file")
        self.path = Path(path)
        self.csv_path = self.path.with_suffix("")
        self.jsonl_path = self.csv_path.with_suffix(".jsonl")
        self.jsonld_path = self.csv_path.with_suffix(".jsonld")

    def read_csv_iter(self, verbose: bool = False) -> Generator[dict, None, None]:
        with open(self.csv_path, "rb") as f:
            n_rows = sum(1 for _ in f)
        progress_bar = None
        if verbose:
            progress_bar = tqdm.tqdm(total=n_rows, desc="Reading Edges")
        self.edges = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            for line in f:
                yield parse_edge(line)
                if progress_bar:
                    progress_bar.update(1)

    def read_jsonl_iter(self, verbose: bool = False) -> Generator[dict, None, None]:
        with open(self.jsonl_path, "rb") as f:
            n_rows = sum(1 for _ in f)
        progress_bar = None
        if verbose:
            progress_bar = tqdm.tqdm(total=n_rows, desc="Loading Edges")
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                yield orjson.loads(line)
                if progress_bar:
                    progress_bar.update(1)

    def to_jsonl(
        self,
        path: Union[str, Path] = None,
        verbose: bool = False,
        exist_ok: bool = False,
    ) -> None:
        if path is None:
            path = self.jsonl_path
        if path.exists():
            if exist_ok:
                return
            raise FileExistsError(f"'{path}' already exists")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"'{self.csv_path}' does not exist")
        with open(path, "w", encoding="utf-8") as f:
            edges = self.edges
            if edges is None:
                with open(self.csv_path, "rb") as fp:
                    n_rows = sum(1 for _ in fp)
                edges = self.read_csv_iter(verbose=False)
                progress_bar = tqdm.tqdm(total=n_rows, desc="Reading and Writing Edges")
            elif verbose:
                progress_bar = tqdm.tqdm(total=len(self.edges), desc="Writing Edges")
            else:
                progress_bar = None
            for edge in edges:
                f.write(
                    json.dumps(edge, ensure_ascii=False).replace("\\u0000", "") + "\n"
                )
                if progress_bar:
                    progress_bar.update(1)

    def to_jsonld(self, path: Union[str, Path] = None, verbose: bool = False) -> dict:
        if path is None:
            path = self.jsonld_path
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "@context": [
                            "http://api.conceptnet.io/ld/conceptnet5.7/context.ld.json"
                        ],
                        "edges": self.edges,
                    },
                    ensure_ascii=False,
                ).replace("\\u0000", "")
            )

    def download(
        self,
        url: str = None,
        verbose: bool = False,
        force: bool = False,
        exist_ok: bool = False,
        unzip: bool = True,
    ) -> None:
        if url is None:
            url = config.external.conceptnet.download_url
        download(
            url,
            self.path,
            verbose=verbose,
            force=force,
            exist_ok=exist_ok,
            unzip=unzip,
        )

    def load(self, verbose: bool = False) -> Generator[dict, None, None]:
        existing = set()
        for edge in self.read_jsonl_iter(verbose=verbose):
            for term_field in ["start", "end"]:
                node_id = edge[term_field]["@id"]
                if node_id not in existing:
                    existing.add(node_id)
                    yield edge[term_field]
