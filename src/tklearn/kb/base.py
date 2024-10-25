from __future__ import annotations

import os
from typing import Any, ClassVar, Dict, Iterable, List, Tuple

from nightjar import AutoModule, BaseConfig, BaseModule
from rdflib import Graph, URIRef, plugin
from rdflib.store import Store
from tqdm import auto as tqdm

from tklearn.config import config

__all__ = [
    "AutoKnowledgeGraph",
    "KnowledgeGraph",
    "KnowledgeGraphConfig",
]


class KnowledgeGraphConfig(BaseConfig, dispatch="identifier"):
    identifier: ClassVar[str]


class AutoKnowledgeGraph(AutoModule):
    def __new__(cls, config: KnowledgeGraphConfig) -> KnowledgeGraph:
        return super().__new__(cls, config)


class KnowledgeGraph(BaseModule):
    config: KnowledgeGraphConfig

    def __post_init__(self) -> None:
        super().__post_init__()
        identifier = self.config.identifier
        if not identifier.isidentifier():
            raise ValueError("identifier must be a valid Python identifier")
        self.identifier = URIRef(identifier)
        db_path = config.cache_dir / "graphs" / f"{identifier}.leveldb"
        self.db_path = db_path.absolute().expanduser()
        self.uri = str(db_path)
        store = plugin.get("LevelDB", Store)(identifier=self.identifier)
        self.graph = Graph(store)
        # self.uri = Literal(f"sqlite:///{self.db_path}")
        # self.graph = Graph("LevelDB", identifier=self.identifier)
        self.download()
        num_edges = 0
        with self:
            num_edges = len(self.graph)
        if num_edges != 0:
            self.cleanup()

    def __enter__(self) -> KnowledgeGraph:
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    def open(self):
        try:
            self.graph.open(self.uri, create=True)
        except Exception:
            self.graph.open(self.uri, create=False)

    def close(self):
        try:
            self.graph.close()
        except Exception:
            pass

    def download(self) -> None:
        """Download resource."""
        raise NotImplementedError

    def values(self) -> Iterable[Dict[str, Any]]:
        """Iterate over edges."""
        raise NotImplementedError

    def cleanup(self):
        self.close()
        if not os.path.exists(self.db_path):
            return
        # self.graph.destroy(self.uri)
        os.remove(self.db_path)

    # def _extend_by_triples(self, triples: List[Tuple]):
    #     self.graph.addN((s, p, o, self.graph) for s, p, o in triples)
    #     self.graph.commit()

    # def batch_extend(
    #     self, items: Iterable[Any], batch_size: int = 50_000
    # ) -> None:
    #     self.open()
    #     triples = []
    #     for item in tqdm.tqdm(items, desc="Extending graph"):
    #         if not hasattr(item, "to_triples"):
    #             raise ValueError(
    #                 f"can only extend with items that have a to_triples method, got '{type(item).__name__}'"
    #             )
    #         triples += item.to_triples()
    #         if len(triples) > batch_size:
    #             self._extend_by_triples(triples)
    #             triples = []
    #     if triples:
    #         self._extend_by_triples(triples)
    #     self.close()

    # def extend(self, items: List[Any]) -> None:
    #     self.open()
    #     for item in tqdm.tqdm(items, desc="Extending graph"):
    #         if not hasattr(item, "to_triples"):
    #             raise ValueError(
    #                 f"can only extend with items that have a to_triples method, got '{type(item).__name__}'"
    #             )
    #         for triple in item.to_triples():
    #             self.graph.add(triple)
    #     self.graph.commit()
    #     self.close()

    # def query(
    #     self, start_label: str, relation: str = None, end_label: str = None
    # ) -> List[Tuple[str, str, str, float]]:
    #     query = """
    #     PREFIX cn: <http://conceptnet.io/>
    #     SELECT ?start ?rel ?end ?weight
    #     WHERE {
    #         ?edge cn:start ?start ;
    #               cn:end ?end ;
    #               cn:relation ?rel ;
    #               cn:weight ?weight .
    #         ?start cn:label ?start_label .
    #         ?end cn:label ?end_label .
    #         FILTER(?start_label = ?param_start_label)
    #         FILTER(?rel = ?param_relation)
    #         OPTIONAL {
    #             FILTER(?end_label = ?param_end_label)
    #         }
    #     }
    #     """

    #     bindings = {"param_start_label": Literal(start_label)}

    #     if relation:
    #         bindings["param_relation"] = URIRef(relation)

    #     if end_label:
    #         bindings["param_end_label"] = Literal(end_label)

    #     results = self.graph.query(query, initBindings=bindings)

    #     return [
    #         (str(row.start), str(row.rel), str(row.end), float(row.weight))
    #         for row in results
    #     ]
