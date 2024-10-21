import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from rdflib import RDF, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DefinedNamespace


class CN(DefinedNamespace):
    _warn = False
    _fail = False

    Concept: URIRef
    label: URIRef
    language: URIRef
    sense_label: URIRef
    term: URIRef
    site: URIRef

    Relation: URIRef
    symmetric: URIRef

    Assertion: URIRef
    relation: URIRef
    start: URIRef
    end: URIRef
    weight: URIRef
    dataset: URIRef
    sources: URIRef
    surfaceText: URIRef
    license: URIRef

    _NS = Namespace("http://conceptnet.io/")


@dataclass
class Node:
    id: str
    label: Optional[str] = None
    language: Optional[str] = None
    sense_label: Optional[str] = None
    term: Optional["Node"] = None
    site: Optional[str] = None


@dataclass
class Relation:
    id: str
    symmetric: bool = False


@dataclass
class Edge:
    id: str
    rel: Relation
    start: Node
    end: Node
    weight: float
    dataset: Optional[str] = None
    sources: List["Source"] = None
    surfaceText: Optional[str] = None
    license: Optional[str] = None


@dataclass
class Feature:
    rel: Optional[Relation] = None
    start: Optional[Node] = None
    end: Optional[Node] = None
    node: Optional[Node] = None


@dataclass
class Source:
    id: str
    contributor: Optional[str] = None
    process: Optional[str] = None
    activity: Optional[str] = None


@dataclass
class RelatedNode:
    id: str
    weight: float


@dataclass
class PartialCollectionView:
    paginatedProperty: str
    firstPage: str
    nextPage: Optional[str] = None
    previousPage: Optional[str] = None


@dataclass
class Query:
    id: str
    edges: Optional[List[Edge]] = None
    features: Optional[List[Feature]] = None
    related: Optional[List[RelatedNode]] = None
    view: Optional[PartialCollectionView] = None
    value: Optional[float] = None
    license: Optional[str] = None


class KnowledgeGraph:
    def __init__(self, identifier: str):
        if not identifier.isidentifier():
            raise ValueError("identifier must be a valid Python identifier")
        self.identifier = URIRef(identifier)
        self.db_file = f"{identifier}.sqlite"
        self.uri = Literal(f"sqlite:///{self.db_file}")
        self.graph = Graph("SQLAlchemy", identifier=self.identifier)

    def open(self):
        self.graph.open(self.uri, create=True)

    def close(self):
        self.graph.close()

    def safe_cleanup(self):
        self.close()
        if os.path.exists(self.db_file):
            os.remove(self.db_file)
            print(f"Removed database file: {self.db_file}")
        else:
            print(f"Database file not found: {self.db_file}")

    def add_node(self, node: Node):
        node_uri = URIRef(CN[node.id])
        self.graph.add((node_uri, RDF.type, CN.Concept))
        self.graph.add((node_uri, CN.label, Literal(node.label)))
        self.graph.add((node_uri, CN.language, Literal(node.language)))
        if node.sense_label:
            self.graph.add((
                node_uri,
                CN.sense_label,
                Literal(node.sense_label),
            ))
        if node.site:
            self.graph.add((node_uri, CN.site, Literal(node.site)))

    def add_edge(self, edge: Edge):
        start_node = URIRef(CN[edge.start.id])
        end_node = URIRef(CN[edge.end.id])
        relation = URIRef(CN[edge.rel.id])
        edge_uri = URIRef(CN[edge.id])

        # add relation
        self.graph.add((relation, RDF.type, CN.Relation))

        # add edge
        self.graph.add((edge_uri, RDF.type, CN.Assertion))
        self.graph.add((edge_uri, CN.relation, relation))
        self.graph.add((edge_uri, CN.start, start_node))
        self.graph.add((edge_uri, CN.end, end_node))
        self.graph.add((edge_uri, CN.weight, Literal(edge.weight)))

        if edge.dataset:
            self.graph.add((
                edge_uri,
                CN.dataset,
                Literal(edge.dataset),
            ))
        if edge.surfaceText:
            self.graph.add((
                edge_uri,
                CN.surfaceText,
                Literal(edge.surfaceText),
            ))
        if edge.license:
            self.graph.add((
                edge_uri,
                CN.license,
                Literal(edge.license),
            ))

        if edge.sources:
            for source in edge.sources:
                source_uri = URIRef(CN[source.id])
                self.graph.add((edge_uri, CN["source"], source_uri))
                self.graph.add((
                    source_uri,
                    CN["contributor"],
                    Literal(source.contributor),
                ))
                if source.process:
                    self.graph.add((
                        source_uri,
                        CN["process"],
                        Literal(source.process),
                    ))
                if source.activity:
                    self.graph.add((
                        source_uri,
                        CN["activity"],
                        Literal(source.activity),
                    ))

    def store_edges(self, edges: List[Edge]):
        for edge in edges:
            self.add_node(edge.start)
            self.add_node(edge.end)
            self.add_edge(edge)

        self.graph.commit()
        print(
            f"Stored {len(edges)} edges. Total triples in graph: {len(self.graph)}"
        )

    def query(
        self, start_label: str, relation: str = None, end_label: str = None
    ) -> List[Tuple[str, str, str, float]]:
        query = """
        PREFIX cn: <http://conceptnet.io/>
        SELECT ?start ?rel ?end ?weight
        WHERE {
            ?edge cn:start ?start ;
                  cn:end ?end ;
                  cn:relation ?rel ;
                  cn:weight ?weight .
            ?start cn:label ?start_label .
            ?end cn:label ?end_label .
            FILTER(?start_label = ?param_start_label)
            FILTER(?rel = ?param_relation)
            OPTIONAL {
                FILTER(?end_label = ?param_end_label)
            }
        }
        """

        bindings = {"param_start_label": Literal(start_label)}

        if relation:
            bindings["param_relation"] = URIRef(relation)

        if end_label:
            bindings["param_end_label"] = Literal(end_label)

        results = self.graph.query(query, initBindings=bindings)

        return [
            (str(row.start), str(row.rel), str(row.end), float(row.weight))
            for row in results
        ]

    def debug_print_all_triples(self):
        print("All triples in the graph:")
        for s, p, o in self.graph:
            print(f"Subject: {s}, Predicate: {p}, Object: {o}")


# Example usage
if __name__ == "__main__":
    # Assuming Edge, Relation, and Node classes are defined elsewhere
    edges = [
        Edge(
            id="e1",
            rel=Relation(id="IsA"),
            start=Node(id="n1", label="cat", language="en"),
            end=Node(id="n2", label="animal", language="en"),
            weight=1.0,
        ),
        Edge(
            id="e2",
            rel=Relation(id="HasProperty"),
            start=Node(id="n1", label="cat", language="en"),
            end=Node(id="n3", label="furry", language="en"),
            weight=0.9,
        ),
    ]

    cn_graph = KnowledgeGraph("example")
    cn_graph.open()
    cn_graph.store_edges(edges)

    # Debug: Print all triples
    cn_graph.debug_print_all_triples()

    # Query the graph
    results = cn_graph.query("cat", "Not", "animal")
    print("Query results for 'cat IsA animal':", results)

    # Query all relations for 'cat'
    all_cat_relations = cn_graph.query("cat")
    print("All relations for 'cat':")
    for rel in all_cat_relations:
        print(rel)

    cn_graph.close()
    # Uncomment the next line if you want to remove the database
    cn_graph.safe_cleanup()
