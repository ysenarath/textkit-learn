import mmap
import os
import struct
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional


class BaseKnowledgeGraph(ABC):
    @abstractmethod
    def add_node(
        self, node_id: str, attributes: Optional[Dict[str, Any]] = None
    ):
        """Add a node to the graph."""
        pass

    @abstractmethod
    def add_edge(self, source: str, target: str, relation: str):
        """Add an edge (relation) between two nodes."""
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node's attributes."""
        pass

    @abstractmethod
    def get_relations(self, node_id: str) -> Dict[str, List[str]]:
        """Get all relations for a given node."""
        pass

    @abstractmethod
    def query(self, source: str, relation: str) -> List[str]:
        """Query the graph for nodes related to the source node by the given relation."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation of the graph."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        pass


class MemoryKnowledgeGraph(BaseKnowledgeGraph):
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, Dict[str, List[str]]] = {}

    def add_node(
        self, node_id: str, attributes: Optional[Dict[str, Any]] = None
    ):
        """Add a node to the graph."""
        if node_id not in self.nodes:
            self.nodes[node_id] = attributes or {}
            self.edges[node_id] = {}

    def add_edge(self, source: str, target: str, relation: str):
        """Add an edge (relation) between two nodes."""
        self.add_node(source)
        self.add_node(target)
        if relation not in self.edges[source]:
            self.edges[source][relation] = []
        self.edges[source][relation].append(target)

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node's attributes."""
        return self.nodes.get(node_id)

    def get_relations(self, node_id: str) -> Dict[str, List[str]]:
        """Get all relations for a given node."""
        return self.edges.get(node_id, {})

    def query(self, source: str, relation: str) -> List[str]:
        """Query the graph for nodes related to the source node by the given relation."""
        return self.edges.get(source, {}).get(relation, [])

    def __str__(self):
        return f"MemoryKnowledgeGraph with {len(self.nodes)} nodes and {sum(len(relations) for relations in self.edges.values())} edges"

    def __len__(self):
        return len(self.nodes)


# Keeping the old name for backward compatibility
KnowledgeGraph = MemoryKnowledgeGraph


class MMapKnowledgeGraph(BaseKnowledgeGraph):
    def __init__(
        self, filename: str, max_nodes: int = 1000000, max_edges: int = 5000000
    ):
        self.filename = filename
        self.max_nodes = max_nodes
        self.max_edges = max_edges

        self.node_size = struct.calcsize("i64si")  # int, 64-byte string, int
        self.edge_size = struct.calcsize("ii32s")  # int, int, 32-byte string

        if not os.path.exists(filename):
            with open(filename, "wb") as f:
                f.write(
                    struct.pack("ii", 0, 0)
                )  # Initial node and edge counts
                f.write(
                    b"\0"
                    * (self.node_size * max_nodes + self.edge_size * max_edges)
                )

        self.file = open(filename, "r+b")
        self.mm = mmap.mmap(self.file.fileno(), 0)

        self.node_count, self.edge_count = struct.unpack("ii", self.mm[:8])
        self.nodes_offset = 8
        self.edges_offset = self.nodes_offset + self.node_size * max_nodes

        # In-memory index for faster querying
        self.edge_index = defaultdict(lambda: defaultdict(list))
        self._build_index()

    def _build_index(self):
        for i in range(self.edge_count):
            offset = self.edges_offset + i * self.edge_size
            src, tgt, rel = struct.unpack(
                "ii32s", self.mm[offset : offset + self.edge_size]
            )
            relation = rel.decode("utf-8").rstrip("\0")
            self.edge_index[src][relation].append(tgt)

    def add_node(
        self, node_id: str, attributes: Optional[Dict[str, Any]] = None
    ):
        if self.node_count >= self.max_nodes:
            raise ValueError("Maximum number of nodes reached")

        name = attributes.get("name", "") if attributes else ""
        if len(name.encode("utf-8")) > 64:
            raise ValueError(
                "Node name exceeds 64 bytes when encoded to UTF-8"
            )

        offset = self.nodes_offset + self.node_count * self.node_size
        self.mm[offset : offset + self.node_size] = struct.pack(
            "i64si", int(node_id), name.encode("utf-8"), 0
        )

        self.node_count += 1
        self.mm[:4] = struct.pack("i", self.node_count)

    def add_edge(self, source: str, target: str, relation: str):
        if self.edge_count >= self.max_edges:
            raise ValueError("Maximum number of edges reached")

        if len(relation.encode("utf-8")) > 32:
            raise ValueError(
                "Edge relation name exceeds 32 bytes when encoded to UTF-8"
            )

        offset = self.edges_offset + self.edge_count * self.edge_size
        self.mm[offset : offset + self.edge_size] = struct.pack(
            "ii32s", int(source), int(target), relation.encode("utf-8")
        )

        self.edge_count += 1
        self.mm[4:8] = struct.pack("i", self.edge_count)

        # Update the in-memory index
        self.edge_index[int(source)][relation].append(int(target))

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        for i in range(self.node_count):
            offset = self.nodes_offset + i * self.node_size
            nid, name, _ = struct.unpack(
                "i64si", self.mm[offset : offset + self.node_size]
            )
            if nid == int(node_id):
                return {
                    "id": str(nid),
                    "name": name.decode("utf-8").rstrip("\0"),
                }
        return None

    def get_relations(self, node_id: str) -> Dict[str, List[str]]:
        return {
            relation: [str(target) for target in targets]
            for relation, targets in self.edge_index[int(node_id)].items()
        }

    def query(self, source: str, relation: str) -> List[str]:
        return [
            str(target) for target in self.edge_index[int(source)][relation]
        ]

    def __str__(self):
        return f"MMapKnowledgeGraph with {self.node_count} nodes and {self.edge_count} edges"

    def __len__(self):
        return self.node_count

    def __del__(self):
        self.mm.close()
        self.file.close()


"""
CFLAGS="-I/opt/homebrew/Cellar/leveldb/1.23_1/include/ -L/opt/homebrew/Cellar/leveldb/1.23_1/lib/ -fno-rtti" pip install --force-reinstall --config-settings='--global-option="build_ext"' --no-cache-dir plyvel
CFLAGS="-I/opt/homebrew/include -w" LDFLAGS="-L/opt/homebrew/lib" pip install --force-reinstall plyvel
"""
