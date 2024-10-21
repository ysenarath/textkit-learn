import os
import time
import unittest

from src.tklearn.core.knowledge_graph import MMapKnowledgeGraph


class TestMMapKnowledgeGraph(unittest.TestCase):
    def setUp(self):
        self.test_file = "test_graph.bin"
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        self.kg = MMapKnowledgeGraph(
            self.test_file, max_nodes=1000000, max_edges=5000000
        )

    def tearDown(self):
        del self.kg
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_add_and_get_node(self):
        self.kg.add_node("1", {"name": "Alice"})
        node = self.kg.get_node("1")
        self.assertEqual(node, {"id": "1", "name": "Alice"})

    def test_add_and_query_edge(self):
        self.kg.add_node("1", {"name": "Alice"})
        self.kg.add_node("2", {"name": "Bob"})
        self.kg.add_edge("1", "2", "knows")
        result = self.kg.query("1", "knows")
        self.assertEqual(result, ["2"])

    def test_get_relations(self):
        self.kg.add_node("1", {"name": "Alice"})
        self.kg.add_node("2", {"name": "Bob"})
        self.kg.add_node("3", {"name": "New York"})
        self.kg.add_edge("1", "3", "lives_in")
        self.kg.add_edge("1", "2", "knows")
        relations = self.kg.get_relations("1")
        self.assertEqual(relations, {"lives_in": ["3"], "knows": ["2"]})

    def test_large_graph_performance(self):
        start_time = time.time()
        for i in range(100000):
            self.kg.add_node(str(i), {"name": f"Node{i}"})
        for i in range(100000):
            self.kg.add_edge(str(i), str((i + 1) % 100000), "connected_to")
            self.kg.add_edge(str(i), str((i + 2) % 100000), "knows")
        add_time = time.time() - start_time

        start_time = time.time()
        for i in range(1000):
            node_id = str(i * 100)
            self.kg.query(node_id, "connected_to")
            self.kg.query(node_id, "knows")
        query_time = time.time() - start_time

        self.assertLess(
            add_time,
            2.0,
            "Adding 100,000 nodes and 200,000 edges took too long",
        )
        self.assertLess(
            query_time, 0.1, "Performing 2,000 queries took too long"
        )

    def test_node_name_too_long(self):
        with self.assertRaises(ValueError):
            self.kg.add_node("1", {"name": "A" * 65})

    def test_edge_relation_too_long(self):
        self.kg.add_node("1", {"name": "Alice"})
        self.kg.add_node("2", {"name": "Bob"})
        with self.assertRaises(ValueError):
            self.kg.add_edge("1", "2", "A" * 33)

    def test_max_nodes_reached(self):
        kg = MMapKnowledgeGraph("small_graph.bin", max_nodes=2, max_edges=5)
        kg.add_node("1", {"name": "Alice"})
        kg.add_node("2", {"name": "Bob"})
        with self.assertRaises(ValueError):
            kg.add_node("3", {"name": "Charlie"})
        os.remove("small_graph.bin")

    def test_max_edges_reached(self):
        kg = MMapKnowledgeGraph("small_graph.bin", max_nodes=5, max_edges=2)
        kg.add_node("1", {"name": "Alice"})
        kg.add_node("2", {"name": "Bob"})
        kg.add_edge("1", "2", "knows")
        kg.add_edge("2", "1", "knows")
        with self.assertRaises(ValueError):
            kg.add_edge("1", "2", "likes")
        os.remove("small_graph.bin")

    def test_str_representation(self):
        self.kg.add_node("1", {"name": "Alice"})
        self.kg.add_node("2", {"name": "Bob"})
        self.kg.add_edge("1", "2", "knows")
        self.assertIn("2 nodes", str(self.kg))
        self.assertIn("1 edges", str(self.kg))

    def test_len(self):
        self.kg.add_node("1", {"name": "Alice"})
        self.kg.add_node("2", {"name": "Bob"})
        self.assertEqual(len(self.kg), 2)


if __name__ == "__main__":
    unittest.main()
