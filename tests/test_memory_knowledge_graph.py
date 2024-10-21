import time
import unittest

from src.tklearn.core.knowledge_graph import MemoryKnowledgeGraph


class TestMemoryKnowledgeGraph(unittest.TestCase):
    def setUp(self):
        self.kg = MemoryKnowledgeGraph()

    def test_add_and_get_node(self):
        self.kg.add_node("person1", {"name": "Alice", "age": 30})
        node = self.kg.get_node("person1")
        self.assertEqual(node, {"name": "Alice", "age": 30})

    def test_add_and_query_edge(self):
        self.kg.add_node("person1", {"name": "Alice"})
        self.kg.add_node("city1", {"name": "New York"})
        self.kg.add_edge("person1", "city1", "lives_in")
        result = self.kg.query("person1", "lives_in")
        self.assertEqual(result, ["city1"])

    def test_get_relations(self):
        self.kg.add_node("person1", {"name": "Alice"})
        self.kg.add_node("person2", {"name": "Bob"})
        self.kg.add_node("city1", {"name": "New York"})
        self.kg.add_edge("person1", "city1", "lives_in")
        self.kg.add_edge("person1", "person2", "knows")
        relations = self.kg.get_relations("person1")
        self.assertEqual(
            relations, {"lives_in": ["city1"], "knows": ["person2"]}
        )

    def test_large_graph_performance(self):
        start_time = time.time()
        for i in range(100000):
            self.kg.add_node(str(i), {"value": i})
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

    def test_str_representation(self):
        self.kg.add_node("person1", {"name": "Alice"})
        self.kg.add_node("person2", {"name": "Bob"})
        self.kg.add_edge("person1", "person2", "knows")
        self.assertIn("2 nodes", str(self.kg))
        self.assertIn("1 edges", str(self.kg))

    def test_len(self):
        self.kg.add_node("person1", {"name": "Alice"})
        self.kg.add_node("person2", {"name": "Bob"})
        self.assertEqual(len(self.kg), 2)


if __name__ == "__main__":
    unittest.main()
