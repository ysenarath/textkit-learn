import unittest

from tklearn.utils import Registry


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = Registry("widget")

        @self.registry.register("box")
        class Box:
            def __init__(self, size=1):
                self.size = size

        self.Box = Box

    def test_create(self):
        self.assertEqual(self.registry.create("box", size=3).size, 3)
        self.assertIs(self.registry.get("box"), self.Box)

    def test_from_config(self):
        box = self.registry.from_config({"type": "box", "size": 2})
        self.assertEqual(box.size, 2)
        with self.assertRaisesRegex(KeyError, "missing the 'type' key"):
            self.registry.from_config({"size": 2})

    def test_unknown_name_lists_available(self):
        with self.assertRaisesRegex(KeyError, "available: box"):
            self.registry.create("ball")

    def test_duplicate_name(self):
        self.registry.register("box", self.Box)  # same object is fine
        with self.assertRaises(ValueError):
            self.registry.register("box", object)

    def test_container_protocol(self):
        self.assertIn("box", self.registry)
        self.assertEqual(list(self.registry), ["box"])
        self.assertEqual(len(self.registry), 1)


if __name__ == "__main__":
    unittest.main()
