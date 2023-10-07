import shutil
import unittest

import pandas as pd
import pyarrow.compute as pc

from tklearn.datasets import Dataset
from tklearn.config import config

MOCK_DATA = [
    {"name": "John", "age": 27},
    {"name": "Mary", "age": 20},
    {"name": "Peter", "age": 25},
    {"name": "Kevin", "age": 50},
    {"name": "Paul", "age": 60},
    {"name": "Kenny", "age": 18},
    {"name": "Micheal", "age": 20},
    {"name": "Tom", "age": 25},
]
INIT_SIZE = 5


def map_arrow_func(x):
    return {
        "name": x["name"],
        "age": x["age"] + 1,
    }


def test_map_arrow_batched_pandas_func(x):
    x["age"] += 1
    return x


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset()
        self.dataset.extend(MOCK_DATA[:INIT_SIZE])

    def test_extend(self):
        # Test that extend method returns a Dataset object
        self.dataset.extend(MOCK_DATA[INIT_SIZE:])
        self.assertEqual(len(self.dataset), len(MOCK_DATA))

    def test_map_arrow(self):
        # Test that map method returns a Dataset object
        result = self.dataset.map(map_arrow_func, mode="arrow", batched=False)
        self.assertIsInstance(result, Dataset)
        for x, y in zip(result, self.dataset):
            self.assertEqual(x["age"], y["age"] + 1)

    def test_map_arrow_batched_pandas(self):
        # Test that map method returns a Dataset object
        result = self.dataset.map(
            test_map_arrow_batched_pandas_func,
            mode="arrow",
            batch_into=pd.DataFrame,
        )
        self.assertIsInstance(result, Dataset)
        for x, y in zip(result, self.dataset):
            self.assertEqual(x["age"], y["age"] + 1)

    def test_filter(self):
        # Test that filter method returns a generator object
        expr = pc.field("age") >= 50
        result = self.dataset.filter(expr)
        self.assertIsInstance(result, Dataset)
        self.assertEqual(len(result), 2)

    def test_filter_by(self):
        # Test that filter_by method returns a Dataset object
        result = self.dataset.filter_by(name="John").first()
        self.assertEqual(result["name"], "John")
        self.assertEqual(result["age"], 27)

    def tearDown(self) -> None:
        return shutil.rmtree(config.resource_dir)


if __name__ == "__main__":
    unittest.main()
