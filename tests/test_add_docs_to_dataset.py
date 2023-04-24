import unittest

import numpy as np

from tklearn.datasets import Dataset


class TestAddDocumentsToDataset(unittest.TestCase):
    def test_add_docs_to_dataset(self):
        docs = [
            {
                'id': 0,
                'text': 'this is the document text',
                'embedding': [1, 2, 3],
                'tokens': {'token_ids': [1, 2, 3], 'mask': [[1, 2, 6], [3, 4]]},

            },
            {
                'id': 1,
                'text': 'this is the document 1 text',
                'embedding': [4, 5, 6],
                'tokens': {'token_ids': [1, 2, 3, 4, 6], 'mask': [[1, 2], [4], [3, 4]]},
            },
            {
                'id': 2,
                'text': 'this is the document 2 text',
                'embedding': [7, 8, 9],
                'tokens': {'token_ids': [1, 2, 3, 4, 6], 'mask': [[1, ]]},
            },
        ]
        dataset = Dataset()
        for i, doc in enumerate(docs):
            dataset.append(doc)
        self.assertEqual(len(dataset), len(docs))  # add assertion here

    def test_add_docs_with_numpy_arrays_to_dataset(self):
        docs = [
            {
                'id': 0,
                'text': 'this is the document text',
                'embedding': np.array([1, 2, 3]),
                'tokens': {'token_ids': [1, 2, 3], 'mask': [[1, 2, 6], [3, 4]]},

            },
            {
                'id': 1,
                'text': 'this is the document 1 text',
                'embedding': np.array([4, 5, 6]),
                'tokens': {'token_ids': [1, 2, 3, 4, 6], 'mask': [[1, 2], [4], [3, 4]]},
            },
            {
                'id': 2,
                'text': 'this is the document 2 text',
                'embedding': np.array([7, 8, 9]),
                'tokens': {'token_ids': [1, 2, 3, 4, 6], 'mask': [[1, ]]},
            },
        ]
        dataset = Dataset()
        for i, doc in enumerate(docs):
            dataset.append(doc)
        self.assertEqual(len(dataset), len(docs))  # add assertion here


if __name__ == '__main__':
    unittest.main()
