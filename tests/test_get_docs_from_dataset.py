import unittest

import numpy as np

from tklearn.datasets import Dataset


class TestGetDocumentsFromDataset(unittest.TestCase):
    def test_get_docs_from_dataset(self):
        docs = [
            {
                'id': 0,
                'text': 'this is the document text',
                'embedding': np.array([1, 2, 3]),
                'tokens': {'token_ids': [1, 2, 3], 'mask': [[1, 2, 6], [3, 4]]},

            },
        ]
        dataset = Dataset()
        for i, doc in enumerate(docs):
            dataset.append(doc)
        embedding = dataset[0]['embedding']
        self.assertIsInstance(embedding, np.ndarray)
        shape = embedding.shape
        ndims = len(shape)
        self.assertEqual(ndims, 1)


if __name__ == '__main__':
    unittest.main()
