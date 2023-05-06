import unittest

import numpy as np

from tklearn.datasets import Dataset


class TestGetFieldsFromDataset(unittest.TestCase):
    def test_get_fields_from_dataset(self):
        dataset = Dataset()
        dataset.append({
            'id': 0,
            'text': 'this is the document text',
            'embedding': np.array([1, 2, 3]),
            'tokens': {'token_ids': [1, 2, 3], 'mask': [0, 1, 1]},

        })
        field_embedding = dataset.fields['embedding']
        embedding = field_embedding[0]
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEquals(embedding.shape[0], 3)


if __name__ == '__main__':
    unittest.main()
