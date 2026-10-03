import tempfile
import unittest
from pathlib import Path

import numpy as np

from tklearn import config
from tklearn.embeddings import WordEmbedding

VECTORS = {
    "king": np.array([1.0, 0.0, 0.5], dtype=np.float32),
    "queen": np.array([0.9, 0.1, 0.5], dtype=np.float32),
    "apple": np.array([0.0, 1.0, 0.0], dtype=np.float32),
}


class CountingEmbedding(WordEmbedding):
    loader = "test"
    loads = 0

    def load_vectors(self):
        type(self).loads += 1
        return VECTORS


class TestWordEmbedding(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_base_dir = config.base_dir
        config.base_dir = Path(self.tmp.name)
        CountingEmbedding.loads = 0

    def tearDown(self):
        config.base_dir = self.old_base_dir
        self.tmp.cleanup()

    def test_cache_round_trip(self):
        built = CountingEmbedding("tiny")
        cached = CountingEmbedding("tiny")
        self.assertEqual(CountingEmbedding.loads, 1)  # second load is cached
        for word, vector in VECTORS.items():
            np.testing.assert_array_equal(built[word], vector)
            np.testing.assert_array_equal(cached[word], vector)

    def test_mapping_and_encode(self):
        emb = CountingEmbedding("tiny")
        self.assertEqual(len(emb), 3)
        self.assertEqual(list(emb), ["king", "queen", "apple"])
        self.assertIn("king", emb)
        self.assertEqual((emb.dim, emb.shape), (3, (3, 3)))
        self.assertEqual(emb.encode("king").shape, (3,))
        self.assertEqual(emb.encode(["king", "apple"]).shape, (2, 3))
        np.testing.assert_array_equal(
            emb.encode_query("king"), emb.encode_document("king")
        )

    def test_unknown_word(self):
        emb = CountingEmbedding("tiny")
        self.assertNotIn("pear", emb)
        with self.assertRaises(KeyError):
            emb["pear"]


if __name__ == "__main__":
    unittest.main()
