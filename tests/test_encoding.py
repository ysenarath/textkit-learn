import unittest

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from tklearn.nn import Encoder
from tklearn.nn.callbacks import ProgbarLogger
from tklearn.nn.models import AutoModel
from tklearn.nn.utils import get_device

MODEL_NAME_OR_PATH = "google-bert/bert-base-uncased"
DATASET = "yelp_review_full"


class TestEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = AutoModel({
            "type": "linear",
            "backbone": {
                "type": "transformer",
                "model_name_or_path": MODEL_NAME_OR_PATH,
            },
            "num_labels": 5,
        })
        cls.model.to(get_device())
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
        dataset = load_dataset(DATASET, split="train").select(range(8))
        dataset = dataset.rename_column("label", "labels").map(
            lambda examples: tokenizer(
                examples["text"], truncation=True, padding="max_length"
            ),
            batched=True,
            remove_columns=["text"],
        )
        dataset.set_format(type="torch")
        cls.dataset = dataset
        cls.hidden_size = cls.model.backbone.hidden_size

    def encode(self, **kwargs):
        dataloader = DataLoader(self.dataset, batch_size=32)
        encoder = Encoder(self.model, callbacks=[ProgbarLogger()])
        return encoder.encode(dataloader, **kwargs)

    def test_encode_return_pt(self):
        encodings = self.encode(return_tensors="pt")
        self.assertIsInstance(encodings, torch.Tensor)
        self.assertEqual(
            tuple(encodings.shape), (len(self.dataset), self.hidden_size)
        )

    def test_encode_return_np(self):
        encodings = self.encode(return_tensors="np")
        self.assertIsInstance(encodings, np.ndarray)
        self.assertEqual(
            encodings.shape, (len(self.dataset), self.hidden_size)
        )

    def test_encode_return_none_requires_list(self):
        with self.assertRaises(ValueError):
            self.encode(return_tensors=None, return_list=False)

    def test_encode_return_lists(self):
        for return_tensors, item_type in [
            ("pt", torch.Tensor),
            ("np", np.ndarray),
            (None, list),
        ]:
            with self.subTest(return_tensors=return_tensors):
                encodings = self.encode(
                    return_tensors=return_tensors, return_list=True
                )
                self.assertIsInstance(encodings, list)
                self.assertEqual(len(encodings), len(self.dataset))
                for item in encodings:
                    self.assertIsInstance(item, item_type)
                    self.assertEqual(len(item), self.hidden_size)


if __name__ == "__main__":
    unittest.main()
