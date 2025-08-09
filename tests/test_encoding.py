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


class TestEarlyStopping(unittest.TestCase):
    # setup
    def setUp(self):
        self.model = AutoModel({
            "type": "linear",
            "backbone": {
                "type": "transformer",
                "model_name_or_path": MODEL_NAME_OR_PATH,
            },
            "num_labels": 5,
        })

        self.auto_device = get_device()
        self.model.to(self.auto_device)

        self.dataset = load_dataset(DATASET, split="train").select(range(8))
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], truncation=True, padding="max_length"
            )

        self.dataset = self.dataset.rename_column("label", "labels").map(
            tokenize_function, batched=True, remove_columns=["text"]
        )
        self.dataset.set_format(type="torch")

    def test_encode_return_pt(self):
        dataloader = DataLoader(
            self.dataset,
            batch_size=32,
            pin_memory=True,
        )
        encoder = Encoder(
            self.model,
            dataloader=dataloader,
            callbacks=[ProgbarLogger()],
        )
        pooler_output = encoder.encode(
            return_tensors="pt",
            return_list=False,
        )
        self.assertIsInstance(pooler_output, torch.Tensor)
        expected_shape = (len(self.dataset), self.model.backbone.hidden_size)
        self.assertEqual(pooler_output.shape, expected_shape)

    def test_encode_return_pt_2(self):
        dataloader = DataLoader(
            self.dataset,
            batch_size=32,
            pin_memory=True,
        )
        encoder = Encoder(
            self.model,
            dataloader=dataloader,
            callbacks=[ProgbarLogger()],
        )
        pooler_output = encoder.encode(
            return_tensors=None,
            return_list=False,
        )
        self.assertIsInstance(pooler_output, torch.Tensor)
        expected_shape = (len(self.dataset), self.model.backbone.hidden_size)
        self.assertEqual(tuple(pooler_output.shape), expected_shape)

    def test_encode_return_np(self):
        dataloader = DataLoader(
            self.dataset,
            batch_size=32,
            pin_memory=True,
        )
        encoder = Encoder(
            self.model,
            dataloader=dataloader,
            callbacks=[ProgbarLogger()],
        )
        pooler_output = encoder.encode(
            return_tensors="np",
            return_list=False,
        )
        self.assertIsInstance(pooler_output, np.ndarray)
        expected_shape = (len(self.dataset), self.model.backbone.hidden_size)
        self.assertEqual(pooler_output.shape, expected_shape)

    def test_encode_return_list_of_np(self):
        dataloader = DataLoader(
            self.dataset,
            batch_size=32,
            pin_memory=True,
        )
        encoder = Encoder(
            self.model,
            dataloader=dataloader,
            callbacks=[ProgbarLogger()],
        )
        pooler_output = encoder.encode(
            return_tensors="np",
            return_list=True,
        )
        self.assertIsInstance(pooler_output, list)
        self.assertEqual(len(pooler_output), len(self.dataset))
        self.assertTrue(all(isinstance(t, np.ndarray) for t in pooler_output))
        hidden_size = self.model.backbone.hidden_size
        self.assertTrue(all(len(t) == hidden_size for t in pooler_output))

    def test_encode_return_list_of_pt(self):
        dataloader = DataLoader(
            self.dataset,
            batch_size=32,
            pin_memory=True,
        )
        encoder = Encoder(
            self.model,
            dataloader=dataloader,
            callbacks=[ProgbarLogger()],
        )
        pooler_output = encoder.encode(
            return_tensors="pt",
            return_list=True,
        )
        self.assertIsInstance(pooler_output, list)
        self.assertEqual(len(pooler_output), len(self.dataset))
        self.assertTrue(
            all(isinstance(t, torch.Tensor) for t in pooler_output)
        )
        hidden_size = self.model.backbone.hidden_size
        self.assertTrue(all(len(t) == hidden_size for t in pooler_output))

    def test_encode_return_list_of_list(self):
        dataloader = DataLoader(
            self.dataset,
            batch_size=32,
            pin_memory=True,
        )
        encoder = Encoder(
            self.model,
            dataloader=dataloader,
            callbacks=[ProgbarLogger()],
        )
        pooler_output = encoder.encode(return_tensors=None, return_list=True)
        self.assertIsInstance(pooler_output, list)
        self.assertEqual(len(pooler_output), len(self.dataset))
        self.assertTrue(all(isinstance(t, list) for t in pooler_output))
        hidden_size = self.model.backbone.hidden_size
        self.assertTrue(all(len(t) == hidden_size for t in pooler_output))


if __name__ == "__main__":
    unittest.main()
