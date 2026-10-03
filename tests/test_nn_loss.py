import unittest

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from tklearn.nn.loss import TargetBasedLoss


class TestTargetBasedLoss(unittest.TestCase):
    def setUp(self):
        # Set a random seed for reproducibility
        torch.manual_seed(42)
        self.batch_size = 8

    def assertLossEqual(self, actual, expected):
        self.assertTrue(torch.isclose(actual, expected), (actual, expected))

    def test_continuous_target(self):
        """A 1D target is matched to (batch, 1) predictions"""
        loss_fn = TargetBasedLoss("continuous")
        inputs = torch.randn(self.batch_size, 1)
        targets = torch.randn(self.batch_size)
        expected = MSELoss()(inputs.squeeze(-1), targets)
        self.assertLossEqual(loss_fn(inputs, targets), expected)
        self.assertEqual(loss_fn.target_type.label, "continuous")

    def test_continuous_multioutput_target(self):
        loss_fn = TargetBasedLoss("continuous-multioutput")
        inputs = torch.randn(self.batch_size, 3)
        targets = torch.randn(self.batch_size, 3)
        self.assertLossEqual(
            loss_fn(inputs, targets), MSELoss()(inputs, targets)
        )

    def test_multiclass_target(self):
        num_labels = 5
        loss_fn = TargetBasedLoss("multiclass")
        inputs = torch.randn(self.batch_size, num_labels)
        targets = torch.randint(0, num_labels, (self.batch_size,))
        expected = CrossEntropyLoss()(inputs, targets)
        self.assertLossEqual(loss_fn(inputs, targets), expected)
        # class indices stored as floats are cast to long
        self.assertLossEqual(loss_fn(inputs, targets.float()), expected)

    def test_multiclass_token_level_target(self):
        """(batch, seq, classes) logits with (batch, seq) targets"""
        num_labels, seq_len = 5, 7
        loss_fn = TargetBasedLoss("multiclass")
        inputs = torch.randn(self.batch_size, seq_len, num_labels)
        targets = torch.randint(0, num_labels, (self.batch_size, seq_len))
        expected = CrossEntropyLoss()(
            inputs.reshape(-1, num_labels), targets.reshape(-1)
        )
        self.assertLossEqual(loss_fn(inputs, targets), expected)

    def test_binary_target(self):
        loss_fn = TargetBasedLoss("binary")
        targets = torch.randint(0, 2, (self.batch_size,))
        # (batch, 1) logits
        inputs = torch.randn(self.batch_size, 1)
        expected = BCEWithLogitsLoss()(inputs, targets.view(-1, 1).float())
        self.assertLossEqual(loss_fn(inputs, targets), expected)
        # (batch,) logits
        inputs = torch.randn(self.batch_size)
        expected = BCEWithLogitsLoss()(inputs, targets.float())
        self.assertLossEqual(loss_fn(inputs, targets), expected)

    def test_multilabel_target(self):
        num_labels = 3
        loss_fn = TargetBasedLoss("multilabel-indicator")
        inputs = torch.randn(self.batch_size, num_labels)
        targets = torch.randint(0, 2, (self.batch_size, num_labels))
        expected = BCEWithLogitsLoss()(inputs, targets.float())
        self.assertLossEqual(loss_fn(inputs, targets), expected)

    def test_kwargs_are_passed_to_loss(self):
        num_labels = 3
        weight = torch.tensor([1.0, 2.0, 0.5])
        loss_fn = TargetBasedLoss("multiclass", weight=weight)
        inputs = torch.randn(self.batch_size, num_labels)
        targets = torch.randint(0, num_labels, (self.batch_size,))
        expected = CrossEntropyLoss(weight=weight)(inputs, targets)
        self.assertLossEqual(loss_fn(inputs, targets), expected)

    def test_unsupported_target_type(self):
        with self.assertRaises(ValueError):
            TargetBasedLoss("multiclass-multioutput")
        with self.assertRaises(ValueError):
            TargetBasedLoss("not-a-target-type")


if __name__ == "__main__":
    unittest.main()
