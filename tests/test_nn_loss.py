import unittest

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from tklearn.nn.loss import TargetBasedLoss


class TestTargetBasedLoss(unittest.TestCase):
    def setUp(self):
        # Set a random seed for reproducibility
        torch.manual_seed(42)
        # Common test dimensions
        self.batch_size = 8
        self.num_features = 10

    def test_continuous_target(self):
        """Test with explicitly defined continuous target type"""
        # Setup
        num_labels = 1
        loss_fn = TargetBasedLoss(
            target_type="continuous", num_labels=num_labels
        )

        # Input tensors
        inputs = torch.randn(self.batch_size, num_labels)
        targets = torch.randn(self.batch_size, num_labels)

        # Expected result using MSELoss
        expected_loss = MSELoss()(inputs.squeeze(), targets.squeeze())

        # Test forward pass
        actual_loss = loss_fn(inputs, targets)

        # Verify loss and internal state
        self.assertTrue(torch.isclose(actual_loss, expected_loss))
        self.assertEqual(loss_fn.target_type.label, "continuous")
        self.assertIsInstance(loss_fn._loss_func, MSELoss)

    def test_multiclass_target(self):
        """Test with explicitly defined multiclass target type"""
        # Setup
        num_labels = 5  # 5 classes
        loss_fn = TargetBasedLoss(
            target_type="multiclass", num_labels=num_labels
        )

        # Input tensors
        inputs = torch.randn(self.batch_size, num_labels)  # logits
        targets = torch.randint(
            0, num_labels, (self.batch_size,)
        )  # class indices

        # Expected result using CrossEntropyLoss
        expected_loss = CrossEntropyLoss()(
            inputs.view(-1, num_labels), targets.view(-1)
        )

        # Test forward pass
        actual_loss = loss_fn(inputs, targets)

        # Verify loss and internal state
        self.assertTrue(torch.isclose(actual_loss, expected_loss))
        self.assertEqual(loss_fn.target_type.label, "multiclass")
        self.assertIsInstance(loss_fn._loss_func, CrossEntropyLoss)

    def test_binary_target(self):
        """Test with explicitly defined binary target type"""
        # Setup
        num_labels = 1
        loss_fn = TargetBasedLoss(target_type="binary", num_labels=num_labels)

        # Input tensors
        inputs = torch.randn(self.batch_size, num_labels)  # logits
        targets = torch.randint(0, 2, (self.batch_size,))  # binary targets

        # Prepare expected result using BCEWithLogitsLoss
        targets_float = targets.view(-1, 1).to(dtype=inputs.dtype)
        expected_loss = BCEWithLogitsLoss()(inputs, targets_float)

        # Test forward pass
        actual_loss = loss_fn(inputs, targets)

        # Verify loss and internal state
        self.assertTrue(torch.isclose(actual_loss, expected_loss))
        self.assertEqual(loss_fn.target_type.label, "binary")
        self.assertIsInstance(loss_fn._loss_func, BCEWithLogitsLoss)

    def test_multilabel_target(self):
        """Test with explicitly defined multilabel-indicator target type"""
        # Setup
        num_labels = 3  # 3 possible labels
        loss_fn = TargetBasedLoss(
            target_type="multilabel-indicator", num_labels=num_labels
        )

        # Input tensors
        inputs = torch.randn(self.batch_size, num_labels)  # logits
        targets = torch.randint(
            0, 2, (self.batch_size, num_labels)
        ).float()  # multi-hot encoding

        # Expected result using BCEWithLogitsLoss
        expected_loss = BCEWithLogitsLoss()(inputs, targets)

        # Test forward pass
        actual_loss = loss_fn(inputs, targets)

        # Verify loss and internal state
        self.assertTrue(torch.isclose(actual_loss, expected_loss))
        self.assertEqual(loss_fn.target_type.label, "multilabel-indicator")
        self.assertIsInstance(loss_fn._loss_func, BCEWithLogitsLoss)


if __name__ == "__main__":
    unittest.main()
