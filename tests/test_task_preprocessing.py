import unittest

import torch
from tklearn.nn.utils.preprocessing import preprocess_input, preprocess_target


class TestPreprocessInput(unittest.TestCase):
    def test_continuous(self):
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        result = preprocess_input("continuous", input_tensor, num_labels=1)
        self.assertTrue(torch.allclose(result, input_tensor))

    def test_continuous_multioutput(self):
        input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = preprocess_input("continuous-multioutput", input_tensor, num_labels=2)
        self.assertTrue(torch.allclose(result, input_tensor))

    def test_binary(self):
        input_tensor = torch.tensor([0, 1, 0, 1])
        result = preprocess_input("binary", input_tensor, num_labels=2)
        self.assertTrue(torch.allclose(result, input_tensor))

    def test_binary_one_hot(self):
        input_tensor = torch.tensor([[1, 0], [0, 1], [1, 0]])
        expected = torch.tensor([0, 1, 0])
        result = preprocess_input("binary", input_tensor, num_labels=2)
        self.assertTrue(torch.allclose(result, expected))

    def test_multiclass(self):
        input_tensor = torch.tensor([0, 2, 1, 3])
        result = preprocess_input("multiclass", input_tensor, num_labels=4)
        self.assertTrue(torch.allclose(result, input_tensor))

    def test_multiclass_one_hot(self):
        input_tensor = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]])
        expected = torch.tensor([0, 2, 1])
        result = preprocess_input("multiclass", input_tensor, num_labels=4)
        self.assertTrue(torch.allclose(result, expected))

    def test_multilabel_indicator(self):
        input_tensor = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
        result = preprocess_input("multilabel-indicator", input_tensor, num_labels=3)
        self.assertTrue(torch.allclose(result, input_tensor))

    def test_multiclass_multioutput(self):
        input_tensor = torch.tensor([[0, 1, 2], [2, 0, 1], [1, 2, 0]])
        result = preprocess_input("multiclass-multioutput", input_tensor, num_labels=3)
        self.assertTrue(torch.allclose(result, input_tensor))

    def test_invalid_shape(self):
        target_types = [
            "continuous",
            "continuous-multioutput",
            "binary",
            "multiclass",
            "multilabel-indicator",
            "multiclass-multioutput",
        ]
        for target_type in target_types:
            with self.subTest(target_type=target_type):
                if target_type in ["continuous", "binary"]:
                    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
                elif target_type in [
                    "continuous-multioutput",
                    "multilabel-indicator",
                    "multiclass-multioutput",
                ]:
                    input_tensor = torch.tensor([1.0, 2.0, 3.0])
                else:  # multiclass
                    input_tensor = torch.tensor([[[1.0]]])

                with self.assertRaises(ValueError):
                    preprocess_input(target_type, input_tensor, num_labels=2)

    def test_invalid_binary_values(self):
        input_tensor = torch.tensor([0, 1, 2])
        with self.assertRaises(ValueError):
            preprocess_input("binary", input_tensor, num_labels=2)

    def test_invalid_multilabel_values(self):
        input_tensor = torch.tensor([[0, 1, 2], [1, 0, 1]])
        with self.assertRaises(ValueError):
            preprocess_input("multilabel-indicator", input_tensor, num_labels=3)


class TestPreprocessTarget(unittest.TestCase):
    def test_continuous(self):
        logits = torch.tensor([[1.0], [2.0], [3.0]])
        y_pred, y_score = preprocess_target("continuous", logits)
        self.assertTrue(torch.allclose(y_pred, torch.tensor([1.0, 2.0, 3.0])))
        self.assertTrue(torch.allclose(y_score, torch.tensor([1.0, 2.0, 3.0])))

    def test_continuous_multioutput(self):
        logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y_pred, y_score = preprocess_target("continuous-multioutput", logits)
        self.assertTrue(torch.allclose(y_pred, logits))
        self.assertTrue(torch.allclose(y_score, logits))

    def test_binary(self):
        logits = torch.tensor([-1.0, 0.0, 1.0])
        y_pred, y_score = preprocess_target("binary", logits)
        self.assertTrue(torch.allclose(y_pred, torch.tensor([0, 0, 1])))
        self.assertTrue(torch.allclose(y_score, torch.sigmoid(logits)))

    def test_binary_2d(self):
        logits = torch.tensor([[-1.0, 1.0], [1.0, -1.0]])
        y_pred, y_score = preprocess_target("binary", logits)
        self.assertTrue(torch.allclose(y_pred, torch.tensor([1, 0])))
        self.assertTrue(torch.allclose(y_score, torch.sigmoid(logits[:, 1])))

    def test_multiclass(self):
        logits = torch.tensor([[1.0, 2.0, 0.0], [0.0, 5.0, 1.0]])
        y_pred, y_score = preprocess_target("multiclass", logits)
        self.assertTrue(torch.allclose(y_pred, torch.tensor([1, 1])))
        self.assertTrue(torch.allclose(y_score, torch.softmax(logits, dim=-1)))

    def test_multilabel_indicator(self):
        logits = torch.tensor([[-0.1, -1.0, 2.0], [-0.5, 1.5, 0.0]])
        y_pred, y_score = preprocess_target("multilabel-indicator", logits)
        expected_pred = torch.tensor([[0, 0, 1], [0, 1, 0]])
        self.assertTrue(torch.allclose(y_pred, expected_pred))
        self.assertTrue(torch.allclose(y_score, torch.sigmoid(logits)))

    def test_multiclass_multioutput(self):
        logits = (
            torch.tensor([[1.0, 2.0], [0.0, 5.0]]),
            torch.tensor([[0.0, 3.0], [2.0, 1.0]]),
        )
        y_pred, y_score = preprocess_target("multiclass-multioutput", logits)
        self.assertTrue(torch.allclose(y_pred[0], torch.tensor([1, 1])))
        self.assertTrue(torch.allclose(y_pred[1], torch.tensor([1, 0])))
        self.assertTrue(torch.allclose(y_score[0], torch.softmax(logits[0], dim=-1)))
        self.assertTrue(torch.allclose(y_score[1], torch.softmax(logits[1], dim=-1)))

    def test_invalid_shape(self):
        target_types = [
            "continuous",
            "continuous-multioutput",
            "binary",
            "multiclass",
            "multilabel-indicator",
            "multiclass-multioutput",
        ]
        for target_type in target_types:
            with self.subTest(target_type=target_type):
                if target_type in ["continuous", "binary"]:
                    logits = torch.tensor([[[1.0]]])
                elif target_type in ["continuous-multioutput", "multilabel-indicator"]:
                    logits = torch.tensor([1.0, 2.0, 3.0])
                elif target_type == "multiclass":
                    logits = torch.tensor([1.0, 2.0, 3.0])
                else:  # multiclass-multioutput
                    logits = torch.tensor([1.0, 2.0, 3.0])
                with self.assertRaises(ValueError):
                    print(preprocess_target(target_type, logits))

    def test_unsupported_target_type(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        with self.assertRaises(NotImplementedError):
            preprocess_target("anything", logits)


if __name__ == "__main__":
    unittest.main()
