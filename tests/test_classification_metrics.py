import unittest

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from tklearn.metrics.confusion_matrix import ConfusionMatrix


class TestClassificationMetrics(unittest.TestCase):  # noqa: PLR0904
    def test_accuracy_score_binary(self):
        # Create a ConfusionMatrix object
        cm = ConfusionMatrix()
        # Generate some example data
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        # Update the confusion matrix
        cm.update(y_true, y_pred)
        # Test accuracy_score
        acc = accuracy_score(y_true, y_pred)
        self.assertEqual(cm.accuracy_score(), acc)

    def test_precision_score_binary(self):
        # Create a ConfusionMatrix object
        cm = ConfusionMatrix()
        # Generate some example data
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        # Update the confusion matrix
        cm.update(y_true, y_pred)
        # Test precision_score
        precision = precision_score(y_true, y_pred)
        self.assertEqual(cm.precision_score(), precision)

    def test_recall_score_binary(self):
        # Create a ConfusionMatrix object
        cm = ConfusionMatrix()
        # Generate some example data
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        # Update the confusion matrix
        cm.update(y_true, y_pred)
        # Test recall_score
        recall = recall_score(y_true, y_pred)
        self.assertEqual(cm.recall_score(), recall)

    def test_f1_score_binary(self):
        # Create a ConfusionMatrix object
        cm = ConfusionMatrix()
        # Generate some example data
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        # Update the confusion matrix
        cm.update(y_true, y_pred)
        # Test f1_score
        f1 = f1_score(y_true, y_pred)
        self.assertEqual(cm.f1_score(), f1)

    def test_accuracy_score_multilabel(self):
        # Create a ConfusionMatrix object
        cm = ConfusionMatrix()
        # Generate some example data
        y_true = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
        y_pred = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 1, 0]])
        # Update the confusion matrix
        cm.update(y_true, y_pred)
        # Test accuracy_score
        acc = accuracy_score(y_true, y_pred)
        self.assertEqual(cm.accuracy_score(), acc)

    def test_micro_f1_score_multilabel(self):
        # Create a ConfusionMatrix object
        cm = ConfusionMatrix()
        # Generate some example data
        y_true = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
        y_pred = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 1, 0]])
        # Update the confusion matrix
        cm.update(y_true, y_pred)
        # Test f1_score
        f1 = f1_score(y_true, y_pred, average="micro")
        self.assertEqual(cm.f1_score(average="micro"), f1)

    def test_macro_f1_score_multilabel(self):
        # Create a ConfusionMatrix object
        cm = ConfusionMatrix()
        # Generate some example data
        y_true = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
        y_pred = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 1, 0]])
        # Update the confusion matrix
        cm.update(y_true, y_pred)
        # Test f1_score
        f1 = f1_score(y_true, y_pred, average="macro")
        self.assertEqual(cm.f1_score(average="macro"), f1)

    def test_micro_precision_score_multilabel(self):
        # Create a ConfusionMatrix object
        cm = ConfusionMatrix()
        # Generate some example data
        y_true = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
        y_pred = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 1, 0]])
        # Update the confusion matrix
        cm.update(y_true, y_pred)
        # Test precision_score
        precision = precision_score(y_true, y_pred, average="micro")
        self.assertEqual(cm.precision_score(average="micro"), precision)

    def test_macro_precision_score_multilabel(self):
        # Create a ConfusionMatrix object
        cm = ConfusionMatrix()
        # Generate some example data
        y_true = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
        y_pred = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 1, 0]])
        # Update the confusion matrix
        cm.update(y_true, y_pred)
        # Test precision_score
        precision = precision_score(y_true, y_pred, average="macro")
        self.assertEqual(cm.precision_score(average="macro"), precision)

    def test_micro_recall_score_multilabel(self):
        # Create a ConfusionMatrix object
        cm = ConfusionMatrix()
        # Generate some example data
        y_true = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
        y_pred = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 1, 0]])
        # Update the confusion matrix
        cm.update(y_true, y_pred)
        # Test recall_score
        recall = recall_score(y_true, y_pred, average="micro")
        self.assertEqual(cm.recall_score(average="micro"), recall)

    def test_macro_recall_score_multilabel(self):
        # Create a ConfusionMatrix object
        cm = ConfusionMatrix()
        # Generate some example data
        y_true = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
        y_pred = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 1, 0]])
        # Update the confusion matrix
        cm.update(y_true, y_pred)
        # Test recall_score
        recall = recall_score(y_true, y_pred, average="macro")
        self.assertEqual(cm.recall_score(average="macro"), recall)

    def test_accuracy_score_multiclass(self):
        # Create a ConfusionMatrix object
        cm = ConfusionMatrix()
        # Generate some example data
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 1, 2])
        # Update the confusion matrix
        cm.update(y_true, y_pred)
        # Test accuracy_score
        acc = accuracy_score(y_true, y_pred)
        self.assertEqual(cm.accuracy_score(), acc)

    def test_micro_f1_score_multiclass(self):
        # Create a ConfusionMatrix object
        cm = ConfusionMatrix()
        # Generate some example data
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 1, 2])
        # Update the confusion matrix
        cm.update(y_true, y_pred)
        # Test accuracy_score
        acc = f1_score(y_true, y_pred, average="micro")
        self.assertEqual(cm.f1_score(average="micro"), acc)

    def test_macro_f1_score_multiclass(self):
        # Create a ConfusionMatrix object
        cm = ConfusionMatrix()
        # Generate some example data
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 1, 2])
        # Update the confusion matrix
        cm.update(y_true, y_pred)
        # Test accuracy_score
        acc = f1_score(y_true, y_pred, average="macro")
        self.assertEqual(cm.f1_score(average="macro"), acc)

    def test_micro_precision_score_multiclass(self):
        # Create a ConfusionMatrix object
        cm = ConfusionMatrix()
        # Generate some example data
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 1, 2])
        # Update the confusion matrix
        cm.update(y_true, y_pred)
        # Test accuracy_score
        acc = precision_score(y_true, y_pred, average="micro")
        self.assertEqual(cm.precision_score(average="micro"), acc)

    def test_macro_precision_score_multiclass(self):
        # Create a ConfusionMatrix object
        cm = ConfusionMatrix()
        # Generate some example data
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 1, 2])
        # Update the confusion matrix
        cm.update(y_true, y_pred)
        # Test accuracy_score
        acc = precision_score(y_true, y_pred, average="macro")
        self.assertEqual(cm.precision_score(average="macro"), acc)

    def test_micro_recall_score_multiclass(self):
        # Create a ConfusionMatrix object
        cm = ConfusionMatrix()
        # Generate some example data
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 1, 2])
        # Update the confusion matrix
        cm.update(y_true, y_pred)
        # Test recall_score
        res = recall_score(y_true, y_pred, average="micro")
        self.assertEqual(cm.recall_score(average="micro"), res)

    def test_macro_recall_score_multiclass(self):
        # Create a ConfusionMatrix object
        cm = ConfusionMatrix()
        # Generate some example data
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 1, 2])
        # Update the confusion matrix
        cm.update(y_true, y_pred)
        # Test recall_score
        res = recall_score(y_true, y_pred, average="macro")
        self.assertEqual(cm.recall_score(average="macro"), res)

    def test_accuracy_score_binary_multi_update(self):
        # Create a ConfusionMatrix object
        cm = ConfusionMatrix()
        # Generate some example data
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        # Split the data into two parts
        y_true_1, y_true_2 = np.split(y_true, 2)
        y_pred_1, y_pred_2 = np.split(y_pred, 2)
        # Update the confusion matrix with the first part
        cm.update(y_true_1, y_pred_1)
        # Update the confusion matrix with the second part
        cm.update(y_true_2, y_pred_2)
        # Test accuracy_score
        acc = accuracy_score(y_true, y_pred)
        self.assertEqual(cm.accuracy_score(), acc)

    def test_f1_score_binary_multi_update(self):
        # Create a ConfusionMatrix object
        cm = ConfusionMatrix()
        # Generate some example data
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        # Split the data into two parts
        y_true_1, y_true_2 = np.split(y_true, 2)
        y_pred_1, y_pred_2 = np.split(y_pred, 2)
        # Update the confusion matrix with the first part
        cm.update(y_true_1, y_pred_1)
        # Update the confusion matrix with the second part
        cm.update(y_true_2, y_pred_2)
        # Test f1_score
        f1 = f1_score(y_true, y_pred)
        self.assertEqual(cm.f1_score(), f1)

    def test_precision_score_binary_multi_update(self):
        # Create a ConfusionMatrix object
        cm = ConfusionMatrix()
        # Generate some example data
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        # Split the data into two parts
        y_true_1, y_true_2 = np.split(y_true, 2)
        y_pred_1, y_pred_2 = np.split(y_pred, 2)
        # Update the confusion matrix with the first part
        cm.update(y_true_1, y_pred_1)
        # Update the confusion matrix with the second part
        cm.update(y_true_2, y_pred_2)
        # Test precision_score
        precision = precision_score(y_true, y_pred)
        self.assertEqual(cm.precision_score(), precision)

    def test_recall_score_binary_multi_update(self):
        # Create a ConfusionMatrix object
        cm = ConfusionMatrix()
        # Generate some example data
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        # Split the data into two parts
        y_true_1, y_true_2 = np.split(y_true, 2)
        y_pred_1, y_pred_2 = np.split(y_pred, 2)
        # Update the confusion matrix with the first part
        cm.update(y_true_1, y_pred_1)
        # Update the confusion matrix with the second part
        cm.update(y_true_2, y_pred_2)
        # Test recall_score
        recall = recall_score(y_true, y_pred)
        self.assertEqual(cm.recall_score(), recall)


if __name__ == "__main__":
    unittest.main()
