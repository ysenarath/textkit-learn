import unittest

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from tklearn.metrics import (
    AUC,
    F1,
    Accuracy,
    Metric,
    MetricCollection,
    OptimalAUCThreshold,
    OptimalPRThreshold,
    Precision,
    Recall,
)


class TestMetricCollection(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.y_true = rng.integers(0, 3, 100)
        scores = rng.random((100, 3))
        self.y_score = scores / scores.sum(axis=1, keepdims=True)
        self.y_pred = self.y_score.argmax(axis=1)

    def update_in_batches(self, metrics, batch_size=32):
        for i in range(0, len(self.y_true), batch_size):
            s = slice(i, i + batch_size)
            metrics.update(
                y_true=torch.tensor(self.y_true[s]),  # tensors are accepted
                y_pred=self.y_pred[s],
                y_score=self.y_score[s],
                embedding=None,  # unused inputs are ignored
            )

    def test_batched_results_match_sklearn(self):
        metrics = MetricCollection({
            "acc": Accuracy(),
            "f1": F1(average="macro"),
            "precision": Precision(average="macro"),
            "recall": Recall(average="macro"),
            "auc": AUC(multi_class="ovr"),
        })
        self.update_in_batches(metrics)
        result = metrics.result()
        y, p, s = self.y_true, self.y_pred, self.y_score
        self.assertAlmostEqual(result["acc"], accuracy_score(y, p))
        self.assertAlmostEqual(result["f1"], f1_score(y, p, average="macro"))
        self.assertAlmostEqual(
            result["precision"], precision_score(y, p, average="macro")
        )
        self.assertAlmostEqual(
            result["recall"], recall_score(y, p, average="macro")
        )
        self.assertAlmostEqual(
            result["auc"], roc_auc_score(y, s, multi_class="ovr")
        )

    def test_results_are_python_scalars(self):
        metrics = MetricCollection({"acc": Accuracy(), "f1": F1("macro")})
        self.update_in_batches(metrics)
        for value in metrics.result().values():
            self.assertIs(type(value), float)

    def test_reset(self):
        metrics = MetricCollection({"acc": Accuracy()})
        metrics.update(y_true=[0, 0], y_pred=[1, 1])
        metrics.reset()
        metrics.update(y_true=[1, 1], y_pred=[1, 1])
        self.assertEqual(metrics.result(), {"acc": 1.0})

    def test_missing_input_raises(self):
        metrics = MetricCollection({"auc": AUC()})
        with self.assertRaisesRegex(KeyError, "y_score"):
            metrics.update(y_true=[0, 1])

    def test_rejects_non_metric(self):
        with self.assertRaises(TypeError):
            MetricCollection({"bad": object()})

    def test_per_class_auc(self):
        metrics = MetricCollection({
            "auc": AUC(multi_class="ovr", average=None)
        })
        self.update_in_batches(metrics)
        self.assertEqual(metrics.result()["auc"].shape, (3,))


class TestMetric(unittest.TestCase):
    def test_direct_call(self):
        f1 = F1(average="macro")
        self.assertAlmostEqual(f1(y_true=[0, 1, 1], y_pred=[0, 1, 0]), 2 / 3)

    def test_custom_metric(self):
        class MeanError(Metric):
            inputs = ("y_true", "y_pred")
            optional_inputs = ()

            def compute(self, y_true, y_pred):
                return np.mean(np.abs(y_true - y_pred))

        self.assertEqual(
            MeanError()(y_true=[1.0, 2.0], y_pred=[1.0, 4.0]), 1.0
        )

    def test_repr(self):
        self.assertEqual(
            repr(Accuracy()),
            "Accuracy(normalize=True, balanced=False, adjusted=False)",
        )

    def test_optimal_thresholds(self):
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.4, 0.35, 0.8]
        roc = OptimalAUCThreshold()(y_true=y_true, y_score=y_score)
        pr = OptimalPRThreshold()(y_true=y_true, y_score=y_score)
        for result in (roc, pr):
            self.assertIn(result["threshold"], y_score + [np.inf])
            optimal = [p for p in result["data"] if p["optimal"]]
            self.assertEqual(len(optimal), 1)
            self.assertEqual(optimal[0]["threshold"], result["threshold"])
        # precision and recall are reported as-is, with F1 alongside
        point = pr["data"][0]
        self.assertAlmostEqual(
            point["f1"],
            2
            * point["precision"]
            * point["recall"]
            / (point["precision"] + point["recall"]),
        )


if __name__ == "__main__":
    unittest.main()
