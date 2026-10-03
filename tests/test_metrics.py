import unittest

import numpy as np
from scipy import stats
from sklearn import metrics as sk
from sklearn.preprocessing import label_binarize

from tklearn.metrics import (
    AUROC,
    F1,
    Accuracy,
    AveragePrecision,
    BalancedAccuracy,
    ConfusionMatrix,
    FBeta,
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
    OptimalThreshold,
    PearsonCorrelation,
    Precision,
    PrecisionRecallCurve,
    R2Score,
    Recall,
    ROCCurve,
    RootMeanSquaredError,
    SpanF1,
    SpanPrecision,
    SpanRecall,
    SpearmanCorrelation,
    get_spans,
)


def accumulate(metric, *arrays, batch_size=37):
    """Update `metric` batch by batch and compute it."""
    for i in range(0, len(arrays[0]), batch_size):
        metric.update(*(a[i : i + batch_size] for a in arrays))
    return metric.compute()


class TestClassification(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 200
        self.y_true = rng.integers(0, 4, n)
        noise = rng.integers(0, 4, n)
        self.y_pred = np.where(rng.random(n) < 0.6, self.y_true, noise)
        self.ml_true = rng.integers(0, 2, (n, 5))
        self.ml_pred = rng.integers(0, 2, (n, 5))
        self.bin_true = rng.integers(0, 2, n)
        self.bin_pred = rng.integers(0, 2, n)

    def test_prf_multiclass(self):
        cases = [
            (Precision, sk.precision_score),
            (Recall, sk.recall_score),
            (F1, sk.f1_score),
        ]
        for cls, func in cases:
            for average in ["micro", "macro", "weighted", None]:
                with self.subTest(metric=cls.__name__, average=average):
                    actual = accumulate(
                        cls(average=average), self.y_true, self.y_pred
                    )
                    expected = func(
                        self.y_true,
                        self.y_pred,
                        average=average,
                        zero_division=0.0,
                    )
                    np.testing.assert_allclose(actual, expected)

    def test_fbeta(self):
        actual = accumulate(
            FBeta(2.0, average="macro"), self.y_true, self.y_pred
        )
        expected = sk.fbeta_score(
            self.y_true, self.y_pred, beta=2.0, average="macro"
        )
        self.assertAlmostEqual(actual, expected)

    def test_zero_division(self):
        # class 3 is never predicted, so its precision is 0/0
        y_pred = np.where(self.y_pred == 3, 0, self.y_pred)
        for zero_division in [0.0, 1.0, np.nan]:
            for average in ["macro", None]:
                with self.subTest(zero_division=zero_division, avg=average):
                    metric = Precision(
                        average=average, zero_division=zero_division
                    )
                    actual = accumulate(metric, self.y_true, y_pred)
                    expected = sk.precision_score(
                        self.y_true,
                        y_pred,
                        average=average,
                        zero_division=zero_division,
                    )
                    np.testing.assert_allclose(actual, expected)

    def test_num_classes_includes_absent_classes(self):
        actual = accumulate(
            F1(average="macro", num_classes=6), self.y_true, self.y_pred
        )
        expected = sk.f1_score(
            self.y_true,
            self.y_pred,
            labels=range(6),
            average="macro",
            zero_division=0.0,
        )
        self.assertAlmostEqual(actual, expected)

    def test_binary(self):
        cases = [
            (Precision(pos_label=0), sk.precision_score, {"pos_label": 0}),
            (Recall(), sk.recall_score, {}),
            (F1(), sk.f1_score, {}),
            (FBeta(0.5), sk.fbeta_score, {"beta": 0.5}),
        ]
        for metric, func, kwargs in cases:
            with self.subTest(metric=metric):
                actual = accumulate(metric, self.bin_true, self.bin_pred)
                expected = func(self.bin_true, self.bin_pred, **kwargs)
                self.assertAlmostEqual(actual, expected)
        with self.assertRaises(ValueError):
            F1()(self.y_true, self.y_pred)
        with self.assertRaises(ValueError):
            F1()(self.ml_true, self.ml_pred)

    def test_multilabel(self):
        for average in ["micro", "macro", "weighted", None]:
            with self.subTest(average=average):
                actual = accumulate(
                    Recall(average=average), self.ml_true, self.ml_pred
                )
                expected = sk.recall_score(
                    self.ml_true, self.ml_pred, average=average
                )
                np.testing.assert_allclose(actual, expected)

    def test_accuracy(self):
        self.assertAlmostEqual(
            accumulate(Accuracy(), self.y_true, self.y_pred),
            sk.accuracy_score(self.y_true, self.y_pred),
        )
        self.assertAlmostEqual(
            accumulate(Accuracy(), self.ml_true, self.ml_pred),
            sk.accuracy_score(self.ml_true, self.ml_pred),
        )

    def test_balanced_accuracy(self):
        for adjusted in [False, True]:
            with self.subTest(adjusted=adjusted):
                actual = accumulate(
                    BalancedAccuracy(adjusted=adjusted),
                    self.y_true,
                    self.y_pred,
                )
                expected = sk.balanced_accuracy_score(
                    self.y_true, self.y_pred, adjusted=adjusted
                )
                self.assertAlmostEqual(actual, expected)

    def test_confusion_matrix(self):
        for normalize in [None, "true", "pred", "all"]:
            with self.subTest(normalize=normalize):
                actual = accumulate(
                    ConfusionMatrix(normalize=normalize),
                    self.y_true,
                    self.y_pred,
                )
                expected = sk.confusion_matrix(
                    self.y_true, self.y_pred, normalize=normalize
                )
                np.testing.assert_allclose(actual, expected)
        np.testing.assert_array_equal(
            accumulate(ConfusionMatrix(), self.ml_true, self.ml_pred),
            sk.multilabel_confusion_matrix(self.ml_true, self.ml_pred),
        )

    def test_torch_tensors(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        metric = F1(average="macro")
        actual = metric(torch.tensor(self.y_true), torch.tensor(self.y_pred))
        self.assertAlmostEqual(actual, metric(self.y_true, self.y_pred))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            Accuracy()([0, -1], [0, 1])
        with self.assertRaises(ValueError):
            Accuracy()([0.5, 1.0], [0, 1])
        with self.assertRaises(ValueError):
            F1(average="macro", num_classes=2)([0, 2], [0, 1])
        with self.assertRaises(ValueError):
            F1(average="macro")([0, 1], [0, 1, 1])
        with self.assertRaises(ValueError):
            F1(average="mean")
        metric = Accuracy()
        with self.assertRaises(ValueError):
            metric.compute()
        metric.update([0, 1], [0, 1])
        with self.assertRaises(ValueError):
            metric.update([[0, 1]], [[0, 1]])


class TestMetric(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.y_true = rng.integers(0, 4, 100)
        self.y_pred = rng.integers(0, 4, 100)

    def test_merge(self):
        # the first half has fewer classes, so the matrices differ in size
        first, second = F1(average="macro"), F1(average="macro")
        first.update(self.y_true[:50] % 2, self.y_pred[:50] % 2)
        second.update(self.y_true[50:], self.y_pred[50:])
        first.merge(second)
        y_true = np.r_[self.y_true[:50] % 2, self.y_true[50:]]
        y_pred = np.r_[self.y_pred[:50] % 2, self.y_pred[50:]]
        self.assertAlmostEqual(
            first.compute(), sk.f1_score(y_true, y_pred, average="macro")
        )
        with self.assertRaises(TypeError):
            first.merge(Precision(average="macro"))

    def test_call_leaves_state_untouched(self):
        metric = Accuracy()
        metric.update(self.y_true, self.y_pred)
        before = metric.compute()
        self.assertEqual(metric([0, 1], [0, 1]), 1.0)
        self.assertEqual(metric.compute(), before)

    def test_reset(self):
        metric = Accuracy()
        metric.update(self.y_true, self.y_pred)
        metric.reset()
        metric.update([0, 1], [0, 1])
        self.assertEqual(metric.compute(), 1.0)

    def test_repr_shows_parameters_only(self):
        metric = F1(average="macro")
        metric.update(self.y_true, self.y_pred)
        self.assertIn("average='macro'", repr(metric))
        self.assertNotIn("confmat", repr(metric))


class TestMetricCollection(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.y_true = rng.integers(0, 3, 120)
        self.y_score = rng.dirichlet(np.ones(3), 120)
        self.y_pred = self.y_score.argmax(axis=1)

    def _collection(self):
        return MetricCollection({
            "f1": F1(average="macro"),
            "auc": AUROC(),
            "acc": Accuracy(),
        })

    def test_routes_inputs_by_name(self):
        metrics = self._collection()
        for i in range(0, 120, 50):
            metrics.update(
                y_true=self.y_true[i : i + 50],
                y_pred=self.y_pred[i : i + 50],
                y_score=self.y_score[i : i + 50],
                unused=None,
            )
        results = metrics.compute()
        self.assertEqual(list(results), ["f1", "auc", "acc"])
        self.assertTrue(all(isinstance(v, float) for v in results.values()))
        self.assertAlmostEqual(
            results["auc"],
            sk.roc_auc_score(self.y_true, self.y_score, multi_class="ovr"),
        )
        self.assertAlmostEqual(
            results["f1"],
            sk.f1_score(self.y_true, self.y_pred, average="macro"),
        )

    def test_missing_input(self):
        with self.assertRaisesRegex(KeyError, "y_score"):
            self._collection().update(y_true=self.y_true, y_pred=self.y_pred)

    def test_merge_and_reset(self):
        first, second = self._collection(), self._collection()
        first.update(
            y_true=self.y_true[:60],
            y_pred=self.y_pred[:60],
            y_score=self.y_score[:60],
        )
        second.update(
            y_true=self.y_true[60:],
            y_pred=self.y_pred[60:],
            y_score=self.y_score[60:],
        )
        first.merge(second)
        whole = self._collection()
        whole.update(
            y_true=self.y_true, y_pred=self.y_pred, y_score=self.y_score
        )
        self.assertEqual(first.compute(), whole.compute())
        first.reset()
        with self.assertRaises(ValueError):
            first.compute()

    def test_rejects_non_metrics(self):
        with self.assertRaises(TypeError):
            MetricCollection({"f1": sk.f1_score})


class TestRanking(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        n = 300
        self.bin_true = rng.integers(0, 2, n)
        self.bin_score = np.clip(
            0.3 * self.bin_true + rng.normal(0.35, 0.2, n), 0, 1
        )
        self.y_true = rng.integers(0, 4, n)
        logits = rng.normal(size=(n, 4))
        logits[np.arange(n), self.y_true] += 1.0
        self.y_score = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        self.ml_true = rng.integers(0, 2, (n, 3))
        self.ml_score = rng.random((n, 3)) * 0.5 + 0.4 * self.ml_true

    def test_binary_exact(self):
        y, s = self.bin_true, self.bin_score
        self.assertAlmostEqual(
            accumulate(AUROC(), y, s), sk.roc_auc_score(y, s)
        )
        self.assertAlmostEqual(
            accumulate(AveragePrecision(), y, s),
            sk.average_precision_score(y, s),
        )
        for actual, expected in zip(
            accumulate(ROCCurve(), y, s), sk.roc_curve(y, s)
        ):
            np.testing.assert_allclose(actual, expected)
        for actual, expected in zip(
            accumulate(PrecisionRecallCurve(), y, s),
            sk.precision_recall_curve(y, s),
        ):
            np.testing.assert_allclose(actual, expected)

    def test_optimal_threshold(self):
        y, s = self.bin_true, self.bin_score
        fpr, tpr, thresholds = sk.roc_curve(y, s)
        self.assertEqual(
            accumulate(OptimalThreshold(), y, s),
            thresholds[np.argmax(tpr - fpr)],
        )
        precision, recall, thresholds = sk.precision_recall_curve(y, s)
        f1 = 2 * precision * recall / (precision + recall)
        self.assertEqual(
            accumulate(OptimalThreshold(criterion="f1"), y, s),
            thresholds[np.nanargmax(f1[:-1])],
        )

    def test_multiclass_exact(self):
        y, s = self.y_true, self.y_score
        onehot = label_binarize(y, classes=range(4))
        for average in ["macro", "weighted"]:
            with self.subTest(average=average):
                self.assertAlmostEqual(
                    accumulate(AUROC(average=average), y, s),
                    sk.roc_auc_score(y, s, multi_class="ovr", average=average),
                )
        for average in ["micro", "macro", "weighted", None]:
            with self.subTest(average=average):
                np.testing.assert_allclose(
                    accumulate(AUROC(average=average), y, s),
                    sk.roc_auc_score(onehot, s, average=average),
                )
                np.testing.assert_allclose(
                    accumulate(AveragePrecision(average=average), y, s),
                    sk.average_precision_score(onehot, s, average=average),
                )

    def test_multilabel_exact(self):
        y, s = self.ml_true, self.ml_score
        for average in ["micro", "macro", "weighted", None]:
            with self.subTest(average=average):
                np.testing.assert_allclose(
                    accumulate(AUROC(average=average), y, s),
                    sk.roc_auc_score(y, s, average=average),
                )
                np.testing.assert_allclose(
                    accumulate(AveragePrecision(average=average), y, s),
                    sk.average_precision_score(y, s, average=average),
                )

    def test_class_without_positives_is_skipped(self):
        y = np.where(self.y_true == 3, 0, self.y_true)
        per_class = AUROC(average=None)(y, self.y_score)
        self.assertTrue(np.isnan(per_class[3]))
        self.assertAlmostEqual(
            AUROC()(y, self.y_score), np.mean(per_class[:3])
        )

    def test_binned_approximates_exact(self):
        y, s = self.bin_true, self.bin_score
        cases = [
            (AUROC, {}),
            (AveragePrecision, {}),
            (OptimalThreshold, {}),
            (AUROC, {"average": "macro"}),
        ]
        for cls, kwargs in cases:
            with self.subTest(metric=cls.__name__):
                exact = accumulate(cls(**kwargs), y, s)
                binned = accumulate(cls(thresholds=1001, **kwargs), y, s)
                self.assertAlmostEqual(binned, exact, delta=0.01)
        exact = accumulate(AUROC(), self.y_true, self.y_score)
        binned = accumulate(AUROC(thresholds=1001), self.y_true, self.y_score)
        self.assertAlmostEqual(binned, exact, delta=0.01)

    def test_binned_merge(self):
        first, second = AUROC(thresholds=50), AUROC(thresholds=50)
        first.update(self.bin_true[:100], self.bin_score[:100])
        second.update(self.bin_true[100:], self.bin_score[100:])
        first.merge(second)
        whole = AUROC(thresholds=50)
        whole.update(self.bin_true, self.bin_score)
        self.assertEqual(first.compute(), whole.compute())

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            AUROC(thresholds=10)([0, 1], [0.5, 1.5])
        with self.assertRaises(ValueError):
            ROCCurve()(self.y_true, self.y_score)
        with self.assertRaises(ValueError):
            AUROC()([0, 4], self.y_score[:2])
        metric = AUROC()
        metric.update(self.bin_true, self.bin_score)
        with self.assertRaises(ValueError):
            metric.update(self.y_true, self.y_score)


class TestRegression(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.y_true = rng.normal(size=(150, 3))
        self.y_pred = self.y_true + rng.normal(scale=0.5, size=(150, 3))

    def test_errors_and_r2(self):
        cases = [
            (MeanSquaredError, sk.mean_squared_error),
            (RootMeanSquaredError, sk.root_mean_squared_error),
            (MeanAbsoluteError, sk.mean_absolute_error),
            (R2Score, sk.r2_score),
        ]
        for cls, func in cases:
            for multioutput in ["uniform_average", "raw_values"]:
                for y_true, y_pred in [
                    (self.y_true[:, 0], self.y_pred[:, 0]),
                    (self.y_true, self.y_pred),
                ]:
                    with self.subTest(
                        metric=cls.__name__,
                        multioutput=multioutput,
                        ndim=y_true.ndim,
                    ):
                        actual = accumulate(
                            cls(multioutput=multioutput), y_true, y_pred
                        )
                        expected = func(
                            y_true, y_pred, multioutput=multioutput
                        )
                        np.testing.assert_allclose(actual, expected)

    def test_r2_constant_target(self):
        y = np.full(10, 2.0)
        self.assertEqual(R2Score()(y, y), sk.r2_score(y, y))
        self.assertEqual(R2Score()(y, y + 1), sk.r2_score(y, y + 1))

    def test_correlations(self):
        y_pred = np.round(self.y_pred, 1)  # ties for Spearman
        for i in range(3):
            with self.subTest(column=i):
                self.assertAlmostEqual(
                    accumulate(
                        PearsonCorrelation(), self.y_true[:, i], y_pred[:, i]
                    ),
                    stats.pearsonr(self.y_true[:, i], y_pred[:, i])[0],
                )
                self.assertAlmostEqual(
                    accumulate(
                        SpearmanCorrelation(), self.y_true[:, i], y_pred[:, i]
                    ),
                    stats.spearmanr(self.y_true[:, i], y_pred[:, i])[0],
                )
        np.testing.assert_allclose(
            accumulate(
                PearsonCorrelation(multioutput="raw_values"),
                self.y_true,
                y_pred,
            ),
            [
                stats.pearsonr(self.y_true[:, i], y_pred[:, i])[0]
                for i in range(3)
            ],
        )

    def test_pearson_is_stable_with_large_offsets(self):
        y_true = 1e9 + self.y_true[:, 0]
        y_pred = 1e9 + self.y_pred[:, 0]
        self.assertAlmostEqual(
            accumulate(PearsonCorrelation(), y_true, y_pred, batch_size=7),
            stats.pearsonr(self.y_true[:, 0], self.y_pred[:, 0])[0],
            places=6,
        )

    def test_merge(self):
        for cls in [R2Score, PearsonCorrelation, SpearmanCorrelation]:
            with self.subTest(metric=cls.__name__):
                first, second = cls(), cls()
                first.update(self.y_true[:40], self.y_pred[:40])
                second.update(self.y_true[40:], self.y_pred[40:])
                first.merge(second)
                self.assertAlmostEqual(
                    first.compute(), cls()(self.y_true, self.y_pred)
                )

    def test_invalid_inputs(self):
        metric = MeanSquaredError()
        metric.update(self.y_true, self.y_pred)
        with self.assertRaises(ValueError):
            metric.update(self.y_true[:, :2], self.y_pred[:, :2])
        with self.assertRaises(ValueError):
            MeanSquaredError()([1.0, 2.0], [1.0])
        with self.assertRaises(ValueError):
            MeanSquaredError().compute()


class TestSpans(unittest.TestCase):
    # the example from seqeval's README
    y_true = [
        ["O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O"],
        ["B-PER", "I-PER", "O"],
    ]
    y_pred = [
        ["O", "O", "B-MISC", "I-MISC", "I-MISC", "I-MISC", "O"],
        ["B-PER", "I-PER", "O"],
    ]

    def test_get_spans(self):
        cases = [
            (["B-PER", "I-PER", "O", "B-LOC"], [("PER", 0, 1), ("LOC", 3, 3)]),
            # I- after O or another type starts a span
            (["O", "I-PER", "I-LOC"], [("PER", 1, 1), ("LOC", 2, 2)]),
            (["B-PER", "B-PER"], [("PER", 0, 0), ("PER", 1, 1)]),
            # IOBES
            (["S-PER", "B-LOC", "E-LOC", "O"], [("PER", 0, 0), ("LOC", 1, 2)]),
            (["O", "O"], []),
        ]
        for tags, expected in cases:
            with self.subTest(tags=tags):
                self.assertEqual(get_spans(tags), expected)
        self.assertEqual(
            get_spans(["PER-B", "PER-I", "O"], suffix=True), [("PER", 0, 1)]
        )

    def test_scores(self):
        self.assertEqual(SpanF1()(self.y_true, self.y_pred), 0.5)
        self.assertEqual(SpanRecall()(self.y_true, self.y_pred), 0.5)
        self.assertEqual(
            SpanPrecision(average=None)(self.y_true, self.y_pred),
            {"MISC": 0.0, "PER": 1.0},
        )
        self.assertEqual(
            SpanF1(average="macro")(self.y_true, self.y_pred), 0.5
        )

    def test_batches_and_merge(self):
        first, second = SpanF1(), SpanF1()
        first.update(self.y_true[:1], self.y_pred[:1])
        second.update(self.y_true[1:], self.y_pred[1:])
        first.merge(second)
        self.assertEqual(first.compute(), 0.5)

    def test_invalid_inputs(self):
        with self.assertRaises(TypeError):
            SpanF1()(["B-PER", "O"], ["B-PER", "O"])
        with self.assertRaises(ValueError):
            SpanF1()([["B-PER", "O"]], [["B-PER"]])
        with self.assertRaises(ValueError):
            SpanF1().compute()


if __name__ == "__main__":
    unittest.main()
