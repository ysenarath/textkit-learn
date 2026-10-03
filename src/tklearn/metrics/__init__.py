from tklearn.metrics.base import Metric, MetricCollection
from tklearn.metrics.classification import (
    F1,
    Accuracy,
    BalancedAccuracy,
    ConfusionMatrix,
    FBeta,
    Precision,
    Recall,
)
from tklearn.metrics.ranking import (
    AUROC,
    AveragePrecision,
    OptimalThreshold,
    PrecisionRecallCurve,
    PRPoints,
    ROCCurve,
    ROCPoints,
)
from tklearn.metrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    PearsonCorrelation,
    R2Score,
    RootMeanSquaredError,
    SpearmanCorrelation,
)
from tklearn.metrics.spans import SpanF1, SpanPrecision, SpanRecall, get_spans

__all__ = [
    # --- Base ---
    "Metric",
    "MetricCollection",
    # --- Classification ---
    "Accuracy",
    "BalancedAccuracy",
    "ConfusionMatrix",
    "F1",
    "FBeta",
    "Precision",
    "Recall",
    # --- Ranking ---
    "AUROC",
    "AveragePrecision",
    "OptimalThreshold",
    "PRPoints",
    "PrecisionRecallCurve",
    "ROCCurve",
    "ROCPoints",
    # --- Regression ---
    "MeanAbsoluteError",
    "MeanSquaredError",
    "PearsonCorrelation",
    "R2Score",
    "RootMeanSquaredError",
    "SpearmanCorrelation",
    # --- Spans ---
    "SpanF1",
    "SpanPrecision",
    "SpanRecall",
    "get_spans",
]
