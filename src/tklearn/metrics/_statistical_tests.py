import numpy as np
from scipy import stats

from tklearn.utils.array import to_numpy

__all__ = [
    "mcnemar",
]


def mcnemar(y_true, y_pred_a, y_pred_b=None):
    """McNemar's test for paired data.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred_a : array-like
        Predicted binary labels from the first classifier.
    y_pred_b : array-like, optional
        Predicted binary labels from the second classifier. If not provided, random labels are generated.
    """
    if y_pred_b is None:
        y_pred_b = np.random.rand(len(y_true)) >= 0.5
    y_true, y_pred, y_pred_b = to_numpy((y_true, y_pred_a, y_pred_b))
    # Create contingency table
    n01 = np.sum((y_true != y_pred_b) & (y_true == y_pred))
    n10 = np.sum((y_true == y_pred_b) & (y_true != y_pred))
    # Handle the case where n01 + n10 = 0
    if n01 + n10 == 0:
        return np.nan, 1.0  # No discordant pairs, p-value = 1
    # Calculate test statistic
    statistic = (n01 - n10) ** 2 / (n01 + n10)
    # Calculate p-value
    p_value = stats.chi2.sf(statistic, df=1)
    return statistic, p_value
