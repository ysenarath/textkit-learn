from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score

from tklearn.nn.utils.array import to_numpy

__all__ = [
    "nuanced_bias_report",
]


def _validate_numpy(arr: np.ndarray, dims: int, name: str) -> None:
    """
    Validate the shape of a numpy array.

    Parameters
    ----------
    arr : array-like
        Array to validate.
    dims : int
        Expected number of dimensions.
    name : str
        Name of the array.
    """
    if len(arr.shape) != dims:
        msg = f"{name} must have {dims} dimensions"
        raise ValueError(msg)


def nuanced_bias_report(
    y_true: np.ndarray, y_score: np.ndarray, subgroup: np.ndarray
) -> Dict[int, Dict[str, float]]:
    """
    Calculate bias metrics for each group in a one-vs-rest scenario.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted probabilities.
    subgroup : array-like
        Group labels.

    Returns
    -------
    dict
        Dictionary of group labels to their bias metrics.
    """
    y_true, y_score, subgroup = to_numpy(y_true, y_score, subgroup)
    _validate_numpy(y_true, dims=1, name="y_true")
    _validate_numpy(y_score, dims=1, name="y_pred")
    # Ensure that the subgroup is an integer array
    if len(subgroup.shape) == 2:
        subgroup = np.argmax(subgroup, axis=1)
    elif len(subgroup.shape) != 1:
        msg = "subgroup must be a 1D or 2D array"
        raise ValueError(msg)
    subgroup = subgroup.astype(int)
    unique_groups = np.unique(subgroup)
    results = {}
    for g in unique_groups:
        # Split data into current group and background
        group_mask = subgroup == g
        group_true, group_pred = y_true[group_mask], y_score[group_mask]
        bg_true, bg_pred = y_true[~group_mask], y_score[~group_mask]
        # Separate positive and negative examples
        group_pos = group_pred[group_true == 1]
        group_neg = group_pred[group_true == 0]
        bg_pos = bg_pred[bg_true == 1]
        bg_neg = bg_pred[bg_true == 0]
        # Subgroup AUC
        subgroup_auc = roc_auc_score(group_true, group_pred)
        # Background Positive, Subgroup Negative (BPSN) AUC
        bpsn_labels = np.concatenate([np.zeros(len(group_neg)), np.ones(len(bg_pos))])
        bpsn_preds = np.concatenate([group_neg, bg_pos])
        bpsn_auc = roc_auc_score(bpsn_labels, bpsn_preds)
        # Background Negative, Subgroup Positive (BNSP) AUC
        bnsp_labels = np.concatenate([np.zeros(len(bg_neg)), np.ones(len(group_pos))])
        bnsp_preds = np.concatenate([bg_neg, group_pos])
        bnsp_auc = roc_auc_score(bnsp_labels, bnsp_preds)
        # Positive Average Equality Gap
        pos_aeg = (
            0.5 - np.mean(bg_pos[:, np.newaxis] > group_pos)
            if len(group_pos) > 0 and len(bg_pos) > 0
            else 0
        )
        # Negative Average Equality Gap
        neg_aeg = (
            0.5 - np.mean(bg_neg[:, np.newaxis] > group_neg)
            if len(group_neg) > 0 and len(bg_neg) > 0
            else 0
        )
        results[g] = {
            "Subgroup AUC": subgroup_auc,
            "BPSN AUC": bpsn_auc,
            "BNSP AUC": bnsp_auc,
            "Positive AEG": pos_aeg,
            "Negative AEG": neg_aeg,
        }
    return results


def _test_nuanced_bias_report():
    # Example usage:
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_pred = np.random.rand(1000)
    group = np.random.randint(0, 3, 1000)
    results = nuanced_bias_report(y_true, y_pred, group)
    for group, metrics in results.items():
        print(f"Metrics for group {group}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print()
