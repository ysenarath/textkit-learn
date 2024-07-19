from typing import Dict

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from tklearn.utils.array import to_numpy

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
    y_score : array-like
        Predicted probabilities.
    subgroup : array-like
        Group labels (dtype=int).

    Returns
    -------
    dict
        Dictionary of group labels to their bias metrics.
    """
    y_true, y_score, subgroup = to_numpy(y_true, y_score, subgroup)
    _validate_numpy(y_true, dims=1, name="y_true")
    _validate_numpy(y_score, dims=1, name="y_score")
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
        # BPSN and BNSP AUC
        # If there are no background examples, the following AUCs are set to 1.0
        #   this indicates that the subgroup is perfectly separated from the
        #   background (since there is no background to compare against)
        bpsn_auc, bnsp_auc = 1.0, 1.0
        # 0.0 is the default value for AEG metrics
        pos_aeg, neg_aeg = 0.0, 0.0
        if not np.all(group_mask):  # check if there is background to compare against
            # Background Positive, Subgroup Negative (BPSN) AUC
            bpsn_labels = np.concatenate([
                np.zeros(len(group_neg)),
                np.ones(len(bg_pos)),
            ])
            bpsn_preds = np.concatenate([group_neg, bg_pos])
            bpsn_auc = roc_auc_score(bpsn_labels, bpsn_preds)
            # Background Negative, Subgroup Positive (BNSP) AUC
            bnsp_labels = np.concatenate([
                np.zeros(len(bg_neg)),
                np.ones(len(group_pos)),
            ])
            bnsp_preds = np.concatenate([bg_neg, group_pos])
            bnsp_auc = roc_auc_score(bnsp_labels, bnsp_preds)
            # Positive Average Equality Gap
            # Average Equality Gap as a metric can range in value from -0.5 to 0.5
            #   At each of these extremes, it represents a different type of bias
            #   where the TPR of the subgroup is consistently higher or lower,
            #   respectively, than that of the background.
            # max value of mannwhitneyu is len(group_pos) * len(bg_pos)
            # note: 0.5 - np.mean(group_pos[:, np.newaxis] > bg_pos) is the same as
            pos_aeg = 0.5 - mannwhitneyu(group_pos, bg_pos).statistic / (
                len(group_pos) * len(bg_pos)
            )
            # Negative Average Equality Gap
            # note: 0.5 - np.mean(group_neg[:, np.newaxis] > bg_neg) is the same as
            neg_aeg = 0.5 - mannwhitneyu(group_neg, bg_neg).statistic / (
                len(group_neg) * len(bg_neg)
            )
        results[g] = {
            "Subgroup AUC": subgroup_auc,
            "BPSN AUC": bpsn_auc,
            "BNSP AUC": bnsp_auc,
            "Positive AEG": pos_aeg,
            "Negative AEG": neg_aeg,
        }
    return results
