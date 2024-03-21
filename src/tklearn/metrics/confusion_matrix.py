from typing import (
    Literal,
    Optional,
    Union,
    overload,
)

import numpy as np
import numpy.typing as nt
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from sklearn.utils.multiclass import unique_labels
from typing_extensions import Annotated

from tklearn.utils.target_types import (
    TargetType,
    check_targets,
    resolve_target_type,
)

ConfusionMatrixType = Annotated[nt.NDArray[np.int32], Literal["N", 2, 2]]

AverageType = Literal["macro", "micro", None]


class ConfusionMatrix:
    def __init__(
        self,
        labels: Optional[ArrayLike] = None,
        target_names: Optional[ArrayLike] = None,
        target_type: TargetType = None,
    ):
        """
        Initialize a confusion matrix.

        Note that this type is the most specific type that can be inferred.
        For example:

            * ``binary`` is more specific but compatible with ``multiclass``.
            * ``multiclass`` of integers is more specific but compatible with
            ``continuous``.
            * ``multilabel-indicator`` is more specific but compatible with
            ``multiclass-multioutput``.

        Parameters
        ----------
        labels : array-like of shape (n_classes,), default=None
            List of labels to index the matrix. This may be used to reorder or
            select a subset of labels. If ``None`` is given, those that appear
            at least once in ``y_true`` or ``y_pred`` are used in sorted order.
        target_names : array-like of shape (n_classes,), default=None
            Optional display names matching the labels (same order).
        target_type : str
            One of:

            * 'continuous': `y` is an array-like of floats that are not all
            integers, and is 1d or a column vector.
            * 'continuous-multioutput': `y` is a 2d array of floats that are
            not all integers, and both dimensions are of size > 1.
            * 'binary': `y` contains <= 2 discrete values and is 1d or a column
            vector.
            * 'multiclass': `y` contains more than two discrete values, is not
            a sequence of sequences, and is 1d or a column vector.
            * 'multiclass-multioutput': `y` is a 2d array that contains more
            than two discrete values, is not a sequence of sequences, and both
            dimensions are of size > 1.
            * 'multilabel-indicator': `y` is a label indicator matrix, an array
            of two dimensions with at least two columns, and at most 2 unique
            values.
            * 'unknown': `y` is array-like but none of the above, such as a 3d
            array, sequence of sequences, or an array of non-sequence objects.
        """
        self.labels = labels
        self.target_names = target_names
        self.target_type = target_type
        self.conf_matrix: Optional[ConfusionMatrixType] = None
        self.num_samples = 0
        self.accuracy = 0

    def update(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        sample_weight: Optional[ArrayLike] = None,
    ) -> None:
        y_type, y_true, y_pred = check_targets(y_true, y_pred)
        self.target_type = resolve_target_type(self.target_type, y_type)
        labels = unique_labels(y_true, y_pred)
        if self.labels is not None and not np.array_equal(self.labels, labels):
            labels = np.union1d(self.labels, labels)
            if self.conf_matrix is not None:
                cm = np.zeros((len(labels), 2, 2), dtype=np.int32)
                for i, label in enumerate(labels):
                    if label not in self.labels:
                        continue
                    cm[i] = self.conf_matrix[self.labels == label]
                self.conf_matrix = cm
        if self.conf_matrix is None:
            self.conf_matrix = np.zeros((len(labels), 2, 2), dtype=np.int32)
        self.conf_matrix += multilabel_confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
            sample_weight=sample_weight,
        )
        self.labels = labels
        self.accuracy += accuracy_score(y_true, y_pred, normalize=False)
        self.num_samples += (
            y_true.shape[0]
            if isinstance(y_true, (csr_matrix, np.ndarray))
            else len(y_true)
        )

    def hamming_score(self, normalize: Optional[bool] = True) -> float:
        score = np.sum(np.diagonal(self.conf_matrix, axis1=1, axis2=2))
        if normalize:
            return score / (self.num_samples * self.conf_matrix.shape[1])
        return score

    def accuracy_score(self, normalize: Optional[bool] = True) -> float:
        if normalize:
            return self.accuracy / self.num_samples
        return self.accuracy

    @overload
    def tp(self, average: Literal["macro"]) -> ArrayLike: ...

    @overload
    def tp(self, average: Literal["micro", None]) -> float: ...

    def tp(self, average: AverageType = None) -> Union[float, ArrayLike, None]:
        if average is None and self.target_type == "binary":
            return self.conf_matrix[1, 1, 1]
        elif average == "micro":
            return np.sum(self.conf_matrix[:, 1, 1])
        elif average == "macro":
            return self.conf_matrix[:, 1, 1]
        else:
            return None

    @overload
    def fp(self, average: Literal["macro"]) -> ArrayLike: ...

    @overload
    def fp(self, average: Literal["micro", None]) -> float: ...

    def fp(self, average: AverageType = None) -> Union[float, ArrayLike, None]:
        if average is None and self.target_type == "binary":
            return self.conf_matrix[0, 1, 1]
        elif average == "micro":
            return np.sum(self.conf_matrix[:, 0, 1])
        elif average == "macro":
            return self.conf_matrix[:, 0, 1]
        else:
            return None

    @overload
    def fn(self, average: Literal["macro"]) -> ArrayLike: ...

    @overload
    def fn(self, average: Literal["micro", None]) -> float: ...

    def fn(self, average: AverageType = None) -> Union[float, ArrayLike, None]:
        if average is None and self.target_type == "binary":
            return self.conf_matrix[1, 1, 0]
        elif average == "micro":
            return np.sum(self.conf_matrix[:, 1, 0])
        elif average == "macro":
            return self.conf_matrix[:, 1, 0]
        else:
            return None

    def f1_score(self, average: AverageType = None) -> float:
        # F1 = 2 * TP / (2 * TP + FN + FP)
        f1 = (
            2
            * self.tp(average)
            / (2 * self.tp(average) + self.fn(average) + self.fp(average))
        )
        if average == "macro":
            f1 = np.mean(f1)
        return f1

    def precision_score(self, average: AverageType = None) -> float:
        # precision = TP / (TP + FP)
        precision = self.tp(average) / (self.tp(average) + self.fp(average))
        if average == "macro":
            precision = np.mean(precision)
        return precision

    def recall_score(self, average: AverageType = None) -> float:
        # recall = TP / (TP + FN)
        recall = self.tp(average) / (self.tp(average) + self.fn(average))
        if average == "macro":
            recall = np.mean(recall)
        return recall
