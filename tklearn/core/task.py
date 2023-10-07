from __future__ import annotations
from typing import Optional, Dict


class Task(object):
    def __init__(
        self,
        *,
        problem_type: Optional[str] = None,
        label2id: Optional[Dict] = None,
        id2label: Optional[Dict] = None,
        num_labels: Optional[int] = None,
    ) -> None:
        self.label2id = label2id
        self.id2label = id2label
        self.num_labels = num_labels
        self.problem_type = problem_type

    @property
    def problem_type(self) -> Optional[str]:
        return self._problem_type

    @problem_type.setter
    def problem_type(self, problem_type: Optional[str]):
        if problem_type is None:
            if self.num_labels is None:
                return None
            if self.num_labels == 1:
                problem_type = "regression"
            elif self.num_labels > 1:
                problem_type = "single_label_classification"
        if problem_type not in [
            "regression",
            "single_label_classification",
            "multi_label_classification",
            "masked_language_modeling",
        ]:
            raise ValueError(
                f"invalid problem type: {problem_type}, "
                'expected one of ("regression", '
                '"single_label_classification", '
                '"multi_label_classification", '
                '"masked_language_modeling")'
            )
        self._problem_type = problem_type

    @property
    def label2id(self) -> Optional[Dict]:
        return self._label2id

    @label2id.setter
    def label2id(self, label2id):
        for _, idx in label2id.items():
            if isinstance(idx, str):
                idx = int(idx)
        self._label2id = label2id

    @property
    def id2label(self) -> Optional[Dict]:
        return self._id2label

    @id2label.setter
    def id2label(self, id2label):
        if id2label is None:
            if self.label2id is None:
                return None
            return {idx: label for label, idx in self.label2id.items()}
        for idx, label in id2label.items():
            if isinstance(idx, str):
                raise ValueError("invalid id2label, id must be int")
            if label not in self.label2id:
                raise ValueError("invalid id2label, label not in label2id")
            elif self.label2id and self.label2id[label] != idx:
                raise ValueError("invalid id2label, label mismatch")
        self._id2label = id2label

    @property
    def num_labels(self) -> Optional[int]:
        return self._num_labels

    @num_labels.setter
    def num_labels(self, num_labels):
        if num_labels is None:
            if self.label2id is None:
                return None
            return len(self.label2id)
        if self.label2id and len(self.label2id) != num_labels:
            raise ValueError(
                f"invalid num_labels: {num_labels}, " "len(label2id) != num_labels"
            )
        self._num_labels = num_labels
