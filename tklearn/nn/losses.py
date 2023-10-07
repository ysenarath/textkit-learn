from __future__ import annotations
import abc

from torch import Tensor
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import PreTrainedModel

from tklearn.core.task import Task

__all__ = [
    "LossFunction",
    "AutoLoss",
]


class LossFunction(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        logits: Tensor,
        target: Tensor,
    ):
        raise NotImplementedError()


class AutoLoss(LossFunction):
    def __init__(
        self,
        task: Task,
    ) -> None:
        super(AutoLoss, self).__init__()
        self.task = task

    @property
    def task(self) -> Task:
        return self._task

    @task.setter
    def task(self, task: Task):
        if not isinstance(task, Task):
            raise TypeError(f"invalid type: {type(task).__name__}")
        self._task = task
        # reset loss function
        self._loss_fct = None
        self._num_labels = None

    def loss_fct(self, logits: Tensor, target: Tensor):
        if self._loss_fct is None:
            if self.task.problem_type == "regression":
                loss_fct = MSELoss()
            elif self.task.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
            elif self.task.problem_type == "masked_language_modeling":
                loss_fct = CrossEntropyLoss()
            elif self.task.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
            else:
                raise ValueError(f"invalid problem type: {self.task.problem_type}")
            self._loss_fct = loss_fct
        return self._loss_fct(logits, target)

    def __call__(
        self,
        logits: Tensor,
        target: Tensor,
    ):
        if target is None:
            return None
        if self._num_labels is None and self.task.num_labels is not None:
            self._num_labels = self.task.num_labels
        else:
            self._num_labels = logits.shape[-1]
        loss = None
        if self.task.problem_type == "regression":
            if self._num_labels == 1:
                loss = self.loss_fct(logits.squeeze(), target.squeeze())
            else:
                loss = self.loss_fct(logits, target)
        elif self.task.problem_type == "single_label_classification":
            loss = self.loss_fct(logits.view(-1, self._num_labels), target.view(-1))
        elif self.task.problem_type == "multi_label_classification":
            input_shape = logits.size()
            loss = self.loss_fct(logits, target.view(*input_shape))
        elif self.task.problem_type == "masked_language_modeling":
            loss = self.loss_fct(logits.view(-1, self._num_labels), target.view(-1))
        return loss

    @classmethod
    def from_pretrained(cls, model: PreTrainedModel):
        try:
            label2id = model.config.label2id
        except (AttributeError, KeyError):
            label2id = None
        try:
            id2label = model.config.id2label
        except (AttributeError, KeyError):
            id2label = None
        try:
            num_labels = model.config.num_labels
        except (AttributeError, KeyError):
            num_labels = None
        try:
            problem_type = model.config.problem_type
        except (AttributeError, KeyError):
            problem_type = None
        task = Task(
            num_labels=num_labels,
            problem_type=problem_type,
            label2id=label2id,
            id2label=id2label,
        )
        return cls(task)
