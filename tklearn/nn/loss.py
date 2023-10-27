from typing import Union, Callable
from collections.abc import Mapping

import torch
from torch import Tensor
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers.utils import ModelOutput


__all__ = [
    "LossFunction",
]


class LossFunction:
    def __init__(self) -> None:
        pass

    def __call__(self, *args, **kwargs):
        pass


class AutoLoss(LossFunction):
    def __init__(
        self,
        loss: LossFunction,
        problem_type: str,
        num_labels: int,
    ) -> None:
        super(AutoLoss, self).__init__()
        self.loss = loss
        self.problem_type = problem_type
        self.num_labels = num_labels

    def __call__(
        self,
        output: Union[Tensor, ModelOutput],
        target: Tensor,
    ):
        if not hasattr(self, "_criterion"):
            self._criterion = self.build_criterion()
        criterion = self._criterion
        logits_based_criterion = True
        if (
            logits_based_criterion
            and isinstance(output, Mapping)
            and "logits" in output
        ):
            # huggingface pretrained model output
            output = output["logits"]
        if target is None:
            return None
        if self.num_labels is None:
            self.num_labels = output.shape[-1]
        loss = None
        if self.problem_type == "regression":
            if self.num_labels == 1:
                loss = criterion(output.squeeze(), target.squeeze())
            else:
                loss = criterion(output, target)
        elif self.problem_type == "single_label_classification":
            loss = criterion(output.view(-1, self.num_labels), target.view(-1))
        elif self.problem_type == "multi_label_classification":
            input_shape = output.size()
            loss = criterion(output, target.view(*input_shape))
        elif self.problem_type == "masked_language_modeling":
            loss = criterion(output.view(-1, self.num_labels), target.view(-1))
        else:
            loss = criterion(output, target)
        return loss

    def build_criterion(self) -> Callable:
        if isinstance(self.loss, str):
            criterion_cls = getattr(torch.nn, self.loss)
            criterion = criterion_cls()
        elif callable(self.loss):
            criterion = self.loss
        elif self.loss is None:
            criterion = self.default_criterion()
        else:
            raise ValueError(f"unsupported loss: {self.loss}")
        return criterion

    def default_criterion(self):
        problem_type = self.problem_type
        if problem_type == "regression":
            return MSELoss()
        elif problem_type == "single_label_classification":
            return CrossEntropyLoss()
        elif problem_type == "masked_language_modeling":
            return CrossEntropyLoss()
        elif problem_type == "multi_label_classification":
            return BCEWithLogitsLoss()
        else:
            raise ValueError(f"invalid problem type: {problem_type}")
