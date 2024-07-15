from functools import singledispatch
from typing import Tuple, Union

import torch
from transformers import (
    BertConfig,
    DistilBertConfig,
    RobertaConfig,
)

from tklearn.utils.targets import (
    BinaryTargetType,
    ContinuousMultioutputTargetType,
    ContinuousTargetType,
    MulticlassMultioutputTargetType,
    MulticlassTargetType,
    MultilabelIndicatorTargetType,
    type_of_target,
)

CT = Union[BertConfig, DistilBertConfig, RobertaConfig]

__all__ = [
    "preprocess_input",
    "preprocess_target",
]


def is_one_hot(arr):
    if not torch.is_tensor(arr):
        arr = torch.tensor(arr)
    is_binary = torch.all((arr == 0) | (arr == 1)).item()
    if len(arr.shape) == 1:
        return is_binary
    return is_binary and ((torch.sum(arr, dim=-1) == 1).all().item())


@singledispatch
def preprocess_input(arg, input: torch.Tensor, num_labels: int) -> torch.Tensor:
    msg = f"unsupported target type: {arg.__class__.__name__}"
    raise NotImplementedError(msg)


@preprocess_input.register(str)
def _(arg, input: torch.Tensor, num_labels: int) -> torch.Tensor:
    return preprocess_input(type_of_target(arg), input, num_labels)


@preprocess_input.register(ContinuousTargetType)
def _(arg, input: torch.Tensor, num_labels: int) -> torch.Tensor:
    # torch.float32, [min, max], 1D
    if len(input.shape) != 1:
        msg = "y_true shape should be 1-dimensional"
        raise ValueError(msg)
    return input


@preprocess_input.register(ContinuousMultioutputTargetType)
def _(arg, input: torch.Tensor, num_labels: int) -> torch.Tensor:
    # torch.float32, [min, max], 2D
    if len(input.shape) != 2:
        msg = "y_true shape should be 2-dimensional"
        raise ValueError(msg)
    return input


@preprocess_input.register(BinaryTargetType)
def _(arg, input: torch.Tensor, num_labels: int) -> torch.Tensor:
    # torch.float32, {0, 1}, 1D
    if not is_one_hot(input):
        msg = "y_true should be one-hot encoded"
        raise ValueError(msg)
    if len(input.shape) == 1:
        return input
    if len(input.shape) != 2:
        msg = "y_true shape should be 1-dimensional or (one-hot encoded) 2-dimensional tensor"
        raise ValueError(msg)
    if input.shape[1] != 2:
        msg = "y_true shape should have exactly 2 columns"
        raise ValueError(msg)
    # let's revert it back to the class index
    return input[:, 1]  # type: torch.LongTensor


@preprocess_input.register(MulticlassTargetType)
def _(arg, input: torch.Tensor, num_labels: int) -> torch.Tensor:
    # torch.long, {0, 1, ..., C-1}, 1D
    if len(input.shape) == 1:
        return input
    if len(input.shape) != 2:  # one-hot encoded
        msg = "y_true shape should be 1-dimensional or (one-hot encoded) 2-dimensional tensor"
        raise ValueError(msg)
    if input.shape[1] != num_labels:  # every row is a one-hot encoded vector
        msg = f"y_true shape {input.shape} does not match num_labels: {num_labels}"
        raise ValueError(msg)
    if not is_one_hot(input):
        msg = "y_true should be one-hot encoded"
        raise ValueError(msg)
    # let's revert it back to the class index
    return torch.argmax(input, dim=-1)  # type: torch.LongTensor


@preprocess_input.register(MulticlassMultioutputTargetType)
def _(arg, input: torch.Tensor, num_labels: int) -> torch.Tensor:
    # torch.long, {0, 1, ..., C-1}, 2D
    if len(input.shape) != 2:
        msg = "y_true shape should be 2-dimensional"
        raise ValueError(msg)
    return input


@preprocess_input.register(MultilabelIndicatorTargetType)
def _(arg, input: torch.Tensor, num_labels: int) -> torch.Tensor:
    # torch.float32, {0, 1}, 2D
    if len(input.shape) != 2:
        msg = "y_true shape should be 2-dimensional"
        raise ValueError(msg)
    # make sure it matches the number of labels
    if input.shape[1] != num_labels:
        msg = f"y_true shape {input.shape} does not match num_labels: {num_labels}"
        raise ValueError(msg)
    # make sure that the values are only 0 or 1
    if not torch.all((input == 0) | (input == 1)):
        raise ValueError("y_true should contain only 0 and 1")
    return input


@singledispatch
def preprocess_target(
    arg, logits: torch.Tensor, threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    msg = f"unsupported target type: {arg.__class__.__name__}"
    raise NotImplementedError(msg)


@preprocess_target.register(str)
def _(
    arg, logits: torch.Tensor, threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    return preprocess_target(type_of_target(arg), logits, threshold)


@preprocess_target.register(ContinuousTargetType)
def _(
    arg, logits: torch.Tensor, threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    # torch.float32, [min, max], 1D
    y_pred = logits.squeeze()  # convert 2D (batch_size, 1) to 1D (batch_size,)
    if len(y_pred.shape) != 1:
        msg = f"expected 1D tensor, got {len(y_pred.shape)}D tensor"
        raise ValueError(msg)
    return y_pred, y_pred


@preprocess_target.register(ContinuousMultioutputTargetType)
def _(
    arg, logits: torch.Tensor, threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    # torch.float32, [min, max], 2D
    if len(logits.shape) != 2:
        msg = f"expected 2D tensor, got {len(logits.shape)}D tensor"
        raise ValueError(msg)
    return logits, logits


@preprocess_target.register(BinaryTargetType)
def _(
    arg, logits: torch.Tensor, threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    # torch.float32, {0, 1}, 1D
    y_score = torch.sigmoid(logits)
    if len(y_score.shape) == 2:
        if y_score.shape[1] == 1:
            # convert 2D (batch_size, 1) to 1D (batch_size,)
            y_score = y_score.squeeze()
        elif y_score.shape[1] == 2:
            # convert 2D (batch_size, 2) to 1D (batch_size,)
            # column 0 is for class 0, column 1 is for class 1
            y_score = y_score[:, 1]
        else:
            msg = f"y_score shape {y_score.shape} not supported"
            raise ValueError(msg)
    if len(y_score.shape) != 1:
        msg = f"expected 1D or 2D tensor, got {len(y_score.shape)}D tensor"
        raise ValueError(msg)
    y_pred = (y_score > threshold).long()
    return y_pred, y_score


@preprocess_target.register(MulticlassTargetType)
def _(
    arg, logits: torch.Tensor, threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    # torch.float32, {0, 1, ..., C-1}, 1D
    if len(logits.shape) != 2:
        msg = f"expected 2D tensor, got {len(logits.shape)}D tensor"
        raise ValueError(msg)
    y_score = torch.softmax(logits, dim=-1)
    y_pred = torch.argmax(logits, dim=-1)
    return y_pred, y_score


@preprocess_target.register(MultilabelIndicatorTargetType)
def _(
    arg, logits: torch.Tensor, threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    # torch.float32, {0, 1}, 2D
    if len(logits.shape) != 2:
        msg = f"expected 2D tensor, got {len(logits.shape)}D tensor"
        raise ValueError(msg)
    y_score = torch.sigmoid(logits)
    y_pred = (y_score > threshold).long()
    return y_pred, y_score


@preprocess_target.register(MulticlassMultioutputTargetType)
def _(
    arg, logits: Tuple[torch.Tensor, ...], threshold: float = 0.5
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
    # torch.long, {0, 1, ..., C-1}, 2D
    # logits is a tuple of tensors
    if not isinstance(logits, (list, tuple)):
        msg = f"expected list or tuple, got {logits.__class__.__name__}"
        raise ValueError(msg)
    y_score = tuple(torch.softmax(logit, dim=-1) for logit in logits)
    y_pred = tuple(torch.argmax(logit, dim=-1) for logit in logits)
    return y_pred, y_score
