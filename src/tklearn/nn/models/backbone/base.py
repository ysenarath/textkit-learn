from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    Union,
    runtime_checkable,
)

import torch

from tklearn.utils.registry import Registry

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
else:
    PreTrainedModel = Any
    PreTrainedTokenizerBase = Any

__all__ = [
    "BACKBONES",
    "Backbone",
    "Tokenizer",
]


@runtime_checkable
class Tokenizer(Protocol):
    def tokenize(self, text: str | list[str], **kwargs) -> Any: ...


class Backbone(torch.nn.Module):
    """An encoder that maps a batch to outputs with a ``pooler_output``.

    Subclasses set `model` and `tokenizer` and implement `hidden_size` and
    `forward`.
    """

    model: Union[torch.nn.Module, PreTrainedModel]
    tokenizer: Union[Tokenizer, PreTrainedTokenizerBase, None] = None

    @property
    def hidden_size(self) -> int:
        raise NotImplementedError

    def forward(self, batch: Any) -> Any:
        raise NotImplementedError


#: Backbones by name, e.g. ``BACKBONES.create("transformer", "bert-base-uncased")``.
BACKBONES: Registry[Backbone] = Registry("backbone")
