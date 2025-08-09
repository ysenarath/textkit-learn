from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    List,
    Protocol,
    runtime_checkable,
)

import torch
from nightjar import AutoModule, BaseConfig, BaseModule

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
else:
    PreTrainedModel = Any
    PreTrainedTokenizerBase = Any


class BackboneConfig(BaseConfig, dispatch=["type"]):
    type: ClassVar[str]


class AutoBackbone(AutoModule):
    def __new__(cls, config: BackboneConfig) -> Backbone:
        approach = super().__new__(cls, config)
        if not isinstance(approach, Backbone):
            msg = (
                f"expected {Backbone.__name__}, "
                f"got {approach.__class__.__name__}"
            )
            raise TypeError(msg)
        return approach


@runtime_checkable
class Tokenizer(Protocol):
    def tokenize(self, text: str | List[str], **kwargs) -> Any:
        raise NotImplementedError


class Backbone(BaseModule, torch.nn.Module):
    config: BackboneConfig
    model: torch.nn.Module | PreTrainedModel
    tokenizer: Tokenizer | PreTrainedTokenizerBase

    @property
    def hidden_size(self) -> int:
        raise NotImplementedError

    def forward(self, batch: Any) -> Any:
        raise NotImplementedError
