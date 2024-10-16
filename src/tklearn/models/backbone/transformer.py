from __future__ import annotations

from typing import ClassVar

from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling

from tklearn.models.backbone.base import Backbone, BackboneConfig

TRANSFORMERS_INPUTS = {
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "position_ids",
    "head_mask",
}


class TransformerConfig(BackboneConfig):
    type: ClassVar[str] = "transformer"
    model_name_or_path: str = "bert-base-uncased"


class Transformer(Backbone):
    config: TransformerConfig

    def __post_init__(self) -> None:
        self.model = AutoModel.from_pretrained(self.config.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path
        )

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def forward(self, batch: dict) -> BaseModelOutputWithPooling:
        kwargs = {}
        for k, v in batch.items():
            if k not in TRANSFORMERS_INPUTS:
                continue
            kwargs[k] = v
        return self.model(**kwargs)
