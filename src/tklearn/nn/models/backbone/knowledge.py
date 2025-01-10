from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from transformers.modeling_outputs import BaseModelOutputWithPooling

import tklearn
from tklearn.etc.model import AutoKnowledgeBasedModel
from tklearn.etc.tokenizer import KnowledgeBasedTokenizer
from tklearn.nn.models.backbone.base import Backbone, BackboneConfig

TRANSFORMERS_INPUTS = {
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "position_ids",
    "head_mask",
}


class KnowledgeBasedTransformerConfig(BackboneConfig):
    type: ClassVar[str] = "knowledge-based-transformer"
    model_name_or_path: str = "bert-base-uncased"


class KnowledgeBasedTransformer(Backbone):
    config: KnowledgeBasedTransformerConfig

    def __post_init__(self) -> None:
        model_name = self.config.model_name_or_path
        model = AutoKnowledgeBasedModel.from_pretrained(model_name)
        tokenizer = KnowledgeBasedTokenizer.from_pretrained(model_name)
        tokenizer.load_triples(
            Path(tklearn.__file__).parents[2] / "resources/triplets.csv"
        )
        tokenizer.prepare_model(model)
        self.tokenizer = tokenizer
        self.model = model

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
