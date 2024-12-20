from __future__ import annotations

from typing import ClassVar, TypeVar, Union

import adapters
from adapters.heads import ModelWithFlexibleHeadsAdaptersMixin
from adapters.model_mixin import EmbeddingAdaptersWrapperMixin
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from tklearn.models.backbone.base import Backbone, BackboneConfig

TRANSFORMERS_INPUTS = {
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "position_ids",
    "head_mask",
}

T = TypeVar("T", bound="Adapter")


class AdapterConfig(BackboneConfig):
    type: ClassVar[str] = "adapter"
    model_name_or_path: str = "bert-base-uncased"
    adapter: Union[dict, None, str] = None


class AdapterModel(
    EmbeddingAdaptersWrapperMixin,
    ModelWithFlexibleHeadsAdaptersMixin,
    PreTrainedModel,
):
    def __new__(cls, *args, **kwargs):
        # this is not a real constructor, but a factory method
        msg = "AdapterModel is not intended to be instantiated directly"
        raise ValueError(msg)

    @staticmethod
    def from_pretrained(*args, **kwargs) -> AdapterModel:
        model = AutoModel.from_pretrained(*args, **kwargs)
        adapters.init(model)
        return model


class Adapter(Backbone):
    config: AdapterConfig

    def __post_init__(self) -> None:
        self.model = AdapterModel.from_pretrained(
            self.config.model_name_or_path
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path
        )
        if self.config.adapter is None:
            msg = "adapter config is not provided"
            # import warnings
            # warnings.warn(msg, stacklevel=2)
            raise ValueError(msg)
        self.model.add_adapter("default", config=self.config.adapter)
        # model.set_active_adapters(adapter_name) for inference
        self.model.set_active_adapters("default")
        # disables training of all weights outside the task adapter
        #   to unfreeze all model weights later on, you can use
        #   self.model.freeze_model(False)
        self.model.train_adapter("default", train_embeddings=False)

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
