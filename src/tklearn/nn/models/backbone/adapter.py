from __future__ import annotations

from typing import ClassVar, TypeVar, Union

import adapters
from adapters.heads import ModelWithFlexibleHeadsAdaptersMixin
from adapters.model_mixin import EmbeddingAdaptersWrapperMixin
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from tklearn.nn.models.backbone.base import Backbone, BackboneConfig
from tklearn.nn.models.backbone.transformer import get_features

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

    @property
    def features(self) -> dict[str, type]:
        """
        Get the signature of the model using the tokenizer.
        """
        if not hasattr(self, "tokenizer"):
            cls = self.tokenizer.__class__.__name__
            msg = f"expected PreTrainedTokenizer, got {cls}"
            raise TypeError(msg)
        if getattr(self, "_features", None) is None:
            self._features = get_features(self.tokenizer)
        return self._features

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
            if k not in self.features:
                continue
            kwargs[k] = v
        return self.model(**kwargs)
