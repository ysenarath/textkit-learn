from __future__ import annotations

from typing import ClassVar, Union

from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.tokenization_utils_base import BatchEncoding

from tklearn.nn.models.backbone.base import Backbone, BackboneConfig

# This is a sample text used to get the features of the tokenizer.
# It does not need to be meaningful, just long enough to test the tokenizer.
SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog."


def get_features(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
) -> dict[str, type]:
    if not isinstance(
        tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)
    ):
        msg = f"expected {PreTrainedTokenizer.__name__}, got {tokenizer.__class__.__name__}"
        raise TypeError(msg)
    encoding = tokenizer(SAMPLE_TEXT, return_tensors="pt")
    if not isinstance(encoding, BatchEncoding):
        msg = f"expected {BatchEncoding.__name__}, got {encoding.__class__.__name__}"
        raise TypeError(msg)
    return {key: type(value) for key, value in encoding.items()}


class TransformerConfig(BackboneConfig):
    type: ClassVar[str] = "transformer"
    model_name_or_path: str = "bert-base-uncased"


class Transformer(Backbone):
    config: TransformerConfig

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
            if k not in self.features:
                continue
            kwargs[k] = v
        return self.model(**kwargs)
