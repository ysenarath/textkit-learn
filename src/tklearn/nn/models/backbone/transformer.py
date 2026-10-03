from __future__ import annotations

from typing import Union

from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.tokenization_utils_base import BatchEncoding

from tklearn.nn.models.backbone.base import Backbone

__all__ = [
    "TransformerBackbone",
]

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


class TransformerBackbone(Backbone):
    """A Hugging Face transformer encoder and its tokenizer.

    Parameters
    ----------
    model_name_or_path : str, default="bert-base-uncased"
        Hub id or local path passed to ``AutoModel.from_pretrained`` and
        ``AutoTokenizer.from_pretrained``.

    Notes
    -----
    `forward` passes only the batch keys the tokenizer produces (such as
    ``input_ids`` and ``attention_mask``) to the model, so batches may carry
    labels and other fields.
    """

    def __init__(self, model_name_or_path: str = "bert-base-uncased") -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.model = self._load_model(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._features: dict[str, type] | None = None

    def _load_model(self, model_name_or_path: str) -> PreTrainedModel:
        return AutoModel.from_pretrained(model_name_or_path)

    @property
    def features(self) -> dict[str, type]:
        """Batch keys the model accepts, as produced by the tokenizer."""
        if self._features is None:
            self._features = get_features(self.tokenizer)
        return self._features

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def forward(self, batch: dict) -> BaseModelOutputWithPooling:
        kwargs = {k: v for k, v in batch.items() if k in self.features}
        return self.model(**kwargs)
