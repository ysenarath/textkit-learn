from __future__ import annotations

import types
import warnings
from typing import Any, Type, Union

import torch
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_utils import ModuleUtilsMixin

__all__ = ["KBertModel"]

UNSUPORTED_MODEL_ERROR = (
    "The base model must have the `get_extended_attention_mask` method."
)


def get_extended_attention_mask(
    self: PreTrainedModel,
    attention_mask: torch.Tensor,
    input_shape: tuple[int],
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """
    if dtype is None:
        dtype = self.dtype

    if not (attention_mask.dim() == 2 and self.config.is_decoder):
        # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
        if device is not None:
            warnings.warn(
                "The `device` argument is deprecated and will be removed in v5 of Transformers.",
                FutureWarning,
            )
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 4:
        extended_attention_mask = attention_mask
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder:
            extended_attention_mask = (
                ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, device
                )
            )
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)

    # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
        dtype
    ).min

    return extended_attention_mask


class KBertModel(torch.nn.Module):
    base_model: PreTrainedModel

    def __init__(self, base_model: PreTrainedModel):
        super().__init__()
        # Locate the correct encoder to patch.
        # HF models expose the inner encoder via the `.base_model` property.
        # - If 'base_model' is a Classifier, .base_model returns the inner BERT/RoBERTa.
        # - If 'base_model' is already the bare BERT, .base_model returns self.
        target_encoder = base_model.base_model
        if not hasattr(target_encoder, "get_extended_attention_mask"):
            raise ValueError(UNSUPORTED_MODEL_ERROR)
        # 2. Patch the method on the specific instance
        target_encoder.get_extended_attention_mask = types.MethodType(
            get_extended_attention_mask, target_encoder
        )
        self.base_model = base_model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Any],
        model_cls: Type[PreTrainedModel] | None = None,
        **kwargs,
    ):
        """
        Load a pretrained model and wrap it.

        Args:
            pretrained_model_name_or_path: Path or HF Hub ID.
            model_cls: The specific HF class to load (e.g., AutoModel,
                       AutoModelForSequenceClassification). Defaults to AutoModel.
            **kwargs: Passed to the HF from_pretrained method.
        """
        if model_cls is None:
            model_cls = AutoModel
        if "attn_implementation" not in kwargs:
            kwargs["attn_implementation"] = "eager"
        # Instantiate the requested model architecture
        base_model = model_cls.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        return cls(base_model)

    @property
    def config(self) -> Any:
        return self.base_model.config

    def resize_token_embeddings(self, *args, **kwargs) -> None:
        self.base_model.resize_token_embeddings(*args, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs):
        self.base_model.save_pretrained(save_directory, **kwargs)

    def forward(self, *args, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        return self.base_model(*args, **kwargs)
