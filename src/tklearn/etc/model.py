from __future__ import annotations

import warnings
from typing import Any, Tuple

import torch
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.modeling_utils import ModuleUtilsMixin
from typing_extensions import Self

__all__ = ["AutoKnowledgeBasedModel"]


class AutoKnowledgeBasedModel(torch.nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.base_model: PreTrainedModel = AutoModel.from_config(config)
        if not hasattr(self.base_model, "get_extended_attention_mask"):
            raise ValueError(
                "The base model must have the `get_extended_attention_mask` method."
            )
        self.base_model.get_extended_attention_mask = (
            self.get_extended_attention_mask
        )

    @property
    def config(self) -> Any:
        return self.base_model.config

    def resize_token_embeddings(
        self,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        self.base_model.resize_token_embeddings(
            new_num_tokens=new_num_tokens,
            pad_to_multiple_of=pad_to_multiple_of,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> Self:
        return cls(AutoConfig.from_pretrained(pretrained_model_name_or_path))

    def __call__(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int],
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
            dtype = self.base_model.dtype

        if not (
            attention_mask.dim() == 2 and self.base_model.config.is_decoder
        ):
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
            if self.base_model.config.is_decoder:
                extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, device
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
        extended_attention_mask = extended_attention_mask.to(
            dtype=dtype
        )  # fp16 compatibility
        extended_attention_mask = (
            1.0 - extended_attention_mask
        ) * torch.finfo(dtype).min
        return extended_attention_mask
