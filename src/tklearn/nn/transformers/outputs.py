from dataclasses import dataclass, is_dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import (
    SequenceClassifierOutput as _SequenceClassifierOutput,
)

__all__ = [
    "SequenceClassifierOutput",
]


@dataclass
class SequenceClassifierOutput(_SequenceClassifierOutput):
    # loss, logits, hidden_states, attentions
    pooler_output: Optional[torch.FloatTensor] = None

    @classmethod
    def from_output(cls, *args, **kwargs):
        if len(args) == 1 and is_dataclass(args[0]):
            data = {k: v for k, v in args[0].items() if v is not None}
            data.update(kwargs)
        else:
            data = dict(*args, **kwargs)
        return cls(
            loss=data.get("loss"),
            logits=data.get("logits"),
            hidden_states=data.get("hidden_states"),
            attentions=data.get("attentions"),
            pooler_output=data.get("pooler_output"),
        )
