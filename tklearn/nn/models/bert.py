from typing import Optional, Union, Tuple

from transformers import AutoModel
import torch
from torch import nn

from tklearn.nn.base import Module

__all__ = [
    "BertSequenceAndTokenClassification",
]


class BertForSequenceAndTokenClassification(Module):
    """BertForSequenceAndTokenClassification

    Examples
    --------
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    >>> base_model = BertModel.from_pretrained("bert-base-uncased")
    """

    def __init__(self, base_model: AutoModel, num_sequence_labels, num_token_labels):
        super(BertForSequenceAndTokenClassification, self).__init__()
        self.bert = base_model
        config = base_model.config
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.token_classifier = nn.Linear(config.hidden_size, num_token_labels)
        self.cls_classifier = nn.Linear(config.hidden_size, num_sequence_labels)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], torch.Tensor]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=None,
        )
        outputs = self.dropout(outputs.hidden_states[-1])
        # class dropout and classifier
        cls_output = outputs[:, 0, :]
        cls_logits = self.cls_classifier(cls_output)
        # token classifier
        token_sequence_output = outputs[:, 1:, :]
        token_logits = self.token_classifier(token_sequence_output)
        # hidden_states=outputs.hidden_states,
        # attentions=outputs.attentions,
        return cls_logits, token_logits
