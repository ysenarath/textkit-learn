"""
This module is a wrapper around the transformers library and relevant libraries
such as adapters and sentence-transformers.
"""

from tklearn.nn.transformers.config import TransformerConfig
from tklearn.nn.transformers.sequence_classification import (
    TransformerForSequenceClassification,
)

__all__ = [
    "TransformerConfig",
    "SequenceClassifierOutput",
    "TransformerForSequenceClassification",
    # supports sequence classification tasks:
    # - binary classification
    # - multi-class classification
    # - multi-label classification
]
