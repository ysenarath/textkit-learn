from tklearn.nn.models.kbert.collator import KBertCollatorWithPadding
from tklearn.nn.models.kbert.model import KBertModel
from tklearn.nn.models.kbert.tokenizer import KBertTokenizer

AutoKnowledgeBaseModel = KBertModel
KnowledgeBaseTokenizer = KBertTokenizer

__all__ = [
    "KBertModel",
    "KBertTokenizer",
    "KBertCollatorWithPadding",
]
