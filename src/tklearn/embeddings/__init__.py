from tklearn.embeddings.base import AutoEmbedding, Embedding
from tklearn.embeddings.fasttext import FastTextEmbedding
from tklearn.embeddings.gensim import GensimEmbedding
from tklearn.embeddings.transformers import SentenceTransformerEmbedding

TransformersEmbedding = SentenceTransformerEmbedding  # backward compatibility

__all__ = [
    "Embedding",
    "AutoEmbedding",
    "FastTextEmbedding",
    "GensimEmbedding",
    "TransformersEmbedding",
    "SentenceTransformerEmbedding",
]
