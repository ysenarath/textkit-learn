from tklearn.embeddings import AutoEmbedding

# Gensim

wv = AutoEmbedding.from_config("glove-twitter-25")

assert wv["hello"].shape == (25,)

wv = AutoEmbedding.from_config({"name": "glove-twitter-25"})

assert wv["hello"].shape == (25,)

wv = AutoEmbedding.from_config({
    "name": "gensim",
    "version": "glove-twitter-25",
})

assert wv["hello"].shape == (25,)

# FastText

wv = AutoEmbedding.from_config({
    "name": "fasttext",
    "version": "cc.en.300.bin",
})

assert wv["hello"].shape == (300,)
