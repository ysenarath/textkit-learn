from tklearn.embeddings import AutoEmbedding

# Gensim

wv = AutoEmbedding.from_config("glove-twitter-25")

assert wv["hello"].shape == (25,)

wv = AutoEmbedding.from_config({"name": "glove-twitter-25"})

assert wv["hello"].shape == (25,)

wv = AutoEmbedding.from_config({
    "loader": "gensim",
    "name": "glove-twitter-25",
})

assert wv["hello"].shape == (25,)

# FastText

wv = AutoEmbedding.from_config({
    "loader": "fasttext",
    "name": "cc.en.300.bin",
})

assert wv["hello"].shape == (300,)

# print sample embedding
print("Sample embedding for 'hello':")
print(wv["hello"])

print("All tests passed.")
