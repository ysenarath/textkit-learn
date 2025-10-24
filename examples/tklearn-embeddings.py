from tklearn.embeddings import AutoEmbedding

# Gensim

wv = AutoEmbedding("glove-twitter-25")

assert wv["hello"].shape == (25,)

wv = AutoEmbedding({"name": "glove-twitter-25"})

assert wv["hello"].shape == (25,)

wv = AutoEmbedding({
    "loader": "gensim",
    "name": "glove-twitter-25",
})

assert wv["hello"].shape == (25,)

# FastText

wv = AutoEmbedding({
    "loader": "fasttext",
    "name": "cc.en.300.bin",
})

assert wv["hello"].shape == (300,)

# print sample embedding
print("Sample embedding for 'hello':")
print(wv["hello"])

# transformers example

wv = AutoEmbedding({
    "loader": "transformers",
    "name": "sentence-transformers/all-MiniLM-L6-v2",
})
print(wv.shape)

assert wv["hello"].shape == (384,)
# print sample embedding
print("Sample embedding for 'hello':")
print(wv["hello"])


# full sentence example
wv = AutoEmbedding({
    "loader": "transformers",
    "name": "sentence-transformers/all-MiniLM-L6-v2",
})
print(wv.shape)
assert wv["hello world"].shape == (384,)
# print sample embedding
print("Sample embedding for 'hello world':")
print(wv["hello world"])

print("All tests passed.")
