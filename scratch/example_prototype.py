import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from tklearn.nn.prototypes import PrototypeForSequenceClassification
from tklearn.nn.prototypes.loss import get_prototype_map
from transformers import AutoTokenizer

MODEL_NAME_OR_PATH = "google-bert/bert-base-uncased"

model = PrototypeForSequenceClassification.from_pretrained(MODEL_NAME_OR_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)


def tokenize(text):
    return tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)


# book review sentiment examples (0: negative, 1: positive, 2: neutral)
# each example must have at least 10 tokens and 10 examples
examples = [
    # positive examples
    "I love this book, it's so interesting.",
    "The book is great, but the ending is so sad.",
    "I'm so glad I bought this book.",
    # negative examples
    "This book is so boring, I can't even finish it.",
    "I can't believe how bad this book is.",
    "I wish I never bought this book.",
    # neutral examples
    "The book is okay, but I don't like it.",
    "I don't know how I feel about this book.",
    "I don't have an opinion about this book.",
]


batch = tokenize(examples)
batch["labels"] = torch.tensor([1, 1, 1, 0, 0, 0, 2, 2, 2], dtype=torch.long)

outputs = model.predict_step(batch)

loss = model.compute_loss(batch, outputs)

print(f"Loss: {loss.item()}")

batch_output = model.predict_step(batch, batch_idx=0, dataloader_idx=0)
pooler_output = batch_output["pooler_output"]

targets = batch["labels"]

prototypes_map = get_prototype_map(pooler_output, targets)

pooler_output = pooler_output.cpu().detach().numpy()
prototypes = np.stack([item.cpu().detach().numpy() for item in prototypes_map.values()])
pooler_output = np.concatenate([pooler_output, prototypes])

tsne = TSNE(
    random_state=1,
    n_iter=15000,
    metric="cosine",
    perplexity=5,
)

embs = tsne.fit_transform(pooler_output)
# Add to dataframe for convenience
df = pd.DataFrame(examples + list(prototypes_map), columns=["text"])

df["x"] = embs[:, 0]
df["y"] = embs[:, 1]
df["c"] = targets.tolist() + list(prototypes_map)

# Plot the embeddings
fig, ax = plt.subplots()
ax.scatter(df["x"], df["y"], c=df["c"], cmap="viridis")
for i, txt in enumerate(df["text"]):
    ax.annotate(txt, (df["x"][i], df["y"][i]))
plt.show()
