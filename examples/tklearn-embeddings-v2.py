from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from tklearn.embeddings import AutoEmbedding

path = Path(__file__).parent

wv = AutoEmbedding({
    "loader": "transformers",
    "name": "google-bert/bert-base-uncased",
})

file = "embedding_hate_gen.csv"
df = pd.read_csv(path / file, header=0)
df = df.dropna(subset=["text"])

tqdm.pandas(desc="Embedding Progress")

result = df["text"].progress_apply(wv.encode)

df["embedding"] = [str(item.tolist()) for item in result]

df.to_csv(path / f"embedded_{file}", index=False)
