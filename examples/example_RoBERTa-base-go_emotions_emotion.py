import re

import pandas as pd
from datasets import load_dataset
from tqdm import auto as tqdm
from transformers import pipeline

classifier = pipeline(
    task="text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None,
)

datasets = load_dataset("go_emotions", "simplified")

test_df = datasets["test"].to_pandas()

results = []
CHUNK_SIZE = 100
for chunk in tqdm.tqdm(range(test_df.shape[0] // CHUNK_SIZE + 1)):
    chunk_texts = test_df[CHUNK_SIZE * chunk : (chunk + 1) * CHUNK_SIZE][
        "text"
    ].to_list()
    results += classifier(chunk_texts)

y_pred = pd.DataFrame([
    pd.DataFrame(row)
    .set_index("label")
    .rename(lambda x: f"y_score[{re.escape(x)}]")["score"]
    for row in results
]).reset_index(drop=True)


def labels_int2str(labels):
    return [
        datasets["train"].features["labels"].feature.int2str(label) for label in labels
    ]


pd.concat(
    [
        test_df.assign(
            label_ids=test_df["labels"],
            labels=test_df["labels"].apply(labels_int2str),
        ),
        y_pred,
    ],
    axis=1,
)
