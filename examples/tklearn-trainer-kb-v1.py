from collections import Counter, defaultdict

import numpy as np
import torch
from datasets import Dataset, load_dataset
from tqdm import auto as tqdm
from transformers import AutoTokenizer

from tklearn.kb import KnowledgeBase
from tklearn.metrics import AUC, F1, Accuracy, Precision, Recall
from tklearn.nn.models import AutoModel
from tklearn.nn.utils import get_device

kb = KnowledgeBase("wiktionary")

MODEL_NAME_OR_PATH = "google-bert/bert-base-uncased"
DATASET = "yelp_review_full"
NUM_EPOCHS = 3
TRAIN_DATASET_SIZE_PER_CLASS = 1000
EVAL_DATASET_SIZE_PER_CLASS = 1000

METRICS = {
    "accuracy": Accuracy(),
    "macro_f1": F1(average="macro"),
    "macro_precision": Precision(average="macro"),
    "macro_recall": Recall(average="macro"),
    "auc": AUC(multi_class="ovr"),
}


model = AutoModel({
    "type": "linear",
    "backbone": {
        "type": "transformer",
        "model_name_or_path": MODEL_NAME_OR_PATH,
    },
    "num_labels": 5,
})

auto_device = get_device()

model.to(auto_device)

model_memory_allocated_bytes = (
    torch.cuda.memory_allocated(auto_device)
    if auto_device.type == "cuda"
    else 0
)
print(
    f"Model memory allocated before training: {model_memory_allocated_bytes / (1024 * 1024)} MB"
)

dataset = load_dataset(DATASET)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
collate_fn = None

dataset = load_dataset(DATASET)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
collate_fn = None


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")


tokenized_datasets = dataset.rename_column("label", "labels").map(
    tokenize_function, batched=True
)
tokenized_datasets.set_format(type="torch")
del dataset


class KeepNPerClass:
    def __init__(self, n, label_key="labels"):
        self.n = n
        self.label_key = label_key
        self.counts = defaultdict(int)

    def __call__(self, example):
        label = example[self.label_key].item()
        if self.counts[label] < self.n:
            self.counts[label] += 1
            return True
        return False


keeper = KeepNPerClass(TRAIN_DATASET_SIZE_PER_CLASS, label_key="labels")
small_train_dataset = (
    tokenized_datasets["train"].filter(keeper).flatten_indices()
)


def extract_definitions(docs, classes):
    data = {
        "text": [],
        "label_counts": [],
    }
    for i in range(len(docs["text"])):
        doc = {k: docs[k][i] for k in docs}
        label = doc["labels"].item()
        def_counts = defaultdict(Counter)
        for m in kb.extract_mentions(doc["text"]):
            for c in m.candidates:
                for char_idx in range(*m.span):
                    def_counts[c.definition].update([label])
        for d, c in def_counts.items():
            data["text"].append(d)
            data["label_counts"].append([
                c[i] for i in range(len(set(classes)))
            ])
    return data


definition_train_dataset = small_train_dataset.map(
    extract_definitions,
    batched=True,
    remove_columns=small_train_dataset.column_names,
    fn_kwargs={"classes": small_train_dataset["labels"].tolist()},
)


# def group_data(batch):
#     summed_counts = []
#     for doc in batch:
#         label_counts = doc["label_counts"]
#         summed_counts.append(label_counts)
#     return {"label_counts": np.sum(summed_counts, axis=0).tolist()}


# GroupBy(definition_train_dataset, by="text", verbose=1).agg(group_data)


# groupby text and then sum label counts
def aggregate_definitions():
    definition_agg = {}
    for i in tqdm.trange(len(definition_train_dataset)):
        d = definition_train_dataset[i]
        text = d["text"]
        label_counts = d["label_counts"]
        if text not in definition_agg:
            definition_agg[text] = np.zeros(len(label_counts), dtype=int)
        definition_agg[text] += np.array(label_counts)
    for item in definition_agg.items():
        denom = np.sum(item[1])
        probas = (
            item[1] / denom
            if denom > 0
            else np.zeros_like(item[1], dtype=float)
        )
        yield {"text": item[0], "labels": probas}


definitions = Dataset.from_generator(aggregate_definitions)
