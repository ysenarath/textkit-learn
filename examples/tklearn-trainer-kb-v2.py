from collections import Counter, defaultdict

import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from tklearn.kb import KnowledgeBase
from tklearn.metrics import AUC, F1, Accuracy, Precision, Recall
from tklearn.nn import Evaluator, Trainer
from tklearn.nn.calibration.temperature import calibrate_model
from tklearn.nn.callbacks import EarlyStopping, ProgbarLogger
from tklearn.nn.models import AutoModel
from tklearn.nn.utils import get_device


def train():
    keeper = KeepNPerClass(EVAL_DATASET_SIZE_PER_CLASS, label_key="labels")
    small_eval_dataset = tokenized_datasets["test"].filter(keeper)

    train_dataloader = DataLoader(
        small_train_dataset,
        shuffle=True,
        batch_size=16,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        small_eval_dataset,
        batch_size=32,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # training the model

    evaluator = Evaluator(
        model,
        valid_dataloader,
        callbacks=[ProgbarLogger()],
        metrics=METRICS,
        prefix="valid_",
    )

    optimizer = AdamW(model.parameters(), lr=5e-5)

    trainer = Trainer(
        model,
        train_dataloader,
        optimizer=optimizer,
        callbacks=[ProgbarLogger(), EarlyStopping(patience=5)],
        evaluator=evaluator,
        epochs=NUM_EPOCHS,
        lr_scheduler="linear",
        lr_scheduler_kwargs={"num_warmup_steps": 0},
    )

    trainer.train()

    evaluator = Evaluator(
        model,
        valid_dataloader,
        callbacks=[ProgbarLogger()],
        metrics=METRICS,
        prefix="valid_",
    )

    valid_final_reslts = evaluator.evaluate()

    print(valid_final_reslts)

    calibrated_model = calibrate_model(model, valid_dataloader)

    print("Temperature: ", calibrated_model.temperature)

    evaluator = Evaluator(
        calibrated_model,
        valid_dataloader,
        callbacks=[ProgbarLogger()],
        metrics=METRICS,
        prefix="valid_",
    )

    valid_final_reslts = evaluator.evaluate()

    print(valid_final_reslts)


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


def group_keys_by_overlap(data: dict[str, list[int]]) -> list[list[str]]:
    # Map each value to the keys that contain it
    value_to_keys = defaultdict(set)
    for key, values in data.items():
        for v in values:
            value_to_keys[v].add(key)

    # Build adjacency list for graph of keys
    adj = defaultdict(set)
    for keys in value_to_keys.values():
        keys = list(keys)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                adj[keys[i]].add(keys[j])
                adj[keys[j]].add(keys[i])

    # Find connected components
    visited = set()
    groups = []
    group_ids = []
    curr_group_id = 0

    for key in data.keys():
        if key not in visited:
            stack = [key]
            component = []
            while stack:
                k = stack.pop()
                if k not in visited:
                    visited.add(k)
                    component.append(k)
                    stack.extend(adj[k])
            groups.extend(component)
            group_ids.extend([curr_group_id] * len(component))
            curr_group_id += 1

    return dict(zip(groups, group_ids))


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

# groupby text and then sum label counts
group_ids = group_keys_by_overlap({
    definition_train_dataset[i]["text"]: i
    for i in range(len(definition_train_dataset))
})
