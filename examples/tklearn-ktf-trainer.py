# script for training knowledge-based transformer model using tklearn
import argparse
import time

import pyinstrument
from datasets import DatasetDict, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tklearn.metrics import Accuracy
from tklearn.nn import Evaluator, Trainer
from tklearn.nn.callbacks import EarlyStopping, ProgbarLogger
from tklearn.nn.models import AutoModel, ModelConfig

parser = argparse.ArgumentParser(description="Train a KBERT model.")
parser.add_argument(
    "-p",
    "--profile",
    action="store_true",
    help="Enable profiling",
    default=False,
)
args = parser.parse_args()

MODEL_NAME_OR_PATH = "google-bert/bert-base-uncased"
DATASET = "yelp_review_full"
NUM_EPOCHS = 3

dataset = load_dataset(DATASET)
dataset = DatasetDict({
    "train": dataset["train"].take(10000),
    "test": dataset["test"].take(10000),
})

model_config = ModelConfig.from_dict({
    "type": "linear",
    "backbone": {
        "type": "knowledge-based-transformer",
        "model_name_or_path": MODEL_NAME_OR_PATH,
        "prepare_model": False,
    },
    "num_labels": 5,
})
model = AutoModel(model_config)

model.to("cuda")


def tokenize_function(examples):
    if args.profile:
        with pyinstrument.Profiler() as profiler:
            encodings = model.tokenizer(examples["text"])
        profiler.open_in_browser()
        while True:
            time.sleep(10)
    else:
        encodings = model.tokenizer(examples["text"])
    visibility_mask = encodings.pop("visibility_mask")
    attention_mask = encodings.pop("attention_mask")
    encodings["attention_mask"] = (
        attention_mask.unsqueeze(1).expand(-1, 512, -1) * visibility_mask
    )
    return encodings


tokenized_datasets = dataset.map(
    tokenize_function, batched=True, batch_size=10
)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42)
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=16)
valid_dataloader = DataLoader(small_eval_dataset, batch_size=32)

optimizer = AdamW(
    model.parameters(),
    lr=5e-5,
)

evaluator = Evaluator(
    model,
    valid_dataloader,
    callbacks=[ProgbarLogger()],
    metrics={"acuracy": Accuracy()},
    prefix="valid_",
)

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
