import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer

from tklearn.metrics import AUC, F1, Accuracy, Precision, Recall
from tklearn.nn import Evaluator, Trainer
from tklearn.nn.calibration.temperature import calibrate_model
from tklearn.nn.callbacks import EarlyStopping, ProgbarLogger
from tklearn.nn.models import AutoModel, ModelConfig
from tklearn.nn.utils.collators import DataCollatorWithPadding

MODEL_NAME_OR_PATH = "google-bert/bert-base-uncased"
DATASET = "yelp_review_full"
NUM_EPOCHS = 3
TRAIN_DATASET_SIZE = 1000
EVAL_DATASET_SIZE = 1000

METRICS = {
    "accuracy": Accuracy(),
    "macro_f1": F1(average="macro"),
    "macro_precision": Precision(average="macro"),
    "macro_recall": Recall(average="macro"),
    "auc": AUC(multi_class="ovr"),
}

dataset = load_dataset(DATASET)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)


dataset = dataset.rename_column("label", "labels")
tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)
del dataset

small_train_dataset = (
    tokenized_datasets["train"]
    .shuffle(seed=42)
    .select(range(TRAIN_DATASET_SIZE))
)
small_eval_dataset = (
    tokenized_datasets["test"]
    .shuffle(seed=42)
    .select(range(EVAL_DATASET_SIZE))
)
train_dataloader = DataLoader(
    small_train_dataset, shuffle=True, batch_size=16, collate_fn=collator
)
valid_dataloader = DataLoader(
    small_eval_dataset, batch_size=32, collate_fn=collator
)

model_config = ModelConfig.from_dict({
    "type": "linear",
    "backbone": {
        "type": "transformer",
        "model_name_or_path": MODEL_NAME_OR_PATH,
    },
    "num_labels": 5,
})
model = AutoModel(model_config)

auto_device = "cpu"
if torch.cuda.is_available():
    auto_device = "cuda"
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    auto_device = "mps"

model.to(auto_device)

optimizer = AdamW(model.parameters(), lr=5e-5)

evaluator = Evaluator(
    model,
    valid_dataloader,
    callbacks=[ProgbarLogger()],
    metrics=METRICS,
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
