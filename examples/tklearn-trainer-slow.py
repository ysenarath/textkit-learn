from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from tklearn.metrics import AUC, F1, Accuracy, Precision, Recall
from tklearn.nn import Evaluator, Trainer
from tklearn.nn.calibration.temperature import calibrate_model
from tklearn.nn.callbacks import EarlyStopping, ProgbarLogger
from tklearn.nn.models import AutoModel
from tklearn.nn.utils import get_device

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
collate_fn = None


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")


tokenized_datasets = dataset.rename_column("label", "labels").map(
    tokenize_function, batched=True, remove_columns=["text"]
)
tokenized_datasets.set_format(type="torch")
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
