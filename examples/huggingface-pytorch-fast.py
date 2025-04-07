# 4352MiB
import evaluate
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)

from tklearn.nn.utils.collators import DataCollatorWithPadding

MODEL_NAME_OR_PATH = "google-bert/bert-base-uncased"
DATASET = "yelp_review_full"
NUM_EPOCHS = 3
TRAIN_DATASET_SIZE = 1000
EVAL_DATASET_SIZE = 1000

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)

dataset = load_dataset(DATASET)


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

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

collate_fn = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt",
)

train_dataloader = DataLoader(
    small_train_dataset, shuffle=True, batch_size=8, collate_fn=collate_fn
)
eval_dataloader = DataLoader(
    small_eval_dataset, batch_size=8, collate_fn=collate_fn
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME_OR_PATH, num_labels=5
)


optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = NUM_EPOCHS
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model.to(device)

progress_bar = tqdm(range(num_training_steps))

model_memory_allocated_bytes = (
    torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
)
print(
    f"Model memory allocated before training: {model_memory_allocated_bytes / (1024 * 1024)} MB"
)
memory_allocated_bytes = []

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        memory_allocated_bytes.append(
            (
                torch.cuda.memory_allocated(device)
                if device.type == "cuda"
                else 0
            )
            - model_memory_allocated_bytes
        )
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

print(
    f"Max memory allocated during training: {max(memory_allocated_bytes) / (1024 * 1024)} MB"
)

metric = evaluate.load("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
