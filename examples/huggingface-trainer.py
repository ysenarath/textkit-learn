import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME_OR_PATH = "google-bert/bert-base-uncased"

# evaluation_strategy
metric = evaluate.load("accuracy")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-uncased", num_labels=5
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


dataset = load_dataset("yelp_review_full")
tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = (
    tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
)
small_eval_dataset = (
    tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
)


# training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")
training_args = TrainingArguments(
    output_dir="./examples/outputs",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    use_mps_device=True,
    num_train_epochs=3,
    evaluation_strategy="epoch",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
