import evaluate
import numpy as np
from datasets import load_dataset
from scipy.special import softmax
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME_OR_PATH = "google-bert/bert-base-uncased"
DATASET = "yelp_review_full"
NUM_EPOCHS = 3
TRAIN_DATASET_SIZE = 1000
EVAL_DATASET_SIZE = 1000

# evaluation_strategy
metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
roc_auc_score = evaluate.load("roc_auc")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME_OR_PATH, num_labels=5
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    labels = labels.astype(np.int32)
    prediction_scores = softmax(logits, axis=-1).astype(np.float32)  # noqa: F841
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": metric.compute(predictions=predictions, references=labels),
        "f1": f1_metric.compute(
            predictions=predictions, references=labels, average="macro"
        ),
        # TODO: Fix this
        # "roc_auc": roc_auc_score.compute(
        #     references=labels,
        #     prediction_scores=prediction_scores,
        #     multi_class="ovr",
        # ),
    }


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


dataset = load_dataset(DATASET)
tokenized_datasets = dataset.map(tokenize_function, batched=True)
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

training_args = TrainingArguments(
    output_dir="./examples/outputs",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    # use_mps_device=True,
    no_cuda=False,
    num_train_epochs=NUM_EPOCHS,
    eval_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
