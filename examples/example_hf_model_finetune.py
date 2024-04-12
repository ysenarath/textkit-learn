import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from tklearn.metrics.classification import Accuracy, F1Score, Precision, Recall
from tklearn.nn.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ProgbarLogger,
    TrackingCallback,
)
from tklearn.nn.data import RecordBatch
from tklearn.nn.torch import Model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from octoflow.tracking import SQLAlchemyTrackingStore, TrackingClient

run_path = Path("./examples/logs/experiment_1")

if run_path.exists():
    shutil.rmtree(run_path)

run_path.mkdir(parents=True, exist_ok=True)

dburi = f"sqlite:///{run_path / 'tracking.db'}"

store = SQLAlchemyTrackingStore(dburi)

client = TrackingClient(store)

experiment = client.create_experiment(
    name="example_hf_model_finetune",
    description="Example of finetuning a Hugging Face model",
)

run = experiment.start_run("example_hf_model_finetune")

raw_datasets = load_dataset("glue", "mrpc")

checkpoint = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(
        example["sentence1"],
        example["sentence2"],
        truncation=True,
        padding="max_length",
    )


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

tokenized_datasets = tokenized_datasets.remove_columns([
    "sentence1",
    "sentence2",
    "idx",
])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")


class BinaryTextClassifier(Model):
    def __init__(self, pretrained_model_name_or_path: str):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            num_labels=2,
        )

    def predict_on_batch(self, batch: RecordBatch) -> SequenceClassifierOutput:
        return self.model(**batch.x)

    def compute_loss(  # noqa: PLR6301
        self,
        batch: RecordBatch,
        output: SequenceClassifierOutput,
    ):
        return output.loss

    def extract_eval_input(  # noqa: PLR6301
        self,
        batch: RecordBatch,
        output: SequenceClassifierOutput,
    ):
        x, y_true = batch
        if y_true is None:
            y_true = x["labels"]
        logits: torch.Tensor = output["logits"]
        return {"y_true": y_true, "y_pred": logits.argmax(dim=1)}


metrics = {
    "accuracy": Accuracy(),
    "f1": F1Score(),
    "precision": Precision(),
    "recall": Recall(),
}

step_val = run.log_param("ll_step", 0)

callbacks = [
    ProgbarLogger(),
    ModelCheckpoint(run_path / "checkpoints"),
    EarlyStopping(monitor="valid_loss", patience=3),
    TrackingCallback(run, step=step_val),
]

model = BinaryTextClassifier(checkpoint).to("mps")

model.fit(
    tokenized_datasets["train"][:20],
    optimizer={
        "type": "AdamW",
        "lr": 1e-5,
        "weight_decay": 0.01,
    },
    lr_scheduler={
        "type": "LinearLR",
        "start_factor": 1 / 3,
        "total_iters": 1,
    },
    batch_size=8,
    epochs=10,
    shuffle=True,
    validation_data=tokenized_datasets["validation"],
    metrics=metrics,
    callbacks=callbacks,
)
