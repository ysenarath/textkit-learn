import torch
from datasets import load_dataset
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from tklearn.metrics.classification import Accuracy, F1Score, Precision, Recall
from tklearn.nn.callbacks import ProgbarLogger
from tklearn.nn.torch import Model
from tklearn.nn.utils.data import RecordBatch

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

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-5)

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

callbacks = [
    ProgbarLogger(),
]

model = BinaryTextClassifier(checkpoint)

model = model.to("mps")

model.fit(
    tokenized_datasets["train"][:20],
    batch_size=8,
    epochs=3,
    shuffle=True,
    validation_data=tokenized_datasets["validation"],
    metrics=metrics,
    callbacks=callbacks,
)
