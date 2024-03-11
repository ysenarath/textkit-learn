from datasets import load_dataset
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.modeling_outputs import SequenceClassifierOutput

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


class TaskModel(Model):
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
        batch_output: SequenceClassifierOutput,
    ):
        return batch_output.loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-5)


model = TaskModel(checkpoint)

model = model.to("mps")

model.fit(
    tokenized_datasets["train"][:20],
    batch_size=8,
    epochs=3,
    shuffle=True,
    validation_data=tokenized_datasets["validation"],
    callbacks=[
        ProgbarLogger(),
    ],
)
