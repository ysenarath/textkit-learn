from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer

from tklearn.metrics import Accuracy
from tklearn.nn import Evaluator, Trainer
from tklearn.nn.callbacks import EarlyStopping, ProgbarLogger
from tklearn.nn.models import AutoModel, ModelConfig

MODEL_NAME_OR_PATH = "google-bert/bert-base-uncased"
DATASET = "yelp_review_full"
NUM_EPOCHS = 3

dataset = load_dataset(DATASET)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

small_train_dataset = (
    tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
)
small_eval_dataset = (
    tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
)
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=16)
valid_dataloader = DataLoader(small_eval_dataset, batch_size=32)

model_config = ModelConfig.from_dict({
    "type": "linear",
    "backbone": {
        "type": "transformer",
        "model_name_or_path": MODEL_NAME_OR_PATH,
    },
    "num_labels": 5,
})
model = AutoModel(model_config)


model.to("mps")

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
