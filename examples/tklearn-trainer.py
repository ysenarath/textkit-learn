from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from tklearn.metrics import Accuracy
from tklearn.nn import Evaluator, Trainer
from tklearn.nn.callbacks import EarlyStopping, ProgbarLogger
from tklearn.nn.models import AutoModel, ModelConfig

MODEL_NAME_OR_PATH = "google-bert/bert-base-uncased"
DATASET = "yelp_review_full"
NUM_EPOCHS = 20

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

model_config = ModelConfig(
    type="linear",
    backbone=dict(
        type="transformer",
        name=MODEL_NAME_OR_PATH,
    ),
    num_labels=5,
)
model = AutoModel(model_config)

optimizer = AdamW(
    model.parameters(),
    lr=2e-6,
    # warmup=0.1,
    # t_total=len(train_dataloader) * NUM_EPOCHS,
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
)


trainer.train()
