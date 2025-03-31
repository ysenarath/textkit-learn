from datasets import load_dataset
from torch.utils.data import DataLoader

from tklearn.nn.base import Predictor
from tklearn.nn.models import AutoModel
from tklearn.nn.utils.collators import DataCollatorWithPadding

# Load the model
model = AutoModel({
    "type": "linear",
    "backbone": {"type": "transformer"},
    "num_labels": 2,
})

# Load the dataset
dataset = load_dataset("imdb", split="test").select(range(100))

# tokenize the dataset
dataset = dataset.map(
    lambda x: model.tokenizer(x["text"], truncation=True),
    batched=True,
    remove_columns=dataset.column_names,
)

# Set the format to torch
dataset.set_format(type="torch")

# Create a dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=DataCollatorWithPadding(model.tokenizer, pad_to_multiple_of=8),
)

# Create a predictor
logits = Predictor(model, dataloader=dataloader).predict()

assert logits.shape[1] == 2
