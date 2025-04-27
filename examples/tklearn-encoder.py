from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from tklearn.nn import Encoder
from tklearn.nn.callbacks import ProgbarLogger
from tklearn.nn.models import AutoModel
from tklearn.nn.utils import get_device

MODEL_NAME_OR_PATH = "google-bert/bert-base-uncased"
DATASET = "yelp_review_full"

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

dataset = load_dataset(DATASET, split="train").select(range(1000))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")


dataset = dataset.rename_column("label", "labels").map(
    tokenize_function, batched=True, remove_columns=["text"]
)

dataset.set_format(type="torch")

dataloader = DataLoader(
    dataset,
    batch_size=32,
    pin_memory=True,
)

encoder = Encoder(
    model,
    dataloader=dataloader,
    callbacks=[ProgbarLogger()],
)

encoded = encoder.encode()

print(encoded)
