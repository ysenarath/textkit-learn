from datasets import load_dataset
from transformers import AutoTokenizer

from tklearn.nn import encode
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

dataset = load_dataset(DATASET, split="train").select(range(8))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")


dataset = dataset.rename_column("label", "labels").map(
    tokenize_function, batched=True, remove_columns=["text"]
)

print(dataset)

dataset.set_format(type="torch")

encoded_dataset = encode(
    dataset,
    model,
    batch_size=10000,
    pin_memory=True,
    desc="Encoding dataset",
    encode_batch_size=32,
)

print(encoded_dataset)
