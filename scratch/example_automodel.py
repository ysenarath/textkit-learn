from transformers import AutoModel, AutoTokenizer

MODEL_NAME_OR_PATH = "google-bert/bert-base-uncased"

model = AutoModel.from_pretrained(MODEL_NAME_OR_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)


def tokenize(text):
    return tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)


examples = [
    "Hello, my dog is cute",
    "Hello, my cat is cute",
]


inputs = tokenize(examples)

outputs = model(**inputs)

print(outputs.pooler_output)
