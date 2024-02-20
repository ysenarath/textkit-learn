from tklearn.preprocessing.tokenization import HuggingFaceTokenizer

tokenizer = HuggingFaceTokenizer.from_pretrained("bert-base-uncased")

enc = tokenizer.tokenize("Hello, world!", return_tensors="pt")

print(enc["input_ids"])
