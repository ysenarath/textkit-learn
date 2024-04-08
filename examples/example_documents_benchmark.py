from tklearn.base import Document, DocumentList
from tqdm import auto as tqdm

index = DocumentList()

pbar = tqdm.trange(1_000_000)

for i in pbar:
    data = {
        "id": f"d{i}",
        "text": "hello world",
        "label": "spam",
        "tokens": ["hello", "world"],
    }
    doc = Document(data)
    index.add(doc)

print(index)
