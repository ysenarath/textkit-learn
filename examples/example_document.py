import pyarrow as pa
from tklearn.base import Document, DocumentList
from tklearn.base.field import field

index = DocumentList()

data = {
    "id": "d40",
    "text": "hello world",
    "label": "spam",
    "tokens": ["hello", "world"],
}
doc = Document(data)
index.add(doc)

data = {"id": "f20", "label": "spam"}
doc = Document(data)
index.add(doc)

data = {
    "I_id": 15,
    "text": "hello world",
    "label": "spam",
    "tokens": ["hello", "world"],
}
id_field = field("I_id", pa.string(), lambda x: str(x))
doc = Document(data, id=id_field)
index.add(doc)

print(index)
print(index._schema)

index = DocumentList.from_table(index.to_table())

print(index._ids)
print(index._schema)
