import pyarrow as pa

schema = pa.schema(
    [pa.field("n_legs", pa.int64()), pa.field("animals", pa.string())],
    metadata={"author": "Wes McKinney"},
)

metadata = schema.metadata

metadata["test"] = "test"

print(schema.metadata)
print(metadata)
