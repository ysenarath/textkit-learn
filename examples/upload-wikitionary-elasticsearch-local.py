import os
from pathlib import Path

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import auto as tqdm

from tklearn.kb.wiktionary.models import parse_jsonl
from tklearn.kb.wiktionary.store_v2 import WIKTIONARY_URL, setup_wiktionary

here = Path(__file__).parent.resolve()

local_dir = here.parent / "cache"

setup_wiktionary(
    wiktionary_url=WIKTIONARY_URL,
    temp_download_path=local_dir / "wiktionary.jsonl.gz",
    temp_extracted_path=local_dir / "wiktionary.jsonl",
    wiktionary_path=None,
    language="en",
    remove_downloaded=False,
)

# upload to local Elasticsearch
ELASTIC_PASSWORD = os.environ.get("ELASTIC_PASSWORD", "")

# Create the client instance
client = Elasticsearch(
    hosts=["https://localhost:9200"],
    basic_auth=("elastic", ELASTIC_PASSWORD),
    verify_certs=False,
)

print(client.info())

INDEX = "wiktionary-20251211"
# drop the index if it exists
if client.indices.exists(index=INDEX):
    client.indices.delete(index=INDEX)

total = sum(1 for _ in open(local_dir / "wiktionary.jsonl", "rb"))

# read the line-jsonl file and upload to Elasticsearch
word_id = int("1" + "0" * len(str(total)))
fp = local_dir / "wiktionary.jsonl"
bulk_data = []
for word in tqdm.tqdm(parse_jsonl(fp), total=total):
    bulk_data.append({
        "_index": INDEX,
        "_id": word_id,
        "_source": word.model_dump(),
    })
    word_id += 1
    if len(bulk_data) >= 100_000:
        bulk(client, bulk_data)
        bulk_data = []
if bulk_data:
    bulk(client, bulk_data)
