from pathlib import Path
from typing import Generator, Iterable, List, Optional, Union

from datasets import load_from_disk
from typing_extensions import Protocol, runtime_checkable

from tklearn.kb.conceptnet.io import ConceptNetIO


class ConceptIndex:
    def __init__(
        self,
        path: Union[str, Path] = None,
        encoder: Union[Transformer, SentenceTransformer, Pipeline] = None,
    ):
        self.io = ConceptNetIO(path)
        self.nodes_path = self.io.csv_path.with_suffix(".dset") / "nodes"
        self.nodes_with_embeddings_path = (
            self.io.csv_path.with_suffix(".dset") / "nodes-with-embeddings"
        )
        self.encoder = encoder

    def extract_embeddings(self, example):
        try:
            return {
                "embeddings": self.encoder.transform(example["label"], batch_size=1000)
            }
        except Exception as e:
            gc.collect()
            torch.cuda.empty_cache()
            return extract_embeddings(example)

    def iter_concepts(self) -> Generator[dict, None, None]:
        self.io.to_jsonl(verbose=True, exist_ok=True)
        existing = set()
        for edge in self.io.read_jsonl_iter(verbose=True):
            for term_field in ["start", "end"]:
                node_id = edge[term_field]["@id"]
                if node_id not in existing:
                    existing.add(node_id)
                    yield edge[term_field]

    def build(self) -> None:
        if self.path.exists():
            return
        ds = load_from_disk(self.path)


if __name__ == "__main__":
    index = ConceptIndex()
