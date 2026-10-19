"""Pick the sense of a word that fits its sentence, with gloss embeddings.

Each sense in the Wiktionary store has an embedding of its gloss. Embed
the sentence with the same model and the sense whose gloss is closest
(highest dot product: the vectors have unit length) is the best guess at
what the word means there.

The published embeddings were made with BAAI/bge-large-en-v1.5 through
OpenRouter, which gives the same vectors as the model run locally, so
the sentences here are embedded on this machine: no API key needed. The
first run downloads the embeddings (about 6.6 GB) and the model (about
1.3 GB).

python examples/kb/disambiguate_senses.py
"""

from __future__ import annotations

from tklearn.embeddings import SentenceTransformerEmbedding
from tklearn.kb import KnowledgeBase, WiktionaryStore

MODEL = "BAAI/bge-large-en-v1.5"
SENTENCES = [
    "We sat on the bank of the river.",
    "I deposited cash at the bank.",
]


def check(condition: bool, message: str) -> None:
    """Print a passed check, or fail the example."""
    if not condition:
        raise AssertionError(f"FAILED: {message}")
    print(f"  ok  {message}")


def main() -> None:
    store = WiktionaryStore(embeddings=True)
    kb = KnowledgeBase(store)
    # OpenRouter's model ids are lowercase
    built_with = store.metadata["embeddings"]["model"]
    embedding = SentenceTransformerEmbedding(MODEL)
    print(f"gloss embeddings: {built_with}; sentences: {embedding}")

    scores = {}
    for sentence in SENTENCES:
        query = embedding.encode_query(sentence)
        bank = next(
            m for m in kb.extract_mentions(sentence) if m.form == "bank"
        )
        ranked = sorted(
            (float(query @ c.embedding), c.definition)
            for c in bank.candidates
            if c.word == "bank"
        )[::-1]
        scores[sentence] = dict((d, s) for s, d in ranked)
        print(f"\n{sentence}")
        for score, definition in ranked[:3]:
            print(f"  {score:.3f}  {definition[:70]}")
    print()

    check(
        built_with.lower() == MODEL.lower(),
        f"the glosses were embedded with {MODEL}",
    )
    river, money = (scores[s] for s in SENTENCES)
    edge = next(d for d in river if "watercourse" in d)
    institution = next(d for d in river if "borrow money" in d)
    check(
        river[edge] > money[edge],
        "the river sentence is closer to 'an edge of river, lake, ...'",
    )
    check(
        money[institution] > river[institution],
        "the cash sentence is closer to 'an institution where one can ...'",
    )


if __name__ == "__main__":
    main()
