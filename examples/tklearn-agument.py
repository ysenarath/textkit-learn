from tklearn.kb.base import KnowledgeBase
from tklearn.kb.models import Triple
from tklearn.kb.wiktionary import WiktionaryArtifactStore

store = WiktionaryArtifactStore()

print("Wiktionary data setup complete.")
print(f"Lexicon size: {len(store.lexicon)}")
print(f"Triplet store size: {len(store.triples)}")
print(f"Gloss2idx size: {len(store.gloss2idx)}")
print(f"Idx2gloss size: {len(store.idx2gloss)}")
print(f"Senses size: {len(store.senses)}")
print(f"Embeddings size: {len(store.embeddings)}")

kb = KnowledgeBase(store)


def filter_func(item: Triple):
    subject_sense = item.subject[1]
    if subject_sense in kb.attrs["hate_related"]:
        return True
    return False


text = "She's a pure Oreo. You know, like the cookie, black outside and white inside."

for item in kb.augment(text, filter_func=filter_func):
    print(item)
