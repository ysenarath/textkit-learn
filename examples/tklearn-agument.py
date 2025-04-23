from tklearn.kb.base import KnowledgeBase
from tklearn.kb.models import Triple

kb = KnowledgeBase("wiktionary")

print("Wiktionary data setup complete.")
print(f"Lexicon size: {len(kb.lexicon)}")
print(f"Triplet store size: {len(kb.triples)}")
print(f"Gloss2idx size: {len(kb.gloss2idx)}")
print(f"Idx2gloss size: {len(kb.idx2gloss)}")
print(f"Senses size: {len(kb.senses)}")
print(f"Embeddings size: {len(kb.embeddings)}")


def filter_func(item: Triple):
    subject_sense = item.subject[1]
    if subject_sense in kb.attrs["hate_related"]:
        return True
    return False


text = "She's a pure Oreo. You know, like the cookie, black outside and white inside."

for item in kb.augment(text, filter_func=filter_func):
    print(item)
