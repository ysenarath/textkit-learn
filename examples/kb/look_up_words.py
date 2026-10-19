"""Look up the senses of a word and its relations in Wiktionary.

`WiktionaryStore()` downloads the English Wiktionary tables from the
Hugging Face Hub the first time (about 170 MB) and reads them from disk
after that. A word's senses are its glosses. The relations of a sense
include those that Wiktionary gives for the word as a whole (their
subject has no sense) and those stated from the other word's side:
"puppy hypernym dog" also gives "dog hyponym puppy".

python examples/kb/look_up_words.py
"""

from __future__ import annotations

from collections import defaultdict

from tklearn.kb import KnowledgeBase, WiktionaryStore


def check(condition: bool, message: str) -> None:
    """Print a passed check, or fail the example."""
    if not condition:
        raise AssertionError(f"FAILED: {message}")
    print(f"  ok  {message}")


def main() -> None:
    store = WiktionaryStore()
    kb = KnowledgeBase(store)
    print(f"Wiktionary dump of {store.metadata['dump_modified']}\n")

    banks = list(kb.extract_candidates("bank"))
    print(f"'bank' has {len(banks)} senses, e.g.:")
    for candidate in banks[:5]:
        print(f"  #{candidate.sense_id}: {candidate.definition[:70]}")

    puppy = next(
        c
        for c in kb.extract_candidates("puppy")
        if c.definition.startswith("A young dog")
    )
    print(f"\npuppy #{puppy.sense_id}: {puppy.definition}")
    relations = defaultdict(list)
    for triple in kb.extract_relations(puppy):
        relations[triple.predicate].append(triple)
    for predicate in ("synonym", "hypernym"):
        triples = relations[predicate]
        print(f"  {len(triples)} {predicate}s, e.g.:")
        for triple in triples[:4]:
            sense = triple.subject[1]
            which = "the word" if sense is None else "this sense"
            print(f"    {triple.object[0]:<12} (of {which})")

    dog = next(kb.extract_candidates("dog"))
    hyponyms = [t.object[0] for t in kb.extract_relations(dog, {"hyponym"})]
    print(f"\ndog #{dog.sense_id} has {len(hyponyms)} hyponyms, e.g.:")
    print(f"  {', '.join(sorted(hyponyms)[:8])}\n")

    check(
        any("money" in c.definition for c in banks),
        "one sense of 'bank' is about money",
    )
    check(
        "dog" in [t.object[0] for t in relations["hypernym"]],
        "a puppy is a dog: 'puppy hypernym dog'",
    )
    check(
        any(t.subject[1] is None for t in relations["synonym"]),
        "some synonyms of puppy hold for the word as a whole",
    )
    check(
        "puppy" in hyponyms,
        "'puppy hypernym dog' gives 'dog hyponym puppy'",
    )


if __name__ == "__main__":
    main()
