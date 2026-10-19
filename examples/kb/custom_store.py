"""Build a small knowledge store of your own and query it.

`write_tables` writes the four tables of a store, `TableStore` reads them,
and `KnowledgeBase` finds the known words in a text, their senses and the
relations between them. The store here has two words for dogs and one for
happiness:

- a sense is a gloss, identified by its position in `glosses`;
- the lexicon maps a form ("puppies") to the word it refers to ("puppy");
- a triple relates a word, or one sense of it, to another word.

Relations are also read from the other word's side: "puppy hypernym dog"
makes "puppy" a hyponym of "dog", and "cheerful synonym happy" makes
"cheerful" a synonym of "happy". A relation with no sense holds for every
sense of the word. A replacement is the related word as it is: it is
neither inflected nor capitalized to match what it replaces.

Runs offline in about a second.

python examples/kb/custom_store.py
"""

from __future__ import annotations

from pathlib import Path

from tklearn.kb import KnowledgeBase, TableStore, write_tables

OUTPUT = Path(__file__).parents[1] / "outputs" / "kb" / "custom_store"

DOG, PUPPY, HAPPY = 0, 1, 2


def check(condition: bool, message: str) -> None:
    """Print a passed check, or fail the example."""
    if not condition:
        raise AssertionError(f"FAILED: {message}")
    print(f"  ok  {message}")


def main() -> None:
    write_tables(
        OUTPUT,
        glosses=[
            "A domesticated canine.",
            "A young dog.",
            "Feeling or showing pleasure.",
        ],
        senses=[("dog", DOG), ("puppy", PUPPY), ("happy", HAPPY)],
        # (form, word, sense): a sense of None means any sense of the word
        lexicon=[
            ("dog", "dog", None),
            ("dogs", "dog", None),
            ("puppy", "puppy", None),
            ("puppies", "puppy", None),
            ("happy", "happy", None),
        ],
        # (subject, subject sense, predicate, object, object sense)
        triples=[
            ("dog", DOG, "synonym", "hound", None),
            ("puppy", PUPPY, "hypernym", "dog", DOG),
            ("happy", HAPPY, "synonym", "glad", None),
            ("cheerful", None, "synonym", "happy", None),
        ],
        metadata={"source": "examples/kb/custom_store.py"},
    )
    kb = KnowledgeBase(TableStore(OUTPUT))

    text = "Two happy puppies chase dogs."
    mentions = list(kb.extract_mentions(text))
    print(f"{text!r}")
    for mention in mentions:
        senses = [c.definition for c in mention.candidates]
        print(f"  {mention.form!r} at {tuple(mention.span)}: {senses}")

    dog = kb.get_candidate("dog", DOG)
    relations = kb.extract_relations(dog)
    print("\nrelations of dog:")
    for triple in relations:
        print(f"  {triple.subject} {triple.predicate} {triple.object}")

    print("\naugmentations:")
    augmentations = list(kb.augment(text, predicates={"synonym", "hyponym"}))
    for augmentation in augmentations:
        print(f"  {augmentation.text}")
    print()

    check(
        [m.form for m in mentions] == ["happy", "puppies", "dogs"],
        "inflected forms are found: 'puppies' and 'dogs'",
    )
    check(
        [(t.predicate, t.object[0]) for t in relations]
        == [("synonym", "hound"), ("hyponym", "puppy")],
        "'puppy hypernym dog' makes puppy a hyponym of dog",
    )
    texts = {a.text for a in augmentations}
    check(
        "Two cheerful puppies chase dogs." in texts,
        "'cheerful synonym happy' makes cheerful a synonym of happy",
    )
    check(
        "Two happy puppies chase puppy." in texts,
        "replacements are not inflected: 'dogs' becomes 'puppy'",
    )


if __name__ == "__main__":
    main()
