"""Make variants of a sentence by replacing words with related ones.

`KnowledgeBase.augment` finds the words of a text that Wiktionary knows
and, for each, yields the text with that word replaced by a related one:
by default a synonym of any of its senses. Each `Augmentation` says which
relations support the replacement.

Most of Wiktionary's synonyms belong to a word as a whole, not to one
sense, so a word with many senses gets replacements from all of them
("quick" becomes "gravid", from "quick with child"). To stay close to one
meaning, pass the mentions with only the senses you want, and keep only
the relations of those senses with `relation_filter`.

python examples/kb/augment_text.py
"""

from __future__ import annotations

from collections import Counter

from tklearn.kb import KnowledgeBase, Mention, WiktionaryStore

TEXT = "The quick dog jumps over the lazy fox."


def check(condition: bool, message: str) -> None:
    """Print a passed check, or fail the example."""
    if not condition:
        raise AssertionError(f"FAILED: {message}")
    print(f"  ok  {message}")


def main() -> None:
    kb = KnowledgeBase(WiktionaryStore())

    synonyms = list(kb.augment(TEXT))
    replaced = Counter(TEXT[a.span.start : a.span.end] for a in synonyms)
    print(f"{len(synonyms)} variants of {TEXT!r} with synonyms, by word:")
    print(f"  {dict(replaced)}")
    for augmentation in synonyms[:3]:
        print(f"  {augmentation.text}  <- {augmentation.relations[0]}")

    mentions = {m.form: m for m in kb.extract_mentions(TEXT)}
    hypernyms = list(
        kb.augment(TEXT, mentions=[mentions["dog"]], predicates={"hypernym"})
    )
    print(f"\n'dog' replaced by its hypernyms ({len(hypernyms)}), e.g.:")
    for augmentation in hypernyms[:3]:
        print(f"  {augmentation.text}")

    quick = mentions["quick"]
    fast = [c for c in quick.candidates if "speed" in c.definition]
    one_sense = list(
        kb.augment(
            TEXT,
            mentions=[Mention(quick.form, quick.span, fast)],
            relation_filter=lambda triple: triple.subject[1] is not None,
        )
    )
    every_sense = [a for a in synonyms if a.span == quick.span]
    print(
        f"\n'quick': {len(every_sense)} synonyms over all its senses, "
        f"{len(one_sense)} for 'moving with speed':"
    )
    for augmentation in one_sense:
        print(f"  {augmentation.text}")
    print()

    check(
        all(
            a.text != TEXT
            and a.text.startswith(TEXT[: a.span.start])
            and a.text.endswith(TEXT[a.span.end :])
            for a in synonyms
        ),
        "each variant replaces exactly one word",
    )
    check(
        all(a.support == len(a.relations) >= 1 for a in synonyms),
        "each variant is supported by at least one relation",
    )
    check(
        "animal" in {a.replacement for a in hypernyms},
        "a dog is an animal: 'The quick animal jumps ...'",
    )
    check(
        {a.replacement for a in one_sense} == {"fast"},
        "only 'fast' is a synonym of quick's 'moving with speed' sense",
    )
    first = next(kb.augment(TEXT, include_original=True))
    check(
        (first.text, first.support) == (TEXT, 0),
        "include_original=True yields the unchanged text first",
    )


if __name__ == "__main__":
    main()
