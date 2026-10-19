import pickle
import tempfile
import unittest
from pathlib import Path

from tklearn.kb import (
    Augmentation,
    KnowledgeBase,
    Span,
    TableStore,
    Triple,
    TripleStore,
    write_tables,
)


def write_tiny_store(path):
    write_tables(
        path,
        glosses=["a canine", "feeling joy"],
        senses=[("dog", 0), ("happy", 1)],
        # surface form -> (word, sense) it may refer to
        lexicon=[
            ("dogs", "dog", None),
            ("dog", "dog", None),
            ("happy", "happy", None),
            ("Hot dog", "hot dog", None),
        ],
        triples=[
            ("dog", 0, "synonym", "hound", None),
            ("dog", 0, "hypernym", "animal", None),
            ("happy", 1, "synonym", "glad", None),
            ("happy", 1, "synonym", "joyful", None),
        ],
        metadata={"source": "test"},
    )


class TestKnowledgeBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "tiny"
        write_tiny_store(self.path)
        self.kb = KnowledgeBase(TableStore(self.path))
        self.text = "The happy dogs."

    def tearDown(self):
        self.tmp.cleanup()

    def test_extract_mentions(self):
        mentions = list(self.kb.extract_mentions(self.text))
        self.assertEqual([m.form for m in mentions], ["happy", "dogs"])
        self.assertEqual(mentions[1].span, Span(10, 14))
        self.assertEqual(
            [c.definition for c in mentions[1].candidates], ["a canine"]
        )

    def test_extract_mentions_overlapping(self):
        mentions = list(self.kb.extract_mentions("A hot dog."))
        self.assertEqual([m.form for m in mentions], ["hot dog", "dog"])

    def test_extract_mentions_exact_form(self):
        mentions = list(self.kb.extract_mentions(self.text, exact_form=True))
        # "dogs" only matches the lemma "dog", so it has no exact candidate
        self.assertEqual(mentions[1].candidates, [])

    def test_extract_mentions_custom_stopwords(self):
        mentions = list(
            self.kb.extract_mentions(self.text, stopwords=["HAPPY"])
        )
        self.assertEqual([m.form for m in mentions], ["dogs"])

    def test_extract_relations(self):
        dog = self.kb.get_candidate("dog", 0)
        self.assertEqual(len(self.kb.extract_relations(dog)), 2)
        synonyms = self.kb.extract_relations(dog, predicates={"synonym"})
        self.assertEqual([t.object for t in synonyms], [("hound", None)])

    def test_augment_uses_synonyms_by_default(self):
        augmentations = list(self.kb.augment(self.text))
        self.assertTrue(
            all(isinstance(a, Augmentation) for a in augmentations)
        )
        self.assertEqual(
            sorted(a.text for a in augmentations),
            ["The glad dogs.", "The happy hound.", "The joyful dogs."],
        )
        hound = next(a for a in augmentations if a.replacement == "hound")
        self.assertEqual(hound.span, Span(10, 14))
        self.assertEqual(hound.original, self.text)
        self.assertEqual(hound.support, 1)
        self.assertEqual(hound.relations, [("dog", 0, "synonym", "hound", -1)])

    def test_augment_options(self):
        texts = {
            a.text
            for a in self.kb.augment(
                self.text, predicates=("synonym", "hypernym")
            )
        }
        self.assertIn("The happy animal.", texts)
        filtered = list(
            self.kb.augment(
                self.text, relation_filter=lambda t: t.object[0] != "glad"
            )
        )
        self.assertNotIn("The glad dogs.", {a.text for a in filtered})
        first = next(self.kb.augment(self.text, include_original=True))
        self.assertEqual((first.text, first.support), (self.text, 0))

    def test_pickles_through_store(self):
        kb = pickle.loads(pickle.dumps(self.kb))
        self.assertIsInstance(kb.store, TableStore)
        self.assertEqual(kb.store.path, self.path)

    def test_requires_a_store(self):
        with self.assertRaisesRegex(TypeError, "WiktionaryStore"):
            KnowledgeBase("wiktionary")


class TestExtractRelations(unittest.TestCase):
    def relations(self, triples, predicates=None):
        with tempfile.TemporaryDirectory() as tmp:
            write_tables(
                Path(tmp) / "store",
                glosses=["a canine", "a young dog"],
                senses=[("dog", 0), ("puppy", 1)],
                lexicon=[],
                triples=triples,
            )
            kb = KnowledgeBase(TableStore(Path(tmp) / "store"))
            dog = kb.get_candidate("dog", 0)
            return sorted(kb.extract_relations(dog, predicates), key=str)

    def test_includes_word_level_relations(self):
        relations = self.relations([
            ("dog", None, "synonym", "pooch", None),
            ("dog", 0, "synonym", "hound", None),
            ("dog", 1, "synonym", "wiener", None),
        ])
        self.assertEqual(
            relations,
            [
                Triple(("dog", 0), "synonym", ("hound", None)),
                Triple(("dog", None), "synonym", ("pooch", None)),
            ],
        )

    def test_reads_relations_from_the_object_side(self):
        relations = self.relations([
            ("puppy", 1, "hypernym", "dog", 0),
            ("hound", None, "synonym", "dog", None),
            ("pet", None, "category", "dog", None),
        ])
        self.assertEqual(
            relations,
            [
                Triple(("dog", 0), "hyponym", ("puppy", 1)),
                Triple(("dog", None), "synonym", ("hound", None)),
            ],
        )

    def test_inverse_follows_the_requested_predicate(self):
        triples = [("puppy", 1, "hypernym", "dog", 0)]
        self.assertEqual(self.relations(triples, {"hypernym"}), [])
        self.assertEqual(len(self.relations(triples, {"hyponym"})), 1)

    def test_keeps_one_duplicate_with_senses(self):
        relations = self.relations([
            ("dog", None, "synonym", "hound", None),
            ("hound", None, "synonym", "dog", 0),
        ])
        self.assertEqual(
            relations, [Triple(("dog", 0), "synonym", ("hound", None))]
        )


class TestTableStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "tiny"
        write_tiny_store(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_lookups(self):
        store = TableStore(self.path)
        self.assertEqual(store.metadata, {"source": "test"})
        self.assertEqual(store.get_gloss(1), "feeling joy")
        with self.assertRaises(KeyError):
            store.get_gloss(2)
        self.assertEqual(store.get_senses("dog"), [0])
        self.assertEqual(store.get_senses("cat"), [])
        self.assertIsNone(store.get_embedding(0))
        self.assertEqual(
            list(store.find_forms("Happy hot dogs")),
            [(0, 5, [("happy", None)]), (10, 14, [("dog", None)])],
        )

    def test_rewrite_replaces_the_store(self):
        write_tables(
            self.path, glosses=["new"], senses=[], lexicon=[], triples=[]
        )
        self.assertEqual(
            sorted(p.name for p in self.path.parent.iterdir()), ["tiny"]
        )
        self.assertEqual(TableStore(self.path).get_gloss(0), "new")

    def test_rejects_another_format(self):
        (self.path / "manifest.json").write_text('{"format": "1.0"}')
        with self.assertRaisesRegex(ValueError, "format"):
            TableStore(self.path)


class TestTripleStore(unittest.TestCase):
    def test_get_merges_senses_for_bare_words(self):
        store = TripleStore()
        store.add((("bank", 0), "synonym", ("shore", None)))
        store.add((("bank", 1), "synonym", ("lender", None)))
        self.assertEqual(
            store.get("bank"),
            {"synonym": {("shore", None), ("lender", None)}},
        )
        self.assertEqual(
            store.get(("bank", 1)), {"synonym": {("lender", None)}}
        )
        self.assertIsNone(store.get(("bank", 7)))


if __name__ == "__main__":
    unittest.main()
