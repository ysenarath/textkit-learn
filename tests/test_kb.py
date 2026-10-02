import pickle
import unittest

from tklearn.kb import (
    KNOWLEDGE_STORES,
    Augmentation,
    KnowledgeBase,
    KnowledgeStore,
    Lexicon,
    Span,
    TripleStore,
)


class TinyStore(KnowledgeStore):
    def __init__(self):
        self.lexicon = Lexicon()
        # surface form -> (word, sense) pairs it may refer to
        self.lexicon["dogs"] = {("dog", None)}
        self.lexicon["dog"] = {("dog", None)}
        self.lexicon["happy"] = {("happy", None)}
        self.gloss2idx = {"a canine": 0, "feeling joy": 1}
        self.idx2gloss = {0: "a canine", 1: "feeling joy"}
        self.senses = {"dog": {0}, "happy": {1}}
        self.embeddings = {}
        self.triples = TripleStore()
        self.triples.add((("dog", 0), "synonym", ("hound", None)))
        self.triples.add((("dog", 0), "hypernym", ("animal", None)))
        self.triples.add((("happy", 1), "synonym", ("glad", None)))
        self.triples.add((("happy", 1), "synonym", ("joyful", None)))


class TestKnowledgeBase(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase(TinyStore())
        self.text = "The happy dogs."

    def test_extract_mentions(self):
        mentions = list(self.kb.extract_mentions(self.text))
        self.assertEqual([m.form for m in mentions], ["happy", "dogs"])
        self.assertEqual(mentions[1].span, Span(10, 14))
        self.assertEqual(
            [c.definition for c in mentions[1].candidates], ["a canine"]
        )

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
        self.assertIsInstance(kb.store, TinyStore)

    def test_store_by_name(self):
        self.assertIn("wiktionary", KNOWLEDGE_STORES)
        with self.assertRaises(TypeError):
            KnowledgeBase(TinyStore(), offline=True)


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
