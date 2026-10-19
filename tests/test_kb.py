import gzip
import os
import pickle
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import orjson
from click.testing import CliRunner

from tklearn.cli import main
from tklearn.embeddings import Embedding
from tklearn.kb import (
    Augmentation,
    KnowledgeBase,
    Span,
    TableStore,
    Triple,
    WiktionaryStore,
    write_tables,
)
from tklearn.kb.wiktionary import build_embeddings, build_wiktionary


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

    def test_augment_skips_the_mention_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_tables(
                Path(tmp) / "store",
                glosses=["a canine", "a surname"],
                senses=[("dog", 0), ("Dog", 1)],
                lexicon=[("dog", "dog", None), ("dog", "Dog", None)],
                triples=[
                    ("dog", 0, "synonym", "hound", None),
                    # a different word, but the same text as the mention
                    ("Dog", 1, "synonym", "DOG", None),
                ],
            )
            kb = KnowledgeBase(TableStore(Path(tmp) / "store"))
            texts = [a.text for a in kb.augment("A dog.")]
        self.assertEqual(texts, ["A hound."])

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

    def test_readable_by_others(self):
        umask = os.umask(0o022)
        try:
            write_tables(
                self.path, glosses=[], senses=[], lexicon=[], triples=[]
            )
        finally:
            os.umask(umask)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o755)

    def test_rejects_another_format(self):
        (self.path / "manifest.json").write_text('{"format": "1.0"}')
        with self.assertRaisesRegex(ValueError, "format"):
            TableStore(self.path)


class TestWiktionaryStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # what the Hub repo holds
        self.hub = Path(self.tmp.name) / "hub"
        write_tiny_store(self.hub)
        np.save(self.hub / "embeddings.npy", np.eye(2, dtype=np.float32))
        self.path = Path(self.tmp.name) / "wiktionary"
        self.downloads = []
        patcher = mock.patch(
            "tklearn.kb.wiktionary.store.snapshot_download", self.download
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def download(self, repo_id, repo_type, local_dir, allow_patterns):
        self.downloads.append((repo_id, sorted(allow_patterns)))
        for name in allow_patterns:
            if (self.hub / name).exists():
                shutil.copy(self.hub / name, Path(local_dir) / name)

    def test_downloads_the_tables_once(self):
        store = WiktionaryStore(path=self.path)
        self.assertEqual(
            self.downloads,
            [
                (
                    "textkit-learn/wiktionary-v0.5",
                    [
                        "glosses.parquet",
                        "lexicon.parquet",
                        "manifest.json",
                        "senses.parquet",
                        "triples.parquet",
                    ],
                )
            ],
        )
        self.assertEqual(store.get_senses("dog"), [0])
        self.assertIsNone(store.get_embedding(0))
        WiktionaryStore(path=self.path)
        self.assertEqual(len(self.downloads), 1)
        self.assertEqual(
            sorted(p.name for p in self.path.parent.iterdir()),
            ["hub", "wiktionary"],
        )

    def test_downloads_embeddings_when_asked(self):
        WiktionaryStore(path=self.path)
        store = WiktionaryStore(path=self.path, embeddings=True)
        self.assertEqual(self.downloads[1][1], ["embeddings.npy"])
        np.testing.assert_array_equal(store.get_embedding(1), [0, 1])

    def test_missing_hub_file(self):
        (self.hub / "embeddings.npy").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "embeddings.npy"):
            WiktionaryStore(path=self.path, embeddings=True)
        self.assertEqual(list(self.path.iterdir()), [])

    def test_pickles_as_its_arguments(self):
        store = WiktionaryStore(repo_id="me/wiktionary", path=self.path)
        copy = pickle.loads(pickle.dumps(store))
        self.assertEqual(repr(copy), repr(store))
        self.assertEqual(copy.get_gloss(0), "a canine")
        self.assertEqual(len(self.downloads), 1)


ENTRIES = [
    {
        "word": "dog",
        "lang_code": "en",
        "forms": [{"form": "dogs"}],
        "synonyms": [
            {"word": "pooch"},
            # Thesaurus lists include the word itself
            {"word": "dog"},
            {"word": "cur", "sense": "a worthless person"},
        ],
        "categories": ["English nouns"],
        "senses": [
            {
                "glosses": ["A canine."],
                "synonyms": [{"word": "hound"}],
                "hyponyms": [{"word": "puppy"}],
                "categories": ["en:Dogs"],
            }
        ],
    },
    {
        "word": "dogs",
        "lang_code": "en",
        "senses": [
            {"glosses": ["plural of dog"], "form_of": [{"word": "dog"}]}
        ],
    },
    {
        "word": "glad",
        "lang_code": "en",
        "antonyms": [{"word": "sad", "sense": "antonym(s) of “joyful”"}],
        "senses": [{"glosses": ["joyful"]}],
    },
    {"word": "Hund", "lang_code": "de", "senses": [{"glosses": ["dog"]}]},
]


class TestBuildWiktionary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dump = Path(self.tmp.name) / "dump.jsonl.gz"
        with gzip.open(self.dump, "wb") as f:
            for entry in ENTRIES:
                f.write(orjson.dumps(entry, option=orjson.OPT_APPEND_NEWLINE))
        self.path = Path(self.tmp.name) / "wiktionary"
        with mock.patch("tklearn.kb.wiktionary.build.tqdm"):
            build_wiktionary(self.dump, self.path)
        self.store = TableStore(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def relations(self, word):
        triples = self.store.get_relations(word)
        return sorted(
            (
                (tuple(t.subject), t.predicate, tuple(t.object))
                for t in triples
            ),
            key=str,
        )

    def test_senses(self):
        glosses = [self.store.get_gloss(i) for i in range(4)]
        self.assertEqual(
            glosses,
            ["a worthless person", "A canine.", "plural of dog", "joyful"],
        )
        with self.assertRaises(KeyError):
            self.store.get_gloss(4)
        self.assertEqual(self.store.get_senses("dog"), [0, 1])
        self.assertEqual(self.store.get_senses("Hund"), [])

    def test_relations(self):
        self.assertEqual(
            self.relations("dog"),
            [
                (("dog", 0), "synonym", ("cur", None)),
                (("dog", 1), "category", ("en:Dogs", None)),
                (("dog", 1), "synonym", ("hound", None)),
                (("dog", None), "category", ("English nouns", None)),
                (("dog", None), "synonym", ("pooch", None)),
            ],
        )
        # the hyponym is stored as the hypernym it implies
        self.assertEqual(
            self.relations("puppy"),
            [(("puppy", None), "hypernym", ("dog", 1))],
        )
        # "antonym(s) of" labels name the sense they are about
        self.assertEqual(
            self.relations("glad"),
            [(("glad", 3), "antonym", ("sad", None))],
        )

    def test_forms(self):
        self.assertEqual(
            sorted(next(self.store.find_forms("Dogs"))[2], key=str),
            [("dog", 2), ("dog", None), ("dogs", None)],
        )

    def test_metadata(self):
        self.assertEqual(self.store.metadata["dump"], "dump.jsonl.gz")
        self.assertEqual(
            self.store.metadata["predicates"],
            ["antonym", "category", "hypernym", "hyponym", "synonym"],
        )

    def test_predicates(self):
        build_wiktionary(self.dump, self.path, predicates=["antonym"])
        store = TableStore(self.path)
        self.assertEqual(store.get_relations("dog"), [])
        self.assertEqual(len(store.get_relations("glad")), 1)
        with self.assertRaisesRegex(ValueError, "unknown predicates"):
            build_wiktionary(self.dump, self.path, predicates=["meronym"])


class FakeEmbedding(Embedding):
    """Embeds a text as [len(text), number of earlier calls]."""

    def __init__(self, name="fake", fail_on_call=None):
        self.name = name
        self.fail_on_call = fail_on_call
        self.calls = 0

    @property
    def dim(self):
        return 2

    def encode(self, texts):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("interrupted")
        return np.array([[len(t), self.calls] for t in texts], np.float32)


class TestBuildEmbeddings(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "tiny"
        write_tiny_store(self.path)
        patcher = mock.patch("tklearn.kb.wiktionary.build.tqdm")
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def embeddings(self):
        store = TableStore(self.path)
        return [store.get_embedding(i).tolist() for i in range(2)]

    def test_writes_embeddings_and_metadata(self):
        build_embeddings(FakeEmbedding(), self.path, batch_size=1)
        # "a canine", "feeling joy"
        self.assertEqual(self.embeddings(), [[8, 1], [11, 2]])
        metadata = TableStore(self.path).metadata["embeddings"]
        self.assertEqual(
            (metadata["model"], metadata["embedding"], metadata["dim"]),
            ("fake", "FakeEmbedding('fake')", 2),
        )
        self.assertEqual(TableStore(self.path).metadata["source"], "test")
        self.assertEqual(
            sorted(p.name for p in self.path.iterdir()),
            [
                "embeddings.npy",
                "glosses.parquet",
                "lexicon.parquet",
                "manifest.json",
                "senses.parquet",
                "triples.parquet",
            ],
        )

    def test_resumes_after_an_interruption(self):
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            build_embeddings(FakeEmbedding(fail_on_call=2), self.path, 1)
        self.assertIsNone(TableStore(self.path).get_embedding(0))
        embedding = FakeEmbedding()
        build_embeddings(embedding, self.path, batch_size=1)
        self.assertEqual(embedding.calls, 1)
        self.assertEqual(self.embeddings(), [[8, 1], [11, 1]])

    def test_starts_over_for_another_model(self):
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            build_embeddings(FakeEmbedding(fail_on_call=2), self.path, 1)
        embedding = FakeEmbedding("other")
        build_embeddings(embedding, self.path, batch_size=1)
        self.assertEqual(embedding.calls, 2)


class TestCommand(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "tiny"
        write_tiny_store(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def invoke(self, *args):
        result = CliRunner().invoke(main, ["kb", "--path", self.path, *args])
        self.assertEqual(result.exit_code, 0, result.output)
        return result.output.splitlines()

    def test_senses(self):
        self.assertEqual(self.invoke("senses", "dog"), ["#0\ta canine"])

    def test_relations(self):
        self.assertEqual(
            self.invoke("relations", "dog"),
            ["dog#0\thypernym\tanimal", "dog#0\tsynonym\thound"],
        )
        self.assertEqual(
            self.invoke("relations", "dog", "-s", "0", "-p", "synonym"),
            ["dog#0\tsynonym\thound"],
        )

    def test_mentions(self):
        self.assertEqual(
            self.invoke("mentions", "The happy dogs."),
            [
                "happy (4-9)",
                "  happy#1\tfeeling joy",
                "dogs (10-14)",
                "  dog#0\ta canine",
            ],
        )

    def test_augment(self):
        lines = self.invoke("augment", "The happy dogs.", "-p", "hypernym")
        self.assertEqual(lines, ["The happy animal."])

    def test_defaults_to_wiktionary(self):
        with mock.patch(
            "tklearn.kb.WiktionaryStore",
            side_effect=lambda: TableStore(self.path),
        ) as store:
            result = CliRunner().invoke(main, ["kb", "senses", "happy"])
        self.assertEqual(result.output, "#1\tfeeling joy\n")
        store.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
