import unittest

from tklearn.nn.models.backbone.knowledge.lexicon import Lexicon


class TestLexicon(unittest.TestCase):
    def setUp(self):
        # Lexicon is like a dictionary but with some additional features.
        self.lexicon = Lexicon()

    def test_add_word(self):
        self.lexicon["hello"] = 1
        self.assertIn("hello", self.lexicon)
        self.assertEqual(self.lexicon["hello"], 1)

    def test_add_word_twice(self):
        self.lexicon["hello"] = 1
        self.lexicon["hello"] = 2
        self.assertEqual(self.lexicon["hello"], 2)

    def test_add_phrase(self):
        self.lexicon["hello world"] = 1
        self.assertIn("hello world", self.lexicon)
        self.assertEqual(self.lexicon["hello world"], 1)

    def test_add_invalid_keys(self):
        with self.assertRaises(ValueError):
            self.lexicon[""] = 1
        with self.assertRaises(ValueError):
            self.lexicon[None] = 1
        with self.assertRaises(ValueError):
            self.lexicon[[]] = 1

    def test_add_non_string_valid_values(self):
        self.lexicon["hello"] = ""
        self.assertEqual(self.lexicon["hello"], "")
        self.lexicon["hello"] = None
        self.assertIsNone(self.lexicon["hello"])
        self.lexicon["hello"] = []
        self.assertListEqual(self.lexicon["hello"], [])

    def test_extract_no_overlap(self):
        self.lexicon["a b c"] = 1
        self.lexicon["d e"] = 2
        self.lexicon["f g h"] = 3
        text = "a b c d e f g h"
        matches = self.lexicon.extract(text)
        strings = []
        values = []
        for match in matches:
            value, start, end = match
            # print(f"\n{text[start:end]}", value, end="")
            strings.append(text[start:end])
            values.append(value)
        # print()
        self.assertListEqual(strings, ["a b c", "d e", "f g h"])
        self.assertListEqual(values, [1, 2, 3])

    def test_extract_full_text_with_overlap(self):
        self.lexicon["a b c d e"] = 0
        self.lexicon["b c"] = 1
        text = "a b c d e"
        matches = self.lexicon.extract(text)
        strings = []
        values = []
        for match in matches:
            value, start, end = match
            # print(f"\n{text[start:end]}", value, end="")
            strings.append(text[start:end])
            values.append(value)
        # print()
        self.assertListEqual(strings, ["a b c d e"])
        self.assertListEqual(values, [0])

    def test_extract_longer_part_overlap(self):
        self.lexicon["hello+world"] = 1
        self.lexicon["world, how"] = 2
        self.lexicon["world, how are"] = 0
        text = "hi! hello + world, how are you?"
        matches = self.lexicon.extract(text)
        strings = []
        values = []
        for match in matches:
            value, start, end = match
            # print(f"\n{text[start:end]}", value, end="")
            strings.append(text[start:end])
            values.append(value)
        # print()
        self.assertListEqual(strings, ["world, how are"])
        self.assertListEqual(values, [0])

    def test_extract_longer_part_overlap_start(self):
        self.lexicon["hello + world, how"] = 0
        self.lexicon["hi! hello"] = 1
        text = "hi! hello + world, how are you?"
        matches = self.lexicon.extract(text)
        strings = []
        values = []
        for match in matches:
            value, start, end = match
            # print(f"\n{text[start:end]}", value, end="")
            strings.append(text[start:end])
            values.append(value)
        # print()
        self.assertListEqual(strings, ["hello + world, how"])
        self.assertListEqual(values, [0])

    def test_extract_longer_part_overlap_end(self):
        self.lexicon["b c d e"] = 0
        self.lexicon["a b c"] = 1
        text = "a b c d e"
        matches = self.lexicon.extract(text)
        strings = []
        values = []
        for match in matches:
            value, start, end = match
            # print(f"\n{text[start:end]}", value, end="")
            strings.append(text[start:end])
            values.append(value)
        # print()
        self.assertListEqual(strings, ["b c d e"])
        self.assertListEqual(values, [0])

    def test_extract_longer_part_overlap_end_extra(self):
        self.lexicon["g h i j"] = 0
        self.lexicon["f g h"] = 1
        self.lexicon["c d e f"] = 2
        # text = "hi! hello + world, how are you?"
        text = "a b c d e f g h i j"
        matches = self.lexicon.extract(text)
        strings = []
        values = []
        for match in matches:
            value, start, end = match
            # print(f"\n{text[start:end]}", value, end="")
            strings.append(text[start:end])
            values.append(value)
        # print()
        self.assertListEqual(strings, ["c d e f", "g h i j"])
        self.assertListEqual(values, [2, 0])

    def test_extract_longer_part_overlap_end_extra2(self):
        self.lexicon["a b c"] = 0
        self.lexicon["b c d"] = 1
        self.lexicon["d e f g"] = 2
        text = "a b c d e f g"
        matches = self.lexicon.extract(text)
        strings = []
        values = []
        for match in matches:
            value, start, end = match
            # print(f"\n{text[start:end]}", value, end="")
            strings.append(text[start:end])
            values.append(value)
        # print()
        self.assertListEqual(strings, ["a b c", "d e f g"])
        self.assertListEqual(values, [0, 2])


if __name__ == "__main__":
    unittest.main()
