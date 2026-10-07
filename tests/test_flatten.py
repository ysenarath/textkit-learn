import copy
import json
import math
import os
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tklearn.metrics import ROCPoints
from tklearn.utils.flatten import flatten, freeze, unflatten


class TestFlatten(unittest.TestCase):
    def test_nested_mappings_and_named_tuples(self):
        roc = ROCPoints(np.array([0.0, 1.0]), np.array([0.5, 1.0]), None)
        values = {"optim": {"lr": 0.1, "adam": {"eps": 1e-8}}, "roc": roc}
        self.assertEqual(
            flatten(values),
            {
                "optim.lr": 0.1,
                "optim.adam.eps": 1e-8,
                "roc.fpr": (0.0, 1.0),
                "roc.tpr": (0.5, 1.0),
                "roc.thresholds": None,
            },
        )

    def test_lists_and_arrays_are_kept_whole(self):
        values = {
            "hidden": [256, 128],
            "betas": (0.9, 0.99),
            "f1": np.array([0.5, 0.7]),
            "confusion_matrix": np.eye(2, dtype=int),
            "empty": [],
        }
        self.assertEqual(
            flatten(values),
            {
                "hidden": (256, 128),
                "betas": (0.9, 0.99),
                "f1": (0.5, 0.7),
                "confusion_matrix": ((1, 0), (0, 1)),
                "empty": (),
            },
        )

    def test_lists_of_mappings(self):
        roc = ROCPoints(np.array([0.0, 1.0]), np.array([0.5, 1.0]), None)
        values = {
            "layers": [{"size": 4}, {"size": 8, "act": "relu"}],
            "blocks": [
                {"layers": [{"k": 1}, {"k": 2}]},
                {"layers": [{"k": 3}]},
            ],
            "roc": [roc, roc],
            "mixed": [1, {"a": 2}],
        }
        self.assertEqual(
            flatten(values),
            {
                # None where an item lacks the key
                "layers.*.size": (4, 8),
                "layers.*.act": (None, "relu"),
                "blocks.*.layers.*.k": ((1, 2), (3,)),
                "roc.*.fpr": ((0.0, 1.0), (0.0, 1.0)),
                "roc.*.tpr": ((0.5, 1.0), (0.5, 1.0)),
                "roc.*.thresholds": (None, None),
                "mixed.*": (1, None),
                "mixed.*.a": (None, 2),
            },
        )

    def test_scalars_and_other_values(self):
        values = {
            "f32": np.float32(0.5),
            "i64": np.int64(3),
            "flag": np.bool_(True),
            "none": None,
            "dtype": np.dtype("float32"),
            3: "key",
        }
        flat = flatten(values)
        self.assertEqual(
            flat,
            {
                "f32": 0.5,
                "i64": 3,
                "flag": True,
                "none": None,
                "dtype": "float32",
                "3": "key",
            },
        )
        self.assertEqual(
            [type(flat[k]) for k in ("f32", "i64", "flag")],
            [float, int, bool],
        )

    def test_empty_mappings_are_values(self):
        flat = flatten({"sched": {}, "optim": {"extra": {}}, "items": [{}]})
        self.assertEqual(
            flat, {"sched": {}, "optim.extra": {}, "items.*": ({},)}
        )
        empty = flat["sched"]
        # read-only and hashable, like the other values
        for mutate in (
            lambda: empty.__setitem__("a", 1),
            lambda: empty.update(a=1),
            lambda: empty.setdefault("a", 1),
        ):
            with self.assertRaises(TypeError):
                mutate()
        self.assertEqual({empty: 1}, {flat["optim.extra"]: 1})
        # written as {}, and the same object after pickle and deepcopy
        self.assertEqual(json.dumps(empty), "{}")
        self.assertIs(pickle.loads(pickle.dumps(empty)), empty)
        self.assertIs(copy.deepcopy(empty), empty)
        self.assertIs(freeze({}), empty)

    def test_paths_are_absolute(self):
        with tempfile.TemporaryDirectory() as tmp:
            link = Path(tmp) / "link"
            link.symlink_to(Path(tmp) / "target", target_is_directory=True)
            values = {
                "data": Path("data/x.csv"),
                "ckpt": link / "a.pt",
                "text": "x/y",
            }
            self.assertEqual(
                flatten(values),
                {
                    # independent of the working directory
                    "data": os.path.join(os.getcwd(), "data", "x.csv"),
                    # symlinks, such as a shared mount, are not resolved
                    "ckpt": str(link / "a.pt"),
                    # strings are not known to be paths
                    "text": "x/y",
                },
            )

    def test_keys_that_flatten_to_the_same_name(self):
        cases = [
            ({"optim": {"lr": 0.1}, "optim.lr": 0.2}, "optim.lr"),
            ({"layers": [{"size": 4}], "layers.*.size": 1}, "layers.*.size"),
        ]
        for values, key in cases:
            with self.subTest(key=key):
                with self.assertRaises(ValueError) as cm:
                    flatten(values)
                self.assertIn(repr(key), str(cm.exception))

    def test_nesting_that_unflatten_cannot_rebuild(self):
        # "s" is None in one item and a mapping of None in the other, so
        # the flat keys would hold only None
        with self.assertRaisesRegex(ValueError, "'x.\\*.s.a'"):
            flatten({"x": [{"s": None}, {"s": {"a": None}}]})
        # when a value tells the forms apart, they are rebuilt
        values = {"x": [{"s": None}, {"s": {"a": 1}}]}
        self.assertEqual(unflatten(flatten(values)), values)
        # keys whose dots clash with the nesting
        with self.assertRaisesRegex(ValueError, "'optim' holds more than"):
            flatten({"optim": "adam", "optim.lr": 0.1})

    def test_nan(self):
        flat = flatten({"a": math.nan, "l": [{"v": math.nan}, {"v": 1.0}]})
        self.assertTrue(math.isnan(flat["a"]))
        self.assertTrue(math.isnan(flat["l.*.v"][0]))


class TestUnflatten(unittest.TestCase):
    def test_round_trip(self):
        roc = ROCPoints(np.array([0.0, 1.0]), np.array([0.5, 1.0]), None)
        cases = [
            {"optim": {"lr": 0.1, "adam": {"eps": 1e-8}}, "hidden": (8, 4)},
            {"layers": [{"size": 4}, {"size": 8, "act": "relu"}]},
            {"blocks": [{"layers": [{"k": 1}, {"k": 2}]}, {"layers": []}]},
            {"blocks": [{"layers": [{"k": 1}]}, {"other": 5}]},
            {"items": [{"a": 1}, {"a": {"b": 2}}]},
            {"items": [{"b": {"b": 2.5}}, {"a": 1}, {"b": [{"b": (1, 2)}]}]},
            {"items": [{}, {"a": None}, {"a": [{"b": None}]}]},
            # empty mappings, also as the only item of a list
            {"sched": {}, "q": [[{}], [], []], "mixed": [{}, 1]},
            {
                "items": [{}, {"a": None}],
                "deep": [{"s": {}}, {"s": {"a": None}}],
            },
            # an optional sub-config: None in one item, a mapping in another
            {"items": [{"s": None}, {"s": {"a": 1}}]},
            # what an item lacked, judged within each list
            {"c": [[{"a": True}, {}], [], [{}, {"a": None}]]},
            # a mapping that holds only a list the item lacked
            {"p": [{"a": {"l": [{"b": 1}]}, "c": 1}, {"c": 2}]},
            # an item that lacks a whole list, next to an empty list
            {"p": [{"v": [{"v": []}]}, {"v": [{}, {"v": [{"v": True}]}]}]},
            {"nested": [[{"a": 1}], None, [{"a": 2}, {"a": 3}]]},
            {"roc": [roc, roc]},
            {"mixed": [1, {"a": 2}]},
            {"mixed": [None, {"a": 2, "b": None}]},
            {"cm": np.eye(2, dtype=int), "empty": (), "none": None},
        ]
        for values in cases:
            with self.subTest(values=values):
                flat = flatten(values)
                self.assertEqual(flatten(unflatten(flat)), flat)

    def test_nesting(self):
        flat = {
            "optim.lr": 0.1,
            "optim.betas": (0.9, 0.99),
            "layers.*.size": (4, 8),
            "layers.*.act": (None, "relu"),
            "blocks.*.layers.*.k": ((1, 2), (3,)),
            "mixed.*": (1, None),
            "mixed.*.a": (None, 2),
        }
        self.assertEqual(
            unflatten(flat),
            {
                "optim": {"lr": 0.1, "betas": (0.9, 0.99)},
                "layers": [
                    {"size": 4, "act": None},
                    {"size": 8, "act": "relu"},
                ],
                "blocks": [
                    {"layers": [{"k": 1}, {"k": 2}]},
                    {"layers": [{"k": 3}]},
                ],
                "mixed": [1, {"a": 2}],
            },
        )

    def test_what_flatten_does_not_record(self):
        roc = ROCPoints((0.0, 1.0), (0.5, 1.0), None)
        values = {
            "roc": roc,
            "hidden": [8, 4],
            "layers": [{"size": 4}, {"size": 8, "act": "relu"}],
        }
        self.assertEqual(
            unflatten(flatten(values)),
            {
                # named tuples become dicts
                "roc": {
                    "fpr": (0.0, 1.0),
                    "tpr": (0.5, 1.0),
                    "thresholds": None,
                },
                # lists of values stay tuples
                "hidden": (8, 4),
                # a key an item lacked becomes None
                "layers": [
                    {"size": 4, "act": None},
                    {"size": 8, "act": "relu"},
                ],
            },
        )

    def test_json_lists(self):
        # as read from run.json, with lists instead of tuples
        flat = {"layers.*.size": [4, 8], "hidden": [8, 4]}
        self.assertEqual(
            unflatten(flat),
            {"layers": [{"size": 4}, {"size": 8}], "hidden": [8, 4]},
        )

    def test_returns_plain_dicts(self):
        nested = unflatten(flatten({"sched": {}, "items": [{}]}))
        self.assertEqual(nested, {"sched": {}, "items": [{}]})
        nested["sched"]["name"] = "cosine"  # not the read-only {}
        self.assertEqual(type(nested["items"][0]), dict)

    def test_nan(self):
        # e.g. a value that pandas fills in for a missing one
        flat = {"a": np.float64("nan"), "b.c": math.nan, "l.*.v": (math.nan,)}
        nested = unflatten(flat)
        self.assertTrue(math.isnan(nested["a"]))
        self.assertTrue(math.isnan(nested["b"]["c"]))
        self.assertTrue(math.isnan(nested["l"][0]["v"]))

    def test_keys_that_do_not_nest(self):
        cases = [
            ({"a": 1, "a.b": 2}, "'a' holds more than one of"),
            ({"a.*.b": (1, 2), "a.c": 1}, "'a' holds more than one of"),
            ({"a.*.b": (1, 2), "a.*.c": (1,)}, "tuples of one length"),
            ({"a.*.b": 3}, "tuples of one length"),
            (
                {"a.*": (1, None), "a.*.b": (2, None)},
                "'a.*' holds more than one of",
            ),
        ]
        for flat, message in cases:
            with self.subTest(flat=flat):
                with self.assertRaises(ValueError) as cm:
                    unflatten(flat)
                self.assertIn(message, str(cm.exception))


if __name__ == "__main__":
    unittest.main()
