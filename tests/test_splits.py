import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from splits import make_grouped_split, class_report, save_split, load_split  # noqa: E402


def _toy(n_groups=200, seed=1):
    rng = np.random.default_rng(seed)
    groups, targets = [], []
    for g in range(n_groups):
        size = rng.choice([3, 6, 12])          # maps per group, like projections x sigmas
        cls = rng.integers(0, 4)               # each group is one class (DARKSKIES-like)
        groups += [f"g{g}"] * size
        targets += [cls] * size
    return np.array(groups), np.array(targets)


class TestGroupedSplit(unittest.TestCase):
    def setUp(self):
        self.groups, self.targets = _toy()
        self.split = make_grouped_split(self.groups, self.targets, seed=0)

    def test_partition_covers_everything_once(self):
        allidx = np.concatenate([self.split[k] for k in ("train", "valid", "test")])
        self.assertEqual(len(allidx), len(self.groups))
        self.assertEqual(len(np.unique(allidx)), len(self.groups))

    def test_no_group_in_two_splits(self):
        seen = {}
        for name in ("train", "valid", "test"):
            for g in np.unique(self.groups[self.split[name]]):
                self.assertNotIn(g, seen, f"group {g} in {seen.get(g)} and {name}")
                seen[g] = name

    def test_fractions_and_class_balance(self):
        n = len(self.groups)
        for name, frac in (("train", 0.70), ("valid", 0.15), ("test", 0.15)):
            self.assertAlmostEqual(len(self.split[name]) / n, frac, delta=0.05)
        rep = class_report(self.split, self.targets)
        for c in np.unique(self.targets):
            total = (self.targets == c).sum()
            for name, frac in (("train", 0.70), ("valid", 0.15), ("test", 0.15)):
                self.assertAlmostEqual(rep[name].get(int(c), 0) / total, frac, delta=0.08)

    def test_deterministic_for_same_seed(self):
        again = make_grouped_split(self.groups, self.targets, seed=0)
        for k in self.split:
            np.testing.assert_array_equal(self.split[k], again[k])

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "split.json")
            save_split(p, self.split, {"seed": 0, "fractions": [0.7, 0.15, 0.15]})
            loaded, meta = load_split(p)
            self.assertEqual(meta["seed"], 0)
            for k in self.split:
                np.testing.assert_array_equal(self.split[k], loaded[k])


if __name__ == "__main__":
    unittest.main()
