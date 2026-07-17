"""Разбиение dev/holdout + 5-fold: непересечение, покрытие, детерминизм."""
from __future__ import annotations

import pathlib
import sys
import unittest

_SOL = pathlib.Path(__file__).resolve().parents[2]
for _p in (_SOL / "src", _SOL / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from support_search.data.splits import Splits, make_splits, stratum_of  # noqa: E402


class TestStratum(unittest.TestCase):
    def test_buckets(self):
        self.assertEqual(stratum_of(1), "1")
        self.assertEqual(stratum_of(2), "2")
        self.assertEqual(stratum_of(4), "3+")


class TestMakeSplits(unittest.TestCase):
    def setUp(self):
        # 100 запросов: 60 с одной GT, 30 с двумя, 10 с тремя.
        counts = {}
        qid = 1
        for _ in range(60):
            counts[qid] = 1; qid += 1
        for _ in range(30):
            counts[qid] = 2; qid += 1
        for _ in range(10):
            counts[qid] = 3; qid += 1
        self.counts = counts
        self.splits = make_splits(counts, n_splits=5, holdout_frac=0.2, seed=42)

    def test_disjoint_and_cover(self):
        holdout = set(self.splits.holdout)
        dev = set(self.splits.dev)
        self.assertEqual(holdout & dev, set())
        self.assertEqual(holdout | dev, set(self.counts))

    def test_holdout_size(self):
        # 20% от каждого страта: 12 + 6 + 2 = 20.
        self.assertEqual(len(self.splits.holdout), 20)
        self.assertEqual(len(self.splits.dev), 80)

    def test_folds_partition_dev(self):
        all_fold_ids = []
        for i in range(self.splits.n_splits):
            all_fold_ids.extend(self.splits.fold(i))
        self.assertEqual(sorted(all_fold_ids), sorted(self.splits.dev))
        # train(f) и val(f) не пересекаются и вместе дают dev.
        for i in range(self.splits.n_splits):
            self.assertEqual(set(self.splits.fold(i)) & set(self.splits.train_ids(i)), set())
            self.assertEqual(set(self.splits.fold(i)) | set(self.splits.train_ids(i)), set(self.splits.dev))

    def test_deterministic(self):
        again = make_splits(self.counts, n_splits=5, holdout_frac=0.2, seed=42)
        self.assertEqual(again.holdout, self.splits.holdout)
        self.assertEqual(again.fold_of, self.splits.fold_of)

    def test_json_roundtrip(self):
        restored = Splits.from_json(self.splits.to_json())
        self.assertEqual(restored.holdout, self.splits.holdout)
        self.assertEqual(restored.fold_of, self.splits.fold_of)


if __name__ == "__main__":
    unittest.main()
