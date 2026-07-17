"""Слияние: нормализация, weighted sum, RRF, поиск весов и честный OOF."""
from __future__ import annotations

import pathlib
import sys
import unittest

_SOL = pathlib.Path(__file__).resolve().parents[2]
for _p in (_SOL / "src", _SOL / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

from support_search.contracts import ScoreMatrix  # noqa: E402
from support_search.data.splits import make_splits  # noqa: E402
from support_search.fusion import (  # noqa: E402
    minmax_rows,
    oof_fusion_search,
    reciprocal_rank_fusion,
    search_weights,
    weighted_sum,
)


def _sm(scores, source, qids=(1, 2), aids=(10, 20, 30)):
    return ScoreMatrix(list(qids), list(aids), np.asarray(scores, dtype=np.float32), source)


class TestNormalize(unittest.TestCase):
    def test_minmax_rows(self):
        out = minmax_rows(np.array([[0.0, 5.0, 10.0], [2.0, 2.0, 2.0]]))
        self.assertEqual(out[0].tolist(), [0.0, 0.5, 1.0])
        self.assertEqual(out[1].tolist(), [0.0, 0.0, 0.0])  # константа → нули


class TestCombine(unittest.TestCase):
    def test_weighted_sum_source_and_shape(self):
        a = _sm([[0, 1, 2], [2, 1, 0]], "a")
        b = _sm([[2, 1, 0], [0, 1, 2]], "b")
        fused = weighted_sum({"a": a, "b": b}, {"a": 1.0, "b": 1.0})
        self.assertEqual(fused.source, "fusion")
        self.assertEqual(fused.shape, (2, 3))

    def test_align_mismatch_raises(self):
        a = _sm([[0, 1, 2], [2, 1, 0]], "a")
        bad = _sm([[0, 1, 2], [2, 1, 0]], "b", aids=(10, 20, 99))
        with self.assertRaises(ValueError):
            weighted_sum({"a": a, "b": bad}, {"a": 1.0, "b": 1.0})

    def test_rrf_prefers_consensus(self):
        # Статья 30 стоит первой у обоих источников → должна лидировать по RRF.
        a = _sm([[0.1, 0.2, 0.9]], "a", qids=(1,))
        b = _sm([[0.3, 0.1, 0.8]], "b", qids=(1,))
        fused = reciprocal_rank_fusion({"a": a, "b": b}, k=60)
        self.assertEqual(fused.rankings(3)[1][0], 30)


class TestSearchWeights(unittest.TestCase):
    def test_search_prefers_informative_source(self):
        gt = {1: {30}, 2: {10}}
        good = _sm([[0, 0, 1], [1, 0, 0]], "good")   # ставит GT наверх
        bad = _sm([[1, 0, 0], [0, 0, 1]], "bad")     # ставит GT вниз
        weights = search_weights({"good": good, "bad": bad}, gt, [1, 2],
                                 names=["good", "bad"], n_samples=200, k=10, seed=0)
        self.assertGreater(weights["good"], weights["bad"])


class TestOOF(unittest.TestCase):
    def test_oof_covers_dev_only(self):
        # 20 запросов, все с одной GT; проверяем, что OOF выдаёт ранжирования
        # ровно для dev-запросов и нужной глубины.
        n = 20
        qids = list(range(1, n + 1))
        aids = [10, 20, 30, 40]
        rng = np.random.default_rng(0)
        scores_a = rng.random((n, 4)).astype(np.float32)
        scores_b = rng.random((n, 4)).astype(np.float32)
        a = ScoreMatrix(qids, aids, scores_a, "a")
        b = ScoreMatrix(qids, aids, scores_b, "b")
        gt = {q: {aids[q % 4]} for q in qids}
        splits = make_splits({q: 1 for q in qids}, n_splits=5, holdout_frac=0.2, seed=0)

        oof, fold_weights, oof_matrix = oof_fusion_search(
            {"a": a, "b": b}, gt, splits, names=["a", "b"],
            method="weighted_sum", n_samples=50, k=10, recall_k=4, depth=4, seed=0,
        )
        self.assertEqual(set(oof), set(splits.dev))          # только dev
        self.assertEqual(len(fold_weights), splits.n_splits)  # веса на каждый фолд
        self.assertEqual(oof_matrix.shape, (n, 4))            # OOF-матрица (для calibration без утечки)
        for ranking in oof.values():
            self.assertLessEqual(len(ranking), 4)


if __name__ == "__main__":
    unittest.main()
