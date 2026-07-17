"""Финальный блэнд: кандидаты выше не-кандидатов, поиск веса, OOF."""
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
from support_search.ranking import blend_matrix, oof_blend, search_blend_weight  # noqa: E402
from support_search.rerank import NON_CANDIDATE  # noqa: E402


def _reranker(scores, source="reranker", qids=(1,), aids=(10, 20, 30)):
    return ScoreMatrix(list(qids), list(aids), np.asarray(scores, dtype=np.float32), source)


class TestBlendMatrix(unittest.TestCase):
    def test_candidates_rank_above_non_candidates(self):
        fusion = _reranker([[0.9, 0.8, 0.1]], "fusion")           # 30 — не-кандидат по реранкеру
        reranker = _reranker([[0.2, 5.0, NON_CANDIDATE]])          # кандидаты 10,20; 30 — сентинел
        blended = blend_matrix(fusion, reranker, weight=0.5)
        ranking = blended.rankings(3)[1]
        self.assertEqual(set(ranking[:2]), {10, 20})               # кандидаты первыми
        self.assertEqual(ranking[2], 30)                           # не-кандидат последним

    def test_weight_one_recovers_fusion_order(self):
        fusion = _reranker([[0.1, 0.9, NON_CANDIDATE]], "fusion")  # среди кандидатов 20 > 10
        reranker = _reranker([[9.0, 0.0, NON_CANDIDATE]])          # реранкер предпочёл бы 10
        blended = blend_matrix(fusion, reranker, weight=1.0)       # только fusion
        self.assertEqual(blended.rankings(2)[1][0], 20)


class TestSearchWeight(unittest.TestCase):
    def test_search_prefers_reranker_when_it_is_right(self):
        gt = {1: {10}, 2: {30}}
        # fusion ставит GT низко, реранкер — высоко → вес должен уйти к реранкеру (w<0.5).
        fusion = ScoreMatrix([1, 2], [10, 20, 30],
                             np.array([[0.1, 0.9, 0.8], [0.9, 0.8, 0.1]], np.float32), "fusion")
        reranker = ScoreMatrix([1, 2], [10, 20, 30],
                               np.array([[5.0, 0.1, 0.2], [0.1, 0.2, 5.0]], np.float32), "reranker")
        w = search_blend_weight(fusion, reranker, gt, [1, 2], k=10, grid=51)
        self.assertLess(w, 0.5)


class TestOOFBlend(unittest.TestCase):
    def test_oof_covers_dev(self):
        n = 20
        qids = list(range(1, n + 1))
        aids = [10, 20, 30, 40]
        rng = np.random.default_rng(0)
        fusion = ScoreMatrix(qids, aids, rng.random((n, 4)).astype(np.float32), "fusion")
        reranker = ScoreMatrix(qids, aids, rng.random((n, 4)).astype(np.float32), "reranker")
        gt = {q: {aids[q % 4]} for q in qids}
        splits = make_splits({q: 1 for q in qids}, n_splits=5, holdout_frac=0.2, seed=0)
        oof, fold_weights = oof_blend(fusion, reranker, gt, splits, k=10, depth=4, grid=11)
        self.assertEqual(set(oof), set(splits.dev))
        self.assertEqual(len(fold_weights), splits.n_splits)


if __name__ == "__main__":
    unittest.main()
