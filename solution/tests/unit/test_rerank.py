"""Реранкер на заглушке: применение к кандидатам, майнинг негативов."""
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
from support_search.rerank import (  # noqa: E402
    NON_CANDIDATE,
    LexicalStubReranker,
    build_training_groups,
    candidate_lists,
    rerank_to_matrix,
)


class TestStubReranker(unittest.TestCase):
    def test_scores_word_overlap(self):
        r = LexicalStubReranker()
        s = r.score_pairs(["вернуть деньги", "вернуть деньги"], ["как вернуть деньги", "профиль блокировка"])
        self.assertGreater(s[0], s[1])


class TestCandidatesAndMatrix(unittest.TestCase):
    def test_candidate_lists_order(self):
        fusion = ScoreMatrix([1, 2], [10, 20, 30], np.array([[0.1, 0.9, 0.5], [0.8, 0.2, 0.7]], np.float32), "fusion")
        cands = candidate_lists(fusion, top_k=2)
        self.assertEqual(cands[0], [20, 30])
        self.assertEqual(cands[1], [10, 30])

    def test_rerank_to_matrix_scatters_and_orders(self):
        fusion = ScoreMatrix([1], [10, 20, 30], np.array([[0.1, 0.9, 0.5]], np.float32), "fusion")
        cands = candidate_lists(fusion, top_k=2)          # [20, 30]
        passages = [["инструкция про доставку товара", "как вернуть деньги за заказ"]]
        matrix = rerank_to_matrix(
            LexicalStubReranker(), ["вернуть деньги"], [1], fusion.article_ids, cands, passages, source="reranker_zs"
        )
        # Не-кандидат (10) остаётся сентинелом (float32 округляет -1e30).
        self.assertLess(matrix.scores[0, 0], -1e29)
        # Среди кандидатов первым — статья, чей пассаж совпадает с запросом (30).
        self.assertEqual(matrix.rankings(2)[1][0], 30)


class TestNegatives(unittest.TestCase):
    def test_groups_fixed_size_and_positive(self):
        query_ids = [1]
        gt = {1: {10}}
        candidates = {1: [10, 20, 30, 40]}          # 10 — позитив, остальные негативы
        passages = {(1, 10): "pos", (1, 20): "n1", (1, 30): "n2", (1, 40): "n3"}
        groups = build_training_groups(
            query_ids, {1: "запрос"}, gt, candidates, passages, n_negatives=2
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].positive, "pos")
        self.assertEqual(len(groups[0].negatives), 2)
        self.assertNotIn("pos", groups[0].negatives)

    def test_group_skipped_if_too_few_negatives(self):
        groups = build_training_groups(
            [1], {1: "q"}, {1: {10}}, {1: [10, 20]}, {(1, 10): "p", (1, 20): "n"}, n_negatives=5
        )
        self.assertEqual(groups, [])


if __name__ == "__main__":
    unittest.main()
