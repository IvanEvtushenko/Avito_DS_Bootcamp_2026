"""Контракты: детерминированное ранжирование и валидация таблиц/матриц."""
from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np
import pandas as pd

_SOL = pathlib.Path(__file__).resolve().parents[2]
for _p in (_SOL / "src", _SOL / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from support_search.contracts import (  # noqa: E402
    ContractError,
    ScoreMatrix,
    check_answer_frame,
    check_retrieval_frame,
    check_score_matrix,
    rank_indices,
)


class TestRankIndices(unittest.TestCase):
    def test_ties_broken_by_article_id(self):
        scores = np.array([0.1, 0.5, 0.5, 0.2])
        article_ids = np.array([10, 20, 30, 40])
        idx = rank_indices(scores, article_ids, k=3)
        # 0.5-ничья: меньший article_id (20) раньше большего (30).
        self.assertEqual(article_ids[idx].tolist(), [20, 30, 40])

    def test_deterministic(self):
        scores = np.array([0.3, 0.3, 0.3])
        article_ids = np.array([7, 3, 5])
        a = rank_indices(scores, article_ids, k=3)
        b = rank_indices(scores, article_ids, k=3)
        self.assertEqual(a.tolist(), b.tolist())
        self.assertEqual(article_ids[a].tolist(), [3, 5, 7])

    def test_k_larger_than_n(self):
        self.assertEqual(len(rank_indices(np.array([1.0, 2.0]), np.array([1, 2]), k=10)), 2)


class TestScoreMatrix(unittest.TestCase):
    def _matrix(self) -> ScoreMatrix:
        scores = np.array([[0.1, 0.9, 0.3], [0.5, 0.2, 0.8]], dtype=np.float32)
        return ScoreMatrix(query_ids=[1, 2], article_ids=[100, 200, 300], scores=scores, source="test")

    def test_rankings(self):
        r = self._matrix().rankings(k=2)
        self.assertEqual(r[1], [200, 300])
        self.assertEqual(r[2], [300, 100])

    def test_retrieval_frame_shape_and_ranks(self):
        frame = self._matrix().to_retrieval_frame(k=2)
        self.assertEqual(len(frame), 4)
        self.assertEqual(sorted(frame.columns), sorted(["query_id", "article_id", "score", "rank", "source"]))
        self.assertEqual(frame[frame.query_id == 1]["rank"].tolist(), [0, 1])

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ScoreMatrix(query_ids=[1], article_ids=[1, 2], scores=np.zeros((2, 2), dtype=np.float32), source="x")


class TestChecks(unittest.TestCase):
    def test_score_matrix_nan_raises(self):
        sm = ScoreMatrix([1], [10, 20], np.array([[np.nan, 0.0]], dtype=np.float32), "s")
        with self.assertRaises(ContractError):
            check_score_matrix(sm, [10, 20])

    def test_score_matrix_unknown_article_raises(self):
        sm = ScoreMatrix([1], [10, 99], np.array([[0.1, 0.2]], dtype=np.float32), "s")
        with self.assertRaises(ContractError):
            check_score_matrix(sm, [10, 20])

    def test_retrieval_frame_duplicate_pair_raises(self):
        df = pd.DataFrame(
            {"query_id": [1, 1], "article_id": [10, 10], "score": [0.5, 0.4], "rank": [0, 1], "source": "s"}
        )
        with self.assertRaises(ContractError):
            check_retrieval_frame(df, [10, 20])

    def test_retrieval_frame_bad_ranks_raise(self):
        df = pd.DataFrame(
            {"query_id": [1, 1], "article_id": [10, 20], "score": [0.5, 0.4], "rank": [0, 2], "source": "s"}
        )
        with self.assertRaises(ContractError):
            check_retrieval_frame(df, [10, 20])

    def test_answer_frame_happy_path(self):
        df = pd.DataFrame({"query_id": [1, 2], "answer": ["10 20", "20"]})
        check_answer_frame(df, expected_query_ids=[1, 2], valid_article_ids=[10, 20], max_k=10)

    def test_answer_frame_missing_query_raises(self):
        df = pd.DataFrame({"query_id": [1], "answer": ["10"]})
        with self.assertRaises(ContractError):
            check_answer_frame(df, expected_query_ids=[1, 2], valid_article_ids=[10], max_k=10)

    def test_answer_frame_too_many_ids_raises(self):
        df = pd.DataFrame({"query_id": [1], "answer": ["1 2 3"]})
        with self.assertRaises(ContractError):
            check_answer_frame(df, expected_query_ids=[1], valid_article_ids=[1, 2, 3], max_k=2)

    def test_answer_frame_duplicate_article_raises(self):
        df = pd.DataFrame({"query_id": [1], "answer": ["10 10"]})
        with self.assertRaises(ContractError):
            check_answer_frame(df, expected_query_ids=[1], valid_article_ids=[10], max_k=10)

    def test_answer_frame_unknown_article_raises(self):
        df = pd.DataFrame({"query_id": [1], "answer": ["99"]})
        with self.assertRaises(ContractError):
            check_answer_frame(df, expected_query_ids=[1], valid_article_ids=[10], max_k=10)


if __name__ == "__main__":
    unittest.main()
