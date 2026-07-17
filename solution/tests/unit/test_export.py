"""Экспорт: сборка answer из ранжирований + соответствие контракту."""
from __future__ import annotations

import pathlib
import sys
import unittest

_SOL = pathlib.Path(__file__).resolve().parents[2]
for _p in (_SOL / "src", _SOL / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from support_search.contracts import ANSWER_COLUMNS, check_answer_frame  # noqa: E402
from support_search.export import build_answer  # noqa: E402


class TestBuildAnswer(unittest.TestCase):
    def test_columns_and_order(self):
        rankings = {1: [10, 20, 30], 2: [20, 10]}
        answer = build_answer(rankings, [1, 2], top_k=10)
        self.assertEqual(list(answer.columns), ANSWER_COLUMNS)
        self.assertEqual(answer["query_id"].tolist(), [1, 2])
        self.assertEqual(answer.iloc[0]["answer"], "10 20 30")

    def test_top_k_truncation(self):
        rankings = {1: list(range(1, 20))}
        answer = build_answer(rankings, [1], top_k=10)
        self.assertEqual(len(answer.iloc[0]["answer"].split()), 10)

    def test_passes_contract(self):
        rankings = {1: [10, 20], 2: [30]}
        answer = build_answer(rankings, [1, 2], top_k=10)
        check_answer_frame(
            answer, expected_query_ids=[1, 2], valid_article_ids=[10, 20, 30], max_k=10
        )


if __name__ == "__main__":
    unittest.main()
