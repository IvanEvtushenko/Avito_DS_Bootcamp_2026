"""Юнит-проверка метрики на ручных примерах (план §4, этап 1).

Метрика — то, во что упирается весь подбор, поэтому AP@k проверяется на числах,
посчитанных руками, а не «на глаз».
"""
from __future__ import annotations

import pathlib
import sys
import unittest

_SOL = pathlib.Path(__file__).resolve().parents[2]
for _p in (_SOL / "src", _SOL / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from support_search.eval.metrics import (  # noqa: E402
    average_precision_at_k,
    mean_average_precision_at_k,
    recall_at_k,
)


class TestAveragePrecision(unittest.TestCase):
    def test_two_relevant_with_gap(self):
        # relevant={1,2}, predicted=[1,9,2]: prec@1=1, prec@3=2/3 → (1+2/3)/2.
        ap = average_precision_at_k([1, 9, 2, 8, 7], {1, 2}, k=10)
        self.assertAlmostEqual(ap, (1.0 + 2.0 / 3.0) / 2.0, places=6)

    def test_single_relevant_second_position(self):
        # relevant={1}, predicted=[9,1]: prec@2=1/2, нормировка min(1,10)=1.
        self.assertAlmostEqual(average_precision_at_k([9, 1, 3], {1}, k=10), 0.5, places=6)

    def test_perfect_ranking(self):
        self.assertAlmostEqual(average_precision_at_k([1, 2, 3], {1, 2}, k=10), 1.0, places=6)

    def test_none_relevant_in_topk(self):
        self.assertEqual(average_precision_at_k([9, 8, 7], {1}, k=10), 0.0)

    def test_empty_relevant_is_zero(self):
        self.assertEqual(average_precision_at_k([1, 2], set(), k=10), 0.0)

    def test_k_truncation_normalizes_by_min(self):
        # relevant={1,2,3}, k=2, оба верхних релевантны → (1 + 1)/min(3,2)=1.0.
        self.assertAlmostEqual(average_precision_at_k([1, 2, 3], {1, 2, 3}, k=2), 1.0, places=6)

    def test_relevant_below_cutoff_ignored(self):
        # relevant={99}, но 99 стоит на позиции 11 → вне k=10 → 0.
        preds = list(range(1, 11)) + [99]
        self.assertEqual(average_precision_at_k(preds, {99}, k=10), 0.0)

    def test_duplicate_predictions_counted_once(self):
        # Повтор 1 не должен добавлять второй «хит».
        ap = average_precision_at_k([1, 1, 2], {1, 2}, k=10)
        self.assertAlmostEqual(ap, (1.0 + 2.0 / 3.0) / 2.0, places=6)


class TestRecall(unittest.TestCase):
    def test_recall_partial(self):
        self.assertAlmostEqual(recall_at_k([1, 9, 8], {1, 2}, k=3), 0.5, places=6)

    def test_recall_full(self):
        self.assertAlmostEqual(recall_at_k([1, 2, 9], {1, 2}, k=3), 1.0, places=6)

    def test_recall_cutoff(self):
        self.assertAlmostEqual(recall_at_k([9, 1, 2], {1, 2}, k=1), 0.0, places=6)


class TestMeanAveragePrecision(unittest.TestCase):
    def test_map_is_average_over_queries(self):
        predictions = {1: [1, 9, 2], 2: [9, 1]}
        ground_truth = {1: {1, 2}, 2: {1}}
        expected = ((1.0 + 2.0 / 3.0) / 2.0 + 0.5) / 2.0
        self.assertAlmostEqual(
            mean_average_precision_at_k(predictions, ground_truth, k=10), expected, places=6
        )

    def test_missing_prediction_scores_zero(self):
        # У запроса 2 нет предсказания → его AP=0, MAP = (1.0 + 0)/2.
        predictions = {1: [1]}
        ground_truth = {1: {1}, 2: {5}}
        self.assertAlmostEqual(mean_average_precision_at_k(predictions, ground_truth, k=10), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
