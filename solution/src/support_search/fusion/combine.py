"""Операторы слияния матриц скоров.

BM25 и эмбеддинги живут в разных шкалах, поэтому перед сложением скоры каждого
источника нормируются **по запросу** (min-max в [0,1]) — тогда вес отражает
вклад источника, а не его случайный масштаб. RRF — безпараметрическая
альтернатива, работающая с рангами, а не значениями.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from ..contracts import ScoreMatrix

FUSION_SOURCE = "fusion"


def align_matrices(matrices: Mapping[str, ScoreMatrix]) -> tuple[np.ndarray, np.ndarray]:
    """Проверить, что все матрицы согласованы по запросам и статьям.

    Возвращает общие (query_ids, article_ids). Fusion — арифметика по позициям,
    поэтому порядок строк/столбцов обязан совпадать у всех источников.
    """
    if not matrices:
        raise ValueError("нет матриц для слияния")
    items = list(matrices.values())
    query_ids, article_ids = items[0].query_ids, items[0].article_ids
    for name, sm in matrices.items():
        if not np.array_equal(sm.query_ids, query_ids):
            raise ValueError(f"[{name}] другой порядок query_ids")
        if not np.array_equal(sm.article_ids, article_ids):
            raise ValueError(f"[{name}] другой порядок article_ids")
    return query_ids, article_ids


def minmax_rows(scores: np.ndarray) -> np.ndarray:
    """Min-max нормализация каждой строки (запроса) в [0,1].

    Константная строка (нет сигнала) → нули, что корректно не влияет на ранжирование.
    """
    lo = scores.min(axis=1, keepdims=True)
    hi = scores.max(axis=1, keepdims=True)
    span = hi - lo
    span[span == 0] = 1.0
    return (scores - lo) / span


def weighted_sum(
    matrices: Mapping[str, ScoreMatrix],
    weights: Mapping[str, float],
    *,
    normalize: bool = True,
) -> ScoreMatrix:
    """Взвешенная сумма (по умолчанию — по min-max-нормированным скорам)."""
    query_ids, article_ids = align_matrices(matrices)
    combined = np.zeros((len(query_ids), len(article_ids)), dtype=np.float32)
    for name, sm in matrices.items():
        values = minmax_rows(sm.scores) if normalize else sm.scores
        combined += float(weights.get(name, 0.0)) * values
    return ScoreMatrix(query_ids=query_ids, article_ids=article_ids, scores=combined, source=FUSION_SOURCE)


def reciprocal_rank_fusion(matrices: Mapping[str, ScoreMatrix], *, k: int = 60) -> ScoreMatrix:
    """RRF: скор статьи = Σ_источников 1 / (k + ранг). Без подбираемых весов.

    Ранг 1-based по убыванию скора источника; ничьи разрешаются порядком столбцов
    (детерминированно). Классический k=60.
    """
    query_ids, article_ids = align_matrices(matrices)
    q, a = len(query_ids), len(article_ids)
    combined = np.zeros((q, a), dtype=np.float32)
    rows = np.arange(q)[:, None]
    for sm in matrices.values():
        order = np.argsort(-sm.scores, axis=1, kind="stable")  # индексы статей по убыванию
        ranks = np.empty((q, a), dtype=np.int64)
        ranks[rows, order] = np.arange(a)[None, :]              # 0-based ранг каждой статьи
        combined += (1.0 / (k + ranks + 1.0)).astype(np.float32)
    return ScoreMatrix(query_ids=query_ids, article_ids=article_ids, scores=combined, source=FUSION_SOURCE)
