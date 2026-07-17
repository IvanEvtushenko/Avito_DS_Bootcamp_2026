"""Метрики ранжирования: AP@k, MAP@k, recall@k.

Основная метрика задачи — **MAP@10**. Для одного запроса считается AP@10, затем
среднее по запросам. Реализация намеренно простая и покрыта юнит-тестом на ручном
примере (план §4, этап 1): метрика — это то, во что упирается весь подбор, её
нельзя держать «на глаз».

Определение (стандартный AP@k для задачи с несколькими релевантными):

    AP@k = ( Σ_{i=1..k} precision@i · rel_i ) / min(|R|, k)

где rel_i = 1, если i-й предсказанный документ релевантен, а precision@i — доля
релевантных среди первых i. Нормировка на min(|R|, k) даёт AP=1.0, когда все
релевантные документы стоят в начале списка.
"""
from __future__ import annotations

from typing import Mapping, Sequence


def average_precision_at_k(
    predicted: Sequence[int],
    relevant: set[int],
    k: int = 10,
) -> float:
    """AP@k для одного запроса.

    Параметры
    ---------
    predicted : ранжированный список article_id (первый — самый релевантный).
    relevant  : множество правильных article_id (ground truth).
    k         : отсечка (10 в этой задаче).

    Пустое множество релевантных → 0.0 (запрос без разметки не участвует).
    Повторы в `predicted` учитываются один раз (защита от некорректного входа).
    """
    if not relevant:
        return 0.0

    hits = 0
    sum_precision = 0.0
    seen: set[int] = set()
    for i, doc_id in enumerate(predicted[:k]):
        if doc_id in seen:
            continue
        seen.add(doc_id)
        if doc_id in relevant:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / min(len(relevant), k)


def recall_at_k(predicted: Sequence[int], relevant: set[int], k: int) -> float:
    """Доля релевантных документов, попавших в топ-k. Потолок для реранкера."""
    if not relevant:
        return 0.0
    top = set(predicted[:k])
    return len(top & relevant) / len(relevant)


def mean_average_precision_at_k(
    predictions: Mapping[int, Sequence[int]],
    ground_truth: Mapping[int, set[int]],
    k: int = 10,
) -> float:
    """MAP@k — среднее AP@k по всем запросам из `ground_truth`."""
    if not ground_truth:
        return 0.0
    total = 0.0
    for qid, relevant in ground_truth.items():
        total += average_precision_at_k(predictions.get(qid, []), relevant, k)
    return total / len(ground_truth)


def per_query_ap(
    predictions: Mapping[int, Sequence[int]],
    ground_truth: Mapping[int, set[int]],
    k: int = 10,
) -> dict[int, float]:
    """AP@k по каждому запросу — сырьё для анализа ошибок и bootstrap CI."""
    return {
        qid: average_precision_at_k(predictions.get(qid, []), relevant, k)
        for qid, relevant in ground_truth.items()
    }
