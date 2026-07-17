"""Оценка неопределённости и значимости различий.

На 500 запросах разница в третьем знаке MAP@10 легко объясняется шумом. Поэтому
каждый репортуемый MAP@10 сопровождается **bootstrap 95% CI** (по запросам), а
ключевые сравнения (BM25 → hybrid, hybrid → rerank) проверяются **парным
permutation-тестом** (план §4.4). Обе процедуры работают с массивом per-query
метрик (AP@10 на запрос), поэтому одинаково применимы к любой стадии.
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(
    per_query_scores: np.ndarray,
    *,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap-доверительный интервал среднего (MAP@k) по запросам.

    Ресемплируем индексы запросов с возвращением `n_resamples` раз, каждый раз
    считаем среднее, берём процентильный интервал. Возвращает (low, high).
    """
    scores = np.asarray(per_query_scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    # [n_resamples, n] индексов за один вызов — векторно и детерминированно.
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = scores[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return (float(low), float(high))


def paired_permutation_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> float:
    """Парный permutation-тест на равенство средних per-query метрик.

    Наблюдаемая статистика — средняя разность `mean(a - b)`. Под нулевой
    гипотезой знак разности на каждом запросе можно менять независимо; случайно
    инвертируем знаки `n_resamples` раз и смотрим, как часто |стат| ≥ наблюдаемой.
    Возвращает двусторонний p-value. Метод парный: `a` и `b` должны быть выровнены
    по одним и тем же запросам в одном порядке.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"несогласованные формы: {a.shape} vs {b.shape}")
    diff = a - b
    n = len(diff)
    if n == 0:
        return 1.0
    observed = abs(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, n))
    perm_means = np.abs((signs * diff).mean(axis=1))
    # +1 в числителе и знаменателе — сглаживание (p-value не бывает ровно 0).
    return float((np.sum(perm_means >= observed) + 1) / (n_resamples + 1))
