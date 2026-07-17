"""Совместный random search весов слияния по честной OOF-схеме (план §4.2–4.3).

Веса fusion — дешёвые параметры: один конфиг оценивается за миллисекунды поверх
кэшированных матриц скоров. Поэтому вместо жадного поэтапного тюнинга — совместный
случайный поиск сразу по всем весам.

Честность (§4.2): для фолда f веса подбираются на train(f) и применяются к val(f);
собранные по всем фолдам val-предсказания дают OOF-оценку, где ни один вес не
подгонялся под оцениваемый запрос. Для финала (предсказания на test) веса
подбираются на всех dev.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..contracts import ScoreMatrix, rank_indices
from ..data.splits import Splits
from ..eval.metrics import average_precision_at_k
from .combine import align_matrices, minmax_rows, reciprocal_rank_fusion


def _weight_samples(n_sources: int, n_samples: int, seed: int) -> np.ndarray:
    """Кандидаты весов на симплексе [M, S]: оси (по одному источнику), равные веса
    и Dirichlet-сэмплы. Оси гарантируют, что «только один источник» рассмотрен."""
    rng = np.random.default_rng(seed)
    base = [np.eye(n_sources), np.full((1, n_sources), 1.0 / n_sources)]
    if n_samples > 0:
        base.append(rng.dirichlet(np.ones(n_sources), size=n_samples))
    return np.vstack(base).astype(np.float32)


def _map_from_combined(
    combined: np.ndarray,
    query_ids: Sequence[int],
    article_ids: np.ndarray,
    ground_truth: Mapping[int, set[int]],
    k: int,
) -> float:
    """MAP@k по объединённой матрице [n_queries, n_articles] на данном подмножестве."""
    total, n = 0.0, 0
    for i, qid in enumerate(query_ids):
        relevant = ground_truth.get(int(qid))
        if not relevant:
            continue
        idx = rank_indices(combined[i], article_ids, k)
        total += average_precision_at_k(article_ids[idx].tolist(), relevant, k)
        n += 1
    return total / n if n else 0.0


def _search_on_stack(
    stack: np.ndarray,
    rows: list[int],
    subset_ids: Sequence[int],
    article_ids: np.ndarray,
    ground_truth: Mapping[int, set[int]],
    names: Sequence[str],
    *,
    n_samples: int,
    k: int,
    seed: int,
) -> dict[str, float]:
    """Перебрать веса на подмножестве строк, вернуть лучший вектор (по MAP@k)."""
    sub = stack[:, rows, :]  # [S, |subset|, A]
    best_weights = None
    best_map = -1.0
    for weight_vec in _weight_samples(len(names), n_samples, seed):
        combined = np.tensordot(weight_vec, sub, axes=1)  # [|subset|, A]
        score = _map_from_combined(combined, subset_ids, article_ids, ground_truth, k)
        if score > best_map:
            best_map, best_weights = score, weight_vec
    return {name: float(best_weights[i]) for i, name in enumerate(names)}


def search_weights(
    matrices: Mapping[str, ScoreMatrix],
    ground_truth: Mapping[int, set[int]],
    query_ids_subset: Sequence[int],
    *,
    names: Sequence[str],
    n_samples: int = 500,
    k: int = 10,
    seed: int = 42,
) -> dict[str, float]:
    """Подобрать веса на заданном подмножестве запросов (напр. все dev — для теста)."""
    all_query_ids, article_ids = align_matrices(matrices)
    row_of = {int(q): i for i, q in enumerate(all_query_ids)}
    stack = np.stack([minmax_rows(matrices[n].scores) for n in names])
    rows = [row_of[int(q)] for q in query_ids_subset]
    return _search_on_stack(
        stack, rows, query_ids_subset, article_ids, ground_truth, names,
        n_samples=n_samples, k=k, seed=seed,
    )


def oof_fusion_search(
    matrices: Mapping[str, ScoreMatrix],
    ground_truth: Mapping[int, set[int]],
    splits: Splits,
    *,
    names: Sequence[str],
    method: str = "weighted_sum",
    n_samples: int = 500,
    k: int = 10,
    rrf_k: int = 60,
    seed: int = 42,
    depth: int = 100,
) -> tuple[dict[int, list[int]], dict[int, dict[str, float]]]:
    """OOF-ранжирования dev-запросов + веса по фолдам.

    weighted_sum: на каждом фолде веса ищутся на train(f), применяются к val(f).
    rrf: параметров нет — просто RRF на каждом запросе. Возвращает
    (oof_rankings: query_id → топ-`depth` article_id, fold_weights).
    """
    all_query_ids, article_ids = align_matrices(matrices)
    row_of = {int(q): i for i, q in enumerate(all_query_ids)}

    if method == "rrf":
        rankings = reciprocal_rank_fusion(matrices, k=rrf_k).rankings(depth)
        return {q: rankings[q] for q in splits.dev}, {}

    stack = np.stack([minmax_rows(matrices[n].scores) for n in names])
    oof_rankings: dict[int, list[int]] = {}
    fold_weights: dict[int, dict[str, float]] = {}
    for fold in range(splits.n_splits):
        weights = _search_on_stack(
            stack, [row_of[q] for q in splits.train_ids(fold)], splits.train_ids(fold),
            article_ids, ground_truth, names, n_samples=n_samples, k=k, seed=seed + fold,
        )
        fold_weights[fold] = weights
        weight_vec = np.array([weights[n] for n in names], dtype=np.float32)
        val_rows = [row_of[q] for q in splits.fold(fold)]
        combined = np.tensordot(weight_vec, stack[:, val_rows, :], axes=1)  # [|val|, A]
        for i, qid in enumerate(splits.fold(fold)):
            idx = rank_indices(combined[i], article_ids, depth)
            oof_rankings[qid] = article_ids[idx].tolist()
    return oof_rankings, fold_weights
