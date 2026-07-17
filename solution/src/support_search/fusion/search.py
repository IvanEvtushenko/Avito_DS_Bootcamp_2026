"""Совместный random search весов слияния по честной OOF-схеме (план §4.2–4.3).

[LEGACY / ABLATION] С введением LR-слияния (`fusion/ltr.py`, план §7.2) этот
модуль — не дефолт, а ablation-базлайн (`fusion.method: weighted_sum`). Код не
удалён: `stage_fuse` сравнивает LR против random search по OOF, а RRF остаётся
безпараметрической альтернативой. Оставлено намеренно.


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
from ..eval.metrics import average_precision_at_k, recall_at_k
from .combine import FUSION_SOURCE, align_matrices, minmax_rows, reciprocal_rank_fusion


def _weight_samples(n_sources: int, n_samples: int, seed: int) -> np.ndarray:
    """Кандидаты весов на симплексе [M, S]: оси (по одному источнику), равные веса
    и Dirichlet-сэмплы. Оси гарантируют, что «только один источник» рассмотрен."""
    rng = np.random.default_rng(seed)
    base = [np.eye(n_sources), np.full((1, n_sources), 1.0 / n_sources)]
    if n_samples > 0:
        base.append(rng.dirichlet(np.ones(n_sources), size=n_samples))
    return np.vstack(base).astype(np.float32)


def _objective_from_combined(
    combined: np.ndarray,
    query_ids: Sequence[int],
    article_ids: np.ndarray,
    ground_truth: Mapping[int, set[int]],
    k: int,
    recall_k: int,
    objective: str = "recall",
) -> tuple[float, float]:
    """Цель подбора весов для КАНДИДАТОВ — лексикографический кортеж (main, tie).

    - `objective="recall"` (дефолт): (recall@recall_k, MAP@k). Слияние отбирает
      кандидатов, поэтому главное — не потерять релевантные в топ-K (потолок
      реранкера), а MAP@k — только тай-брейк.
    - `objective="map"`: (MAP@k, recall@recall_k) — максимизируем MAP@10 напрямую.

    Обе метрики считаются всегда; порядок в кортеже задаёт, что главное.
    """
    depth = max(k, recall_k)
    total_recall, total_ap, n = 0.0, 0.0, 0
    for i, qid in enumerate(query_ids):
        relevant = ground_truth.get(int(qid))
        if not relevant:
            continue
        pred = article_ids[rank_indices(combined[i], article_ids, depth)].tolist()
        total_recall += recall_at_k(pred, relevant, recall_k)
        total_ap += average_precision_at_k(pred, relevant, k)
        n += 1
    if n == 0:
        return (0.0, 0.0)
    mean_recall, mean_ap = total_recall / n, total_ap / n
    return (mean_ap, mean_recall) if objective == "map" else (mean_recall, mean_ap)


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
    recall_k: int,
    objective: str,
    seed: int,
) -> dict[str, float]:
    """Перебрать веса на подмножестве строк, вернуть лучший вектор по objective."""
    sub = stack[:, rows, :]  # [S, |subset|, A]
    best_weights = None
    best_obj = (-1.0, -1.0)
    for weight_vec in _weight_samples(len(names), n_samples, seed):
        combined = np.tensordot(weight_vec, sub, axes=1)  # [|subset|, A]
        obj = _objective_from_combined(combined, subset_ids, article_ids, ground_truth, k, recall_k, objective)
        if obj > best_obj:  # лексикографическое сравнение кортежа
            best_obj, best_weights = obj, weight_vec
    return {name: float(best_weights[i]) for i, name in enumerate(names)}


def search_weights(
    matrices: Mapping[str, ScoreMatrix],
    ground_truth: Mapping[int, set[int]],
    query_ids_subset: Sequence[int],
    *,
    names: Sequence[str],
    n_samples: int = 500,
    k: int = 10,
    recall_k: int = 30,
    objective: str = "recall",
    seed: int = 42,
) -> dict[str, float]:
    """Подобрать веса на заданном подмножестве запросов (напр. все dev — для теста)."""
    all_query_ids, article_ids = align_matrices(matrices)
    row_of = {int(q): i for i, q in enumerate(all_query_ids)}
    stack = np.stack([minmax_rows(matrices[n].scores) for n in names])
    rows = [row_of[int(q)] for q in query_ids_subset]
    return _search_on_stack(
        stack, rows, query_ids_subset, article_ids, ground_truth, names,
        n_samples=n_samples, k=k, recall_k=recall_k, objective=objective, seed=seed,
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
    recall_k: int = 30,
    objective: str = "recall",
    rrf_k: int = 60,
    seed: int = 42,
    depth: int = 100,
) -> tuple[dict[int, list[int]], dict[int, dict[str, float]], ScoreMatrix]:
    """OOF-ранжирования dev + веса по фолдам + OOF-матрица скоров (dev по фолдам).

    weighted_sum: на фолде f веса ищутся на train(f) (по recall@recall_k, tie MAP@k) и
    применяются к val(f). rrf: параметров нет → RRF на каждом запросе (метка не
    видна, поэтому OOF-безопасно для всех). OOF-матрица нужна, чтобы кандидаты и
    признаки dev ниже по пайплайну не текли из весов, обученных на этих же запросах.
    """
    all_query_ids, article_ids = align_matrices(matrices)
    row_of = {int(q): i for i, q in enumerate(all_query_ids)}

    if method == "rrf":
        rrf_matrix = reciprocal_rank_fusion(matrices, k=rrf_k)  # label-independent → OOF-safe
        rankings = rrf_matrix.rankings(depth)
        return {q: rankings[q] for q in splits.dev}, {}, rrf_matrix

    stack = np.stack([minmax_rows(matrices[n].scores) for n in names])
    oof_scores = np.zeros((len(all_query_ids), len(article_ids)), dtype=np.float32)
    oof_rankings: dict[int, list[int]] = {}
    fold_weights: dict[int, dict[str, float]] = {}
    for fold in range(splits.n_splits):
        weights = _search_on_stack(
            stack, [row_of[q] for q in splits.train_ids(fold)], splits.train_ids(fold),
            article_ids, ground_truth, names,
            n_samples=n_samples, k=k, recall_k=recall_k, objective=objective, seed=seed + fold,
        )
        fold_weights[fold] = weights
        weight_vec = np.array([weights[n] for n in names], dtype=np.float32)
        val_rows = [row_of[q] for q in splits.fold(fold)]
        combined = np.tensordot(weight_vec, stack[:, val_rows, :], axes=1)  # [|val|, A]
        oof_scores[val_rows] = combined.astype(np.float32)
        for i, qid in enumerate(splits.fold(fold)):
            idx = rank_indices(combined[i], article_ids, depth)
            oof_rankings[qid] = article_ids[idx].tolist()
    oof_matrix = ScoreMatrix(all_query_ids, article_ids, oof_scores, FUSION_SOURCE)
    return oof_rankings, fold_weights, oof_matrix
