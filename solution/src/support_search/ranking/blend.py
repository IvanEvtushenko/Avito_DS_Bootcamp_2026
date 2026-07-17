"""Финальное ранжирование: blend(fusion_score, reranker_score) (план §7).

Реранкер переставляет только кандидатов, поэтому итоговый скор считается **внутри
кандидатов**: fusion-скор и reranker-логит нормируются по кандидатам запроса
(min-max) и складываются с весом `weight`. Не-кандидаты остаются ниже любого
кандидата (recall@50 ≈ 0.975 → топ-10 всегда среди кандидатов).

Вес `weight` — дешёвый параметр: подбирается тем же честным OOF-протоколом, что и
веса fusion (на train(f), оценка на val(f)).
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..contracts import ScoreMatrix, rank_indices
from ..data.splits import Splits
from ..eval.metrics import average_precision_at_k
from ..rerank.apply import NON_CANDIDATE

BLEND_SOURCE = "blend"
_CAND_BASE = 1.0  # база для кандидатов, чтобы они всегда были выше не-кандидатов


def _candidate_mask(reranker: np.ndarray) -> np.ndarray:
    return reranker > (NON_CANDIDATE / 2.0)


def _minmax_masked(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max по кандидатам каждой строки; не-кандидаты → 0."""
    out = np.zeros_like(scores, dtype=np.float32)
    for i in range(scores.shape[0]):
        cols = np.where(mask[i])[0]
        if cols.size == 0:
            continue
        values = scores[i, cols]
        lo, hi = float(values.min()), float(values.max())
        span = (hi - lo) or 1.0
        out[i, cols] = (values - lo) / span
    return out


def blend_matrix(fusion: ScoreMatrix, reranker: ScoreMatrix, *, weight: float) -> ScoreMatrix:
    """Итоговая матрица: кандидаты = base + (w·fusion_norm + (1-w)·reranker_norm)."""
    mask = _candidate_mask(reranker.scores)
    fusion_norm = _minmax_masked(fusion.scores, mask)
    reranker_norm = _minmax_masked(reranker.scores, mask)
    blended = weight * fusion_norm + (1.0 - weight) * reranker_norm
    final = np.where(mask, _CAND_BASE + blended, 0.0).astype(np.float32)  # кандидаты > не-кандидаты
    return ScoreMatrix(query_ids=fusion.query_ids, article_ids=fusion.article_ids, scores=final, source=BLEND_SOURCE)


def _map_for_weight(
    fusion_norm: np.ndarray,
    reranker_norm: np.ndarray,
    mask: np.ndarray,
    article_ids: np.ndarray,
    query_ids: Sequence[int],
    ground_truth: Mapping[int, set[int]],
    weight: float,
    k: int,
) -> float:
    total, n = 0.0, 0
    for i, qid in enumerate(query_ids):
        relevant = ground_truth.get(int(qid))
        if not relevant:
            continue
        blended = np.where(mask[i], _CAND_BASE + weight * fusion_norm[i] + (1 - weight) * reranker_norm[i], 0.0)
        idx = rank_indices(blended, article_ids, k)
        total += average_precision_at_k(article_ids[idx].tolist(), relevant, k)
        n += 1
    return total / n if n else 0.0


def search_blend_weight(
    fusion: ScoreMatrix,
    reranker: ScoreMatrix,
    ground_truth: Mapping[int, set[int]],
    subset_ids: Sequence[int],
    *,
    k: int = 10,
    grid: int = 51,
) -> float:
    """Одномерный поиск веса на подмножестве запросов (по сетке [0,1])."""
    row_of = {int(q): i for i, q in enumerate(fusion.query_ids)}
    rows = [row_of[int(q)] for q in subset_ids]
    mask = _candidate_mask(reranker.scores)[rows]
    fusion_norm = _minmax_masked(fusion.scores, _candidate_mask(reranker.scores))[rows]
    reranker_norm = _minmax_masked(reranker.scores, _candidate_mask(reranker.scores))[rows]

    best_w, best_map = 0.0, -1.0
    for weight in np.linspace(0.0, 1.0, grid):
        score = _map_for_weight(fusion_norm, reranker_norm, mask, fusion.article_ids, subset_ids, ground_truth, weight, k)
        if score > best_map:
            best_map, best_w = score, float(weight)
    return best_w


def oof_blend(
    fusion: ScoreMatrix,
    reranker: ScoreMatrix,
    ground_truth: Mapping[int, set[int]],
    splits: Splits,
    *,
    k: int = 10,
    depth: int = 100,
    grid: int = 51,
) -> tuple[dict[int, list[int]], dict[int, float]]:
    """OOF-ранжирования dev: на каждом фолде вес ищется на train(f), применяется к val(f)."""
    row_of = {int(q): i for i, q in enumerate(fusion.query_ids)}
    mask_full = _candidate_mask(reranker.scores)
    fusion_norm = _minmax_masked(fusion.scores, mask_full)
    reranker_norm = _minmax_masked(reranker.scores, mask_full)

    oof: dict[int, list[int]] = {}
    fold_weights: dict[int, float] = {}
    for fold in range(splits.n_splits):
        weight = search_blend_weight(fusion, reranker, ground_truth, splits.train_ids(fold), k=k, grid=grid)
        fold_weights[fold] = weight
        for qid in splits.fold(fold):
            i = row_of[qid]
            blended = np.where(mask_full[i], _CAND_BASE + weight * fusion_norm[i] + (1 - weight) * reranker_norm[i], 0.0)
            idx = rank_indices(blended, fusion.article_ids, depth)
            oof[qid] = fusion.article_ids[idx].tolist()
    return oof, fold_weights
