"""Применение реранкера к кандидатам → матрица скоров реранкера.

Реранкер оценивает только топ-K кандидатов каждого запроса (переставлять весь
корпус незачем). Результат кладётся в матрицу запрос × статья, где у не-кандидатов
стоит сентинел `NON_CANDIDATE` — так матрица остаётся единым форматом обмена, а
бленд (этап 7) знает, что переставлять можно только кандидатов.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..contracts import ScoreMatrix
from ..retrievers import DenseRetriever
from .cross_encoder import Reranker

# Скор не-кандидата: конечное, но заведомо ниже любого реального логита.
NON_CANDIDATE = -1.0e30


def candidate_lists(fusion: ScoreMatrix, top_k: int) -> list[list[int]]:
    """Топ-k article_id по fusion для каждого запроса (в порядке строк матрицы)."""
    rankings = fusion.rankings(top_k)
    return [rankings[int(q)] for q in fusion.query_ids]


def build_passages(
    dense: DenseRetriever, query_texts: Sequence[str], cand_lists: Sequence[Sequence[int]]
) -> list[list[str]]:
    """Пассаж для пары (запрос, кандидат) = лучший по dense-сходству чанк статьи."""
    q_emb = dense.encode_queries(list(query_texts))
    return dense.best_chunk_texts(q_emb, cand_lists)


def build_passages_from_qemb(
    dense: DenseRetriever, query_embeddings: np.ndarray, cand_lists: Sequence[Sequence[int]]
) -> list[list[str]]:
    """То же, но по кэшированным эмбеддингам запросов (без повторного кодирования)."""
    return dense.best_chunk_texts(query_embeddings, cand_lists)


def _as_chunks(passage: str | Sequence[str]) -> list[str]:
    """Пассаж кандидата → список чанков (строка = один чанк)."""
    return [passage] if isinstance(passage, str) else list(passage)


def rerank_to_matrix(
    reranker: Reranker,
    query_texts: Sequence[str],
    query_ids: Sequence[int],
    article_ids: np.ndarray,
    cand_lists: Sequence[Sequence[int]],
    passages: Sequence[Sequence[str | Sequence[str]]],
    *,
    source: str = "reranker",
) -> ScoreMatrix:
    """Оценить все пары (запрос, кандидат) и разложить логиты в матрицу [Q, A].

    Пассаж кандидата — строка (один чанк) или список чанков: тогда скор статьи =
    max по её чанкам (идея №3 MAP_IMPROVEMENT_IDEAS — не наследовать ошибку
    единственного чанка, выбранного dense-моделью).
    """
    col_of = {int(a): j for j, a in enumerate(article_ids)}
    scores = np.full((len(query_ids), len(article_ids)), NON_CANDIDATE, dtype=np.float32)

    # Листовой реранкер (напр. jina): запрос и весь его список кандидатов — вместе,
    # по-запросно, а не одной сплющенной пачкой пар. Чанки статьи входят в тот же
    # листовой контекст, скор статьи — max по её чанкам.
    if hasattr(reranker, "score_listwise"):
        for i, (cands, passs) in enumerate(zip(cand_lists, passages)):
            if not cands:
                continue
            chunk_lists = [_as_chunks(p) for p in passs]
            flat = [c for chunks in chunk_lists for c in chunks]
            flat_scores = reranker.score_listwise(query_texts[i], flat)
            pos = 0
            for aid, chunks in zip(cands, chunk_lists):
                j = col_of[int(aid)]
                scores[i, j] = float(np.max(flat_scores[pos : pos + len(chunks)]))
                pos += len(chunks)
        return ScoreMatrix(query_ids=np.asarray(query_ids), article_ids=np.asarray(article_ids), scores=scores, source=source)

    flat_queries: list[str] = []
    flat_passages: list[str] = []
    positions: list[tuple[int, int]] = []
    for i, (cands, passs) in enumerate(zip(cand_lists, passages)):
        for aid, passage in zip(cands, passs):
            for chunk in _as_chunks(passage):
                flat_queries.append(query_texts[i])
                flat_passages.append(chunk)
                positions.append((i, col_of[int(aid)]))

    if flat_queries:
        pair_scores = reranker.score_pairs(flat_queries, flat_passages)
        for (i, j), s in zip(positions, pair_scores):
            scores[i, j] = max(scores[i, j], float(s))  # max по чанкам статьи
    return ScoreMatrix(query_ids=np.asarray(query_ids), article_ids=np.asarray(article_ids), scores=scores, source=source)
