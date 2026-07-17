"""Mini-LTR: финальное ранжирование логистической регрессией (план §7.2).

Альтернатива ручному блэнду из `ranking/blend.py`. Вместо одного веса
`w·fusion + (1-w)·reranker` обучаем LR поверх **всех** признаков кандидата:
min-max-нормированные (внутри кандидатов запроса) скоры BM25, char-TF-IDF, dense и
реранкера + их обратные ранги. Ранжирование — по P(релевантна).

Так финальный скор объединяет все сигналы и их ранги сразу, а не через
двухступенчатую ручную формулу; коэффициенты интерпретируемы. Обучение строго
out-of-fold (§4.2); переставляются только кандидаты (recall@K — потолок).
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..contracts import ScoreMatrix, rank_indices
from ..data.splits import Splits
from ..ltr import LogisticRegressionLTR
from ..rerank.apply import NON_CANDIDATE

BLEND_SOURCE = "blend"
_CAND_BASE = 1.0  # кандидаты всегда выше не-кандидатов


def _candidate_mask(reranker_scores: np.ndarray) -> np.ndarray:
    return reranker_scores > (NON_CANDIDATE / 2.0)


def _masked_norm(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max каждой строки по кандидатам; вне кандидатов — 0."""
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


def _masked_reciprocal_rank(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """1/(ранг+1) среди кандидатов запроса; вне кандидатов — 0."""
    out = np.zeros_like(scores, dtype=np.float32)
    for i in range(scores.shape[0]):
        cols = np.where(mask[i])[0]
        if cols.size == 0:
            continue
        ordered = cols[np.argsort(-scores[i, cols], kind="stable")]
        for rank, j in enumerate(ordered):
            out[i, j] = 1.0 / (rank + 1.0)
    return out


def build_blend_features(
    source_matrices: Mapping[str, ScoreMatrix],
    reranker: ScoreMatrix,
    *,
    names: Sequence[str],
    use_ranks: bool = True,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Признаки кандидатов [Q, A, F], их имена и маска кандидатов [Q, A]."""
    mask = _candidate_mask(reranker.scores)
    ordered_sources = list(names) + ["reranker"]
    score_by_name = {**{n: source_matrices[n].scores for n in names}, "reranker": reranker.scores}

    feats: list[np.ndarray] = []
    feature_names: list[str] = []
    for name in ordered_sources:
        feats.append(_masked_norm(score_by_name[name], mask))
        feature_names.append(f"{name}_norm")
        if use_ranks:
            feats.append(_masked_reciprocal_rank(score_by_name[name], mask))
            feature_names.append(f"{name}_rr")
    return np.stack(feats, axis=-1), feature_names, mask


def _labels(query_ids: np.ndarray, article_ids: np.ndarray, gt: Mapping[int, set[int]]) -> np.ndarray:
    col_of = {int(a): j for j, a in enumerate(article_ids)}
    y = np.zeros((len(query_ids), len(article_ids)), dtype=np.float64)
    for i, qid in enumerate(query_ids):
        for aid in gt.get(int(qid), ()):  # type: ignore[union-attr]
            j = col_of.get(int(aid))
            if j is not None:
                y[i, j] = 1.0
    return y


def _scatter(proba_rows: np.ndarray, mask_rows: np.ndarray) -> np.ndarray:
    """Вероятности кандидатов → матрица: кандидаты = base+proba, остальные = 0."""
    out = np.zeros(mask_rows.shape, dtype=np.float32)
    out[mask_rows] = _CAND_BASE + proba_rows.astype(np.float32)
    return out


def fit_blend_ltr(
    source_matrices, reranker, gt, subset_ids, *, names, l2=1.0, use_ranks=True
) -> tuple[LogisticRegressionLTR, list[str]]:
    features, feature_names, mask = build_blend_features(source_matrices, reranker, names=names, use_ranks=use_ranks)
    labels = _labels(reranker.query_ids, reranker.article_ids, gt)
    row_of = {int(q): i for i, q in enumerate(reranker.query_ids)}
    rows = [row_of[int(q)] for q in subset_ids]
    sel = mask[rows]
    lr = LogisticRegressionLTR(l2=l2).fit(features[rows][sel], labels[rows][sel])
    return lr, feature_names


def predict_blend_ltr(lr, source_matrices, reranker, *, names, use_ranks=True) -> ScoreMatrix:
    features, _, mask = build_blend_features(source_matrices, reranker, names=names, use_ranks=use_ranks)
    n_feat = features.shape[-1]
    proba = lr.predict_proba(features[mask].reshape(-1, n_feat))
    scores = np.zeros(mask.shape, dtype=np.float32)
    scores[mask] = _CAND_BASE + proba.astype(np.float32)
    return ScoreMatrix(reranker.query_ids, reranker.article_ids, scores, BLEND_SOURCE)


def oof_blend_ltr(
    source_matrices: Mapping[str, ScoreMatrix],
    reranker: ScoreMatrix,
    gt: Mapping[int, set[int]],
    splits: Splits,
    *,
    names: Sequence[str],
    l2: float = 1.0,
    use_ranks: bool = True,
    depth: int = 100,
) -> tuple[dict[int, list[int]], list[str]]:
    """OOF-ранжирования dev: LR фолда учится на train(f), предсказывает val(f)."""
    features, feature_names, mask = build_blend_features(source_matrices, reranker, names=names, use_ranks=use_ranks)
    labels = _labels(reranker.query_ids, reranker.article_ids, gt)
    row_of = {int(q): i for i, q in enumerate(reranker.query_ids)}
    n_feat = features.shape[-1]
    article_ids = reranker.article_ids

    oof: dict[int, list[int]] = {}
    for fold in range(splits.n_splits):
        tr = [row_of[q] for q in splits.train_ids(fold)]
        sel_tr = mask[tr]
        lr = LogisticRegressionLTR(l2=l2).fit(features[tr][sel_tr], labels[tr][sel_tr])
        for qid in splits.fold(fold):
            i = row_of[qid]
            row_scores = np.zeros(article_ids.shape[0], dtype=np.float32)
            cand = mask[i]
            if cand.any():
                row_scores[cand] = _CAND_BASE + lr.predict_proba(features[i][cand].reshape(-1, n_feat)).astype(np.float32)
            idx = rank_indices(row_scores, article_ids, depth)
            oof[qid] = article_ids[idx].tolist()
    return oof, feature_names
