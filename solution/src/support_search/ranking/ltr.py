"""Mini-LTR: финальное ранжирование обучаемой головой (план §7.2).

Альтернатива ручному блэнду из `ranking/blend.py`. Вместо одного веса
`w·fusion + (1-w)·reranker` обучаем модель поверх **всех** признаков кандидата:
min-max-нормированные (внутри кандидатов запроса) скоры BM25, char-TF-IDF, dense и
реранкера + их обратные ранги; опционально — граф-фичи связей кандидата с
топ-соседями (`features/graphs.py`). Ранжирование — по P(релевантна).

Голова сменная: LR (дефолт, коэффициенты интерпретируемы) или listwise MLP
(`ranking/mlp.py`) — обе с одним интерфейсом `fit(X, y, groups)`/`predict_proba`.
Обучение строго out-of-fold (§4.2); граф-фичи каждого фолда строятся только по
его train-разметке (протокол — в докстринге `features/graphs.py`).
Переставляются только кандидаты (recall@K — потолок).
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

from ..contracts import ScoreMatrix, rank_indices
from ..data.splits import Splits
from ..features.graphs import GraphFeaturizer
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


def _with_graph_features(
    base: np.ndarray,
    base_names: list[str],
    reranker: ScoreMatrix,
    mask: np.ndarray,
    featurizer: GraphFeaturizer | None,
    gt_train: Mapping[int, set[int]] | None,
) -> tuple[np.ndarray, list[str]]:
    """Базовые признаки + граф-фичи от разметки `gt_train` (если заданы)."""
    if featurizer is None:
        return base, base_names
    extra = featurizer.features(reranker.scores, mask, reranker.query_ids, gt_train or {})
    return np.concatenate([base, extra], axis=-1), base_names + featurizer.feature_names()


def _fit_head(head, features: np.ndarray, labels: np.ndarray, rows: Sequence[int], mask: np.ndarray):
    """Обучить голову на строках-кандидатах запросов `rows`; groups = запрос строки."""
    sel = mask[rows]
    groups = np.nonzero(sel)[0]  # индекс запроса каждой строки-кандидата
    return head.fit(features[rows][sel], labels[rows][sel], groups)


def fit_blend_ltr(
    source_matrices, reranker, gt, subset_ids, *, names, l2=1.0, use_ranks=True,
    featurizer: GraphFeaturizer | None = None, head=None,
) -> tuple[LogisticRegressionLTR, list[str]]:
    """Обучить финальную голову на запросах `subset_ids` (граф — по их же разметке)."""
    base, base_names, mask = build_blend_features(source_matrices, reranker, names=names, use_ranks=use_ranks)
    gt_train = {int(q): gt[int(q)] for q in subset_ids if int(q) in gt}
    features, feature_names = _with_graph_features(base, base_names, reranker, mask, featurizer, gt_train)
    labels = _labels(reranker.query_ids, reranker.article_ids, gt)
    row_of = {int(q): i for i, q in enumerate(reranker.query_ids)}
    rows = [row_of[int(q)] for q in subset_ids]
    model = _fit_head(head if head is not None else LogisticRegressionLTR(l2=l2), features, labels, rows, mask)
    return model, feature_names


def predict_blend_ltr(
    model, source_matrices, reranker, *, names, use_ranks=True,
    featurizer: GraphFeaturizer | None = None, gt_train: Mapping[int, set[int]] | None = None,
) -> ScoreMatrix:
    """Скоры головы; `gt_train` — та же разметка, на которой строился граф при fit."""
    base, base_names, mask = build_blend_features(source_matrices, reranker, names=names, use_ranks=use_ranks)
    features, _ = _with_graph_features(base, base_names, reranker, mask, featurizer, gt_train)
    n_feat = features.shape[-1]
    proba = model.predict_proba(features[mask].reshape(-1, n_feat))
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
    featurizer: GraphFeaturizer | None = None,
    head_factory: Callable[[], object] | None = None,
) -> tuple[dict[int, list[int]], list[str]]:
    """OOF-ранжирования dev: голова фолда учится на train(f), предсказывает val(f).

    Граф-фичи фолда строятся только по разметке train(f) — и для train-строк
    (с leave-one-out внутри featurizer), и для val-строк (их GT в графе нет).
    """
    base, base_names, mask = build_blend_features(source_matrices, reranker, names=names, use_ranks=use_ranks)
    labels = _labels(reranker.query_ids, reranker.article_ids, gt)
    row_of = {int(q): i for i, q in enumerate(reranker.query_ids)}
    article_ids = reranker.article_ids
    make_head = head_factory if head_factory is not None else (lambda: LogisticRegressionLTR(l2=l2))

    oof: dict[int, list[int]] = {}
    feature_names = base_names
    for fold in range(splits.n_splits):
        train_ids = splits.train_ids(fold)
        gt_train = {int(q): gt[int(q)] for q in train_ids if int(q) in gt}
        features, feature_names = _with_graph_features(base, base_names, reranker, mask, featurizer, gt_train)
        n_feat = features.shape[-1]
        tr = [row_of[q] for q in train_ids]
        model = _fit_head(make_head(), features, labels, tr, mask)
        for qid in splits.fold(fold):
            i = row_of[qid]
            row_scores = np.zeros(article_ids.shape[0], dtype=np.float32)
            cand = mask[i]
            if cand.any():
                row_scores[cand] = _CAND_BASE + model.predict_proba(features[i][cand].reshape(-1, n_feat)).astype(np.float32)
            idx = rank_indices(row_scores, article_ids, depth)
            oof[qid] = article_ids[idx].tolist()
    return oof, feature_names
