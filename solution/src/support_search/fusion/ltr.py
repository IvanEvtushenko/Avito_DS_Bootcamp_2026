"""Слияние ретриверов логистической регрессией (LR-вариант fusion, план §7.2).

Альтернатива случайному поиску весов из `fusion/search.py`: вместо подбора
скаляров-весов под MAP@10 обучаем LR на релевантности пар (запрос, статья). Вход —
признаки каждой статьи: min-max-нормированный скор источника **и** его обратный
ранг (1/(rank+1)). Выход — P(релевантна), по которой ранжируем.

Чем LR интереснее взвешенной суммы: (1) учитывает ранги, а не только значения;
(2) калибрует смещение; (3) даёт интерпретируемые коэффициенты (какой источник и
насколько важен). Обучение строго out-of-fold (§4.2), как и подбор весов.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..contracts import ScoreMatrix, rank_indices
from ..data.splits import Splits
from ..ltr import LogisticRegressionLTR, reciprocal_ranks
from .combine import FUSION_SOURCE, align_matrices, minmax_rows


def build_fusion_features(
    matrices: Mapping[str, ScoreMatrix], names: Sequence[str], *, use_ranks: bool = True
) -> tuple[np.ndarray, list[str]]:
    """Тензор признаков [Q, A, F] и их имена: на источник — норм-скор (+ обратный ранг)."""
    feats: list[np.ndarray] = []
    feature_names: list[str] = []
    for name in names:
        feats.append(minmax_rows(matrices[name].scores))
        feature_names.append(f"{name}_norm")
        if use_ranks:
            feats.append(reciprocal_ranks(matrices[name].scores))
            feature_names.append(f"{name}_rr")
    return np.stack(feats, axis=-1), feature_names


def relevance_matrix(
    query_ids: np.ndarray, article_ids: np.ndarray, ground_truth: Mapping[int, set[int]]
) -> np.ndarray:
    """Метки релевантности [Q, A] ∈ {0,1} из ground truth."""
    col_of = {int(a): j for j, a in enumerate(article_ids)}
    labels = np.zeros((len(query_ids), len(article_ids)), dtype=np.float64)
    for i, qid in enumerate(query_ids):
        for aid in ground_truth.get(int(qid), ()):  # type: ignore[union-attr]
            j = col_of.get(int(aid))
            if j is not None:
                labels[i, j] = 1.0
    return labels


def _fit_on_rows(features: np.ndarray, labels: np.ndarray, rows: Sequence[int], l2: float) -> LogisticRegressionLTR:
    n_feat = features.shape[-1]
    x = features[list(rows)].reshape(-1, n_feat)
    y = labels[list(rows)].reshape(-1)
    return LogisticRegressionLTR(l2=l2).fit(x, y)


def fit_fusion_ltr(
    matrices: Mapping[str, ScoreMatrix],
    ground_truth: Mapping[int, set[int]],
    subset_ids: Sequence[int],
    *,
    names: Sequence[str],
    l2: float = 1.0,
    use_ranks: bool = True,
) -> tuple[LogisticRegressionLTR, list[str]]:
    """Обучить LR на подмножестве запросов (напр. все dev — для предсказаний на test)."""
    query_ids, article_ids = align_matrices(matrices)
    features, feature_names = build_fusion_features(matrices, names, use_ranks=use_ranks)
    labels = relevance_matrix(query_ids, article_ids, ground_truth)
    row_of = {int(q): i for i, q in enumerate(query_ids)}
    lr = _fit_on_rows(features, labels, [row_of[int(q)] for q in subset_ids], l2)
    return lr, feature_names


def predict_fusion_ltr(
    lr: LogisticRegressionLTR, matrices: Mapping[str, ScoreMatrix], *, names: Sequence[str], use_ranks: bool = True
) -> ScoreMatrix:
    """Применить обученную LR ко всем статьям → матрица скоров [Q, A]."""
    query_ids, article_ids = align_matrices(matrices)
    features, _ = build_fusion_features(matrices, names, use_ranks=use_ranks)
    n_feat = features.shape[-1]
    proba = lr.predict_proba(features.reshape(-1, n_feat)).reshape(len(query_ids), len(article_ids))
    return ScoreMatrix(query_ids=query_ids, article_ids=article_ids, scores=proba.astype(np.float32), source=FUSION_SOURCE)


def oof_fusion_ltr(
    matrices: Mapping[str, ScoreMatrix],
    ground_truth: Mapping[int, set[int]],
    splits: Splits,
    *,
    names: Sequence[str],
    l2: float = 1.0,
    use_ranks: bool = True,
    depth: int = 100,
) -> tuple[dict[int, list[int]], list[str], ScoreMatrix]:
    """OOF-ранжирования dev + OOF-матрица: на фолде f LR учится на train(f), предсказывает val(f)."""
    query_ids, article_ids = align_matrices(matrices)
    features, feature_names = build_fusion_features(matrices, names, use_ranks=use_ranks)
    labels = relevance_matrix(query_ids, article_ids, ground_truth)
    row_of = {int(q): i for i, q in enumerate(query_ids)}
    n_feat = features.shape[-1]

    oof: dict[int, list[int]] = {}
    oof_scores = np.zeros((len(query_ids), len(article_ids)), dtype=np.float32)
    for fold in range(splits.n_splits):
        lr = _fit_on_rows(features, labels, [row_of[q] for q in splits.train_ids(fold)], l2)
        val_rows = [row_of[q] for q in splits.fold(fold)]
        proba = lr.predict_proba(features[val_rows].reshape(-1, n_feat)).reshape(len(val_rows), -1)
        oof_scores[val_rows] = proba.astype(np.float32)
        for i, qid in enumerate(splits.fold(fold)):
            idx = rank_indices(proba[i], article_ids, depth)
            oof[qid] = article_ids[idx].tolist()
    oof_matrix = ScoreMatrix(query_ids, article_ids, oof_scores, FUSION_SOURCE)
    return oof, feature_names, oof_matrix
