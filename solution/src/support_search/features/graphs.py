"""Граф-фичи кандидатов для финального ранжирования (план §7.1 + диагностика).

Диагностика (experiments/article_query_graph_diagnostics.ipynb) показала, что
44% запросов имеют 2+ GT-статьи и пары «инструкция + правила» образуют плотный
co-label граф; реранкер же скорит кандидатов независимо и не знает, что статья
дополняет уже поднятый топ. Эти признаки дают mini-LTR ровно этот сигнал:
связь кандидата с текущими топ-`m` кандидатами реранкера.

Признаки (на пару запрос × кандидат):

- ``co_top_p`` (дефолт, co_weight=cond) — max по лидерам запроса условная
  вероятность P(кандидат в GT | лидер в GT) = W[кандидат, лидер] / freq(лидер).
  Нормировка отличает эксклюзивные связки («правила ↔ инструкция», ходят только
  вместе) от частых хабовых совместностей: пара 2665–4214 (7 из ~8 запросов
  якоря) сильнее пары 4219–2646 (45 из ~200). Сырой вариант ``co_top_w`` =
  log1p(max вес) оставлен как ablation (co_weight=raw), OOF: cond +0.0014;
- ``link_top``  — есть HTML-ссылка между кандидатом и одним из лидеров
  (строится по всем 793 статьям, разметка не нужна);
- ``sim_top``   — max косинус dense-вектора кандидата к лидерам (тоже без
  разметки: закрывает статьи, которых нет среди 79 размеченных);
- ``gt_freq``   — log1p(частота кандидата как GT): явный прайор популярности;
- ``co_x_margin`` (опц.) — co-фича × отрыв топ-1 от топ-2 по реранкеру:
  «поднимать пару к лидеру стоит тем сильнее, чем увереннее сам лидер» —
  взаимодействие, которое линейная голова иначе не выразит (OOF: +0.0014
  к lr_graph, стабильно по фолдам — см. README).

Протокол без утечки — как у остальных обучаемых стадий (§4.2):

- co-label веса и частоты строятся ТОЛЬКО по train-части фолда (`gt_train`);
  для val/test-строк GT запроса в графе нет по построению;
- для train-строк собственный вклад запроса вычитается (leave-one-out): иначе
  признак частично кодирует собственную метку, и LR переоценивает его вес.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..preprocess import extract_article_links

GRAPH_FEATURES = ("link_top", "sim_top", "gt_freq")  # + co-фича и (опц.) interaction


def link_adjacency(article_ids: Sequence[int], bodies: Sequence[str]) -> np.ndarray:
    """Симметричная [A, A] матрица «статьи связаны HTML-ссылкой» (bool).

    Направление ссылки для ранжирования не важно: «инструкция → правила»
    поддерживает пару в обе стороны. Ссылки на чужие ID игнорируются.
    """
    col_of = {int(a): j for j, a in enumerate(article_ids)}
    adj = np.zeros((len(col_of), len(col_of)), dtype=bool)
    for source, body in zip(article_ids, bodies):
        i = col_of[int(source)]
        for target in extract_article_links(str(body)):
            j = col_of.get(int(target))
            if j is not None and j != i:
                adj[i, j] = adj[j, i] = True
    return adj


def co_label_weights(
    gt: Mapping[int, set[int]], article_ids: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Co-label веса [A, A] и частоты GT [A] по переданной разметке.

    Вес (a, b) = число запросов, где обе статьи в GT; диагональ нулевая.
    Вызывающий отвечает за состав `gt` (train-часть фолда — см. модульный
    докстринг про протокол).
    """
    col_of = {int(a): j for j, a in enumerate(article_ids)}
    weights = np.zeros((len(col_of), len(col_of)), dtype=np.float32)
    freq = np.zeros(len(col_of), dtype=np.float32)
    for relevant in gt.values():
        cols = sorted(col_of[int(a)] for a in relevant if int(a) in col_of)
        for j in cols:
            freq[j] += 1.0
        for x, j in enumerate(cols):
            for t in cols[x + 1:]:
                weights[j, t] += 1.0
                weights[t, j] += 1.0
    return weights, freq


def article_vectors(chunk_emb: np.ndarray, title_emb: np.ndarray, chunk_offsets: np.ndarray) -> np.ndarray:
    """L2-нормированный вектор статьи: среднее её чанков + заголовок [A, D].

    Кэш dense-индекса уже на диске (retrieve), так что «семантический граф»
    статей достаётся без единого нового прогона энкодера.
    """
    sums = np.add.reduceat(chunk_emb, chunk_offsets, axis=0)
    counts = np.diff(np.append(chunk_offsets, chunk_emb.shape[0]))[:, None]
    vec = sums / np.maximum(counts, 1) + title_emb
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    return (vec / np.maximum(norms, 1e-12)).astype(np.float32)


class GraphFeaturizer:
    """Фичи связей кандидата с топ-`m` кандидатами реранкера.

    Фолд-независимые входы (ссылки, векторы статей) передаются один раз;
    разметка — при каждом вызове `features` (своя на фолд).
    """

    def __init__(
        self,
        article_ids: Sequence[int],
        *,
        link_adj: np.ndarray,
        article_vec: np.ndarray,
        top_m: int = 3,
        interaction: bool = True,
        co_weight: str = "cond",
    ) -> None:
        if co_weight not in ("cond", "raw"):
            raise ValueError(f"co_weight: ожидали 'cond' или 'raw', получили {co_weight!r}")
        self.article_ids = np.asarray(article_ids, dtype=np.int64)
        self.link_adj = link_adj
        self.article_vec = article_vec
        self.top_m = int(top_m)
        self.interaction = bool(interaction)
        self.co_weight = co_weight
        self.sim = article_vec @ article_vec.T  # [A, A], косинусы (векторы нормированы)

    def feature_names(self) -> list[str]:
        co_name = "co_top_p" if self.co_weight == "cond" else "co_top_w"
        return [co_name, *GRAPH_FEATURES] + (["co_x_margin"] if self.interaction else [])

    def features(
        self,
        scores: np.ndarray,
        mask: np.ndarray,
        query_ids: Sequence[int],
        gt_train: Mapping[int, set[int]],
    ) -> np.ndarray:
        """Признаки [Q, A, 4]; вне кандидатов — 0 (как у базовых признаков).

        `scores` — скоры реранкера (лидеры запроса = его топ-m кандидатов),
        `gt_train` — разметка, по которой строится граф. Для запросов из
        `gt_train` применяется leave-one-out (см. модульный докстринг).
        """
        weights, freq = co_label_weights(gt_train, self.article_ids)
        col_of = {int(a): j for j, a in enumerate(self.article_ids)}
        out = np.zeros((*mask.shape, len(self.feature_names())), dtype=np.float32)

        for i, qid in enumerate(query_ids):
            cols = np.where(mask[i])[0]
            if cols.size == 0:
                continue
            ordered = np.sort(scores[i, cols])[::-1]
            span = float(ordered[0] - ordered[-1]) or 1.0
            margin = float(ordered[0] - ordered[1]) / span if cols.size > 1 else 0.0
            top = cols[np.argsort(-scores[i, cols], kind="stable")][: self.top_m]
            own = {col_of[int(a)] for a in gt_train.get(int(qid), ()) if int(a) in col_of}
            own_freq = 1.0 if own else 0.0
            for j in cols:
                anchors = top[top != j]
                if anchors.size:
                    # LOO: собственные пары/частоты запроса вычитаются из графа.
                    co_w = weights[j, anchors]
                    if j in own:
                        co_w = co_w - np.isin(anchors, list(own)).astype(np.float32)
                    if self.co_weight == "cond":
                        anchor_freq = freq[anchors] - np.isin(anchors, list(own)).astype(np.float32)
                        cond = np.where(anchor_freq > 0, np.maximum(co_w, 0.0) / np.maximum(anchor_freq, 1.0), 0.0)
                        out[i, j, 0] = float(cond.max())
                    else:
                        out[i, j, 0] = np.log1p(max(float(co_w.max()), 0.0))
                    out[i, j, 1] = float(self.link_adj[j, anchors].any())
                    out[i, j, 2] = float(self.sim[j, anchors].max())
                out[i, j, 3] = np.log1p(freq[j] - (own_freq if j in own else 0.0))
                if self.interaction:
                    # Уверенность лидера: нормированный отрыв топ-1 от топ-2.
                    out[i, j, 4] = out[i, j, 0] * margin
        return out
