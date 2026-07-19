"""Граф-фичи кандидатов (план §7.1): связи co-label / HTML-ссылки / dense-похожесть.

Рёбра article→article извлекает `preprocess.html.extract_article_links`; здесь —
сборка графов и признаков «кандидат связан с топ-соседями запроса» для mini-LTR
(протокол без утечки — в докстринге `graphs`).
"""
from __future__ import annotations

from .graphs import GRAPH_FEATURES, GraphFeaturizer, article_vectors, co_label_weights, link_adjacency

__all__ = [
    "GRAPH_FEATURES",
    "GraphFeaturizer",
    "article_vectors",
    "co_label_weights",
    "link_adjacency",
]
