"""Слияние кандидатов (план §6, этап 5).

Fusion — не ещё одна модель поиска, а правило слияния списков с парой
подбираемых весов. `combine` даёт операторы (min-max взвешенная сумма и RRF),
`search` — совместный random search весов по честной OOF-схеме (§4.2–4.3).
"""
from __future__ import annotations

from .combine import (
    align_matrices,
    minmax_rows,
    reciprocal_rank_fusion,
    weighted_sum,
)
from .search import oof_fusion_search, search_weights

__all__ = [
    "align_matrices",
    "minmax_rows",
    "reciprocal_rank_fusion",
    "weighted_sum",
    "oof_fusion_search",
    "search_weights",
]
