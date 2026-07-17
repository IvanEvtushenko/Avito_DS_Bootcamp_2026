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
from .search import oof_fusion_search, search_weights  # legacy: random search весов (ablation)
from .ltr import (
    build_fusion_features,
    fit_fusion_ltr,
    oof_fusion_ltr,
    predict_fusion_ltr,
)

__all__ = [
    "align_matrices",
    "minmax_rows",
    "reciprocal_rank_fusion",
    "weighted_sum",
    # LR-вариант (дефолт, §7.2)
    "build_fusion_features",
    "fit_fusion_ltr",
    "oof_fusion_ltr",
    "predict_fusion_ltr",
    # legacy random search (ablation)
    "oof_fusion_search",
    "search_weights",
]
