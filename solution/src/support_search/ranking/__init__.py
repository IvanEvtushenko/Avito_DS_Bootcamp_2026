"""Финальное ранжирование (этап 7): blend(fusion_score, reranker_score)."""
from __future__ import annotations

from .blend import blend_matrix, oof_blend, search_blend_weight  # legacy: ручной вес (ablation)
from .ltr import fit_blend_ltr, oof_blend_ltr, predict_blend_ltr
from .mlp import MLPRankerLTR

__all__ = [
    # mini-LTR (дефолт, §7.2)
    "fit_blend_ltr",
    "oof_blend_ltr",
    "predict_blend_ltr",
    # listwise MLP-голова (альтернатива LR)
    "MLPRankerLTR",
    # legacy ручной блэнд (ablation)
    "blend_matrix",
    "oof_blend",
    "search_blend_weight",
]
