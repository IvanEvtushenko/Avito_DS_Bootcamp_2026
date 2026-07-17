"""Метрики, статистика значимости и eval-harness.

Сквозной пакет: не зависит от ступеней пайплайна, оценивает любой ранжировщик
по единому протоколу (план §4, §5.1.1).
"""
from __future__ import annotations

from .metrics import (
    average_precision_at_k,
    mean_average_precision_at_k,
    recall_at_k,
)
from .stats import bootstrap_ci, paired_permutation_test
from .harness import EvalResult, evaluate

__all__ = [
    "average_precision_at_k",
    "mean_average_precision_at_k",
    "recall_at_k",
    "bootstrap_ci",
    "paired_permutation_test",
    "EvalResult",
    "evaluate",
]
