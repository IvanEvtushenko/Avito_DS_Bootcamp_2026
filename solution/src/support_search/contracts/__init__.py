"""Контракты между стадиями: схемы таблиц, матрица скоров, проверки.

Пакет сквозной — не импортирует ни одной ступени пайплайна, поэтому от него
можно зависеть всем (план §5.1.1).
"""
from __future__ import annotations

from .schemas import (
    ANSWER_COLUMNS,
    RETRIEVAL_COLUMNS,
    ScoreMatrix,
    rank_indices,
)
from .checks import (
    ContractError,
    check_answer_frame,
    check_retrieval_frame,
    check_score_matrix,
)

__all__ = [
    "ANSWER_COLUMNS",
    "RETRIEVAL_COLUMNS",
    "ScoreMatrix",
    "rank_indices",
    "ContractError",
    "check_answer_frame",
    "check_retrieval_frame",
    "check_score_matrix",
]
