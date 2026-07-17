"""Сборка answer.csv из ранжирований.

Формат (ТЗ): колонки `query_id`, `answer` — до 10 article_id через пробел, без
повторов. Метрика не штрафует хвост, поэтому всегда возвращаем ровно top_k
(обычно 10): лишние документы в конце не портят MAP@10, если порядок уже
выбранных не меняется.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from ..contracts import ANSWER_COLUMNS


def build_answer(
    rankings: Mapping[int, Sequence[int]],
    query_ids: Sequence[int],
    *,
    top_k: int = 10,
) -> pd.DataFrame:
    """Ранжирования → таблица answer (по одной строке на каждый query_id).

    `query_ids` задаёт полный набор и порядок строк (все запросы из test.f);
    если для запроса нет ранжирования, ставится пустая строка — контракт
    экспорта такой ответ отклонит, что и нужно (пустых ответов быть не должно).
    """
    rows = []
    for qid in query_ids:
        ids = list(rankings.get(int(qid), []))[:top_k]
        rows.append({"query_id": int(qid), "answer": " ".join(str(a) for a in ids)})
    return pd.DataFrame(rows, columns=ANSWER_COLUMNS)


def write_answer(answer: pd.DataFrame, path: str | Path) -> Path:
    """Записать answer.csv (utf-8, без индекса)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    answer.to_csv(path, index=False)
    return path
