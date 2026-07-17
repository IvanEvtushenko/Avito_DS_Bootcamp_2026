"""Санити-проверки answer.csv (план §8, этап 8).

Отдельная команда `validate-answer`: читает готовый файл и прогоняет контракт
ответа (все query_id из test.f ровно по разу, ≤10 уникальных существующих
article_id, без пустых). Ошибка — понятное исключение с указанием запроса.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..contracts import check_answer_frame
from ..logging_utils import get_logger

logger = get_logger("export.validate")


def validate_answer_file(
    answer_path: str | Path,
    *,
    test: pd.DataFrame,
    articles: pd.DataFrame,
    max_k: int = 10,
) -> pd.DataFrame:
    """Проверить файл ответа против test.f и корпуса статей. Вернуть прочитанный df."""
    answer_path = Path(answer_path)
    if not answer_path.exists():
        raise FileNotFoundError(f"answer.csv не найден: {answer_path}")

    df = pd.read_csv(answer_path)
    # answer читается как int, если все строки — одно число; приводим к строке.
    df["answer"] = df["answer"].astype(str)
    check_answer_frame(
        df,
        expected_query_ids=test["query_id"].tolist(),
        valid_article_ids=articles["article_id"].tolist(),
        max_k=max_k,
    )
    n_ids = df["answer"].str.split().apply(len)
    logger.info(
        "answer.csv OK: %d запросов, статей на запрос min=%d/median=%d/max=%d",
        len(df), int(n_ids.min()), int(n_ids.median()), int(n_ids.max()),
    )
    return df
