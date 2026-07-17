"""Проверки контрактов между стадиями.

Каждая функция либо возвращает управление, либо кидает `ContractError` с
понятным сообщением — ошибка контракта должна завершать запуск сразу и явно
(план §5.1.8), а не всплывать искажённой пять стадий спустя.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .schemas import ANSWER_COLUMNS, RETRIEVAL_COLUMNS, ScoreMatrix


class ContractError(ValueError):
    """Нарушение контракта данных между стадиями."""


def check_score_matrix(sm: ScoreMatrix, valid_article_ids: Iterable[int]) -> ScoreMatrix:
    """Матрица скоров: конечные числа, статьи из корпуса, нет дублей id."""
    valid = set(int(a) for a in valid_article_ids)
    if not np.isfinite(sm.scores).all():
        raise ContractError(f"[{sm.source}] в матрице скоров есть NaN/inf")
    if len(set(sm.article_ids.tolist())) != len(sm.article_ids):
        raise ContractError(f"[{sm.source}] дубли article_id в столбцах матрицы")
    if len(set(sm.query_ids.tolist())) != len(sm.query_ids):
        raise ContractError(f"[{sm.source}] дубли query_id в строках матрицы")
    unknown = set(sm.article_ids.tolist()) - valid
    if unknown:
        raise ContractError(f"[{sm.source}] article_id вне корпуса: {sorted(unknown)[:5]}")
    return sm


def check_retrieval_frame(df: pd.DataFrame, valid_article_ids: Iterable[int]) -> pd.DataFrame:
    """Длинная таблица ретривера: колонки, типы, дубли, ранги, допустимость id."""
    missing = [c for c in RETRIEVAL_COLUMNS if c not in df.columns]
    if missing:
        raise ContractError(f"в таблице ретривера нет колонок: {missing}")

    dup = df.duplicated(subset=["query_id", "article_id"])
    if dup.any():
        n = int(dup.sum())
        raise ContractError(f"дубли пар (query_id, article_id): {n} строк")

    valid = set(int(a) for a in valid_article_ids)
    unknown = set(df["article_id"].tolist()) - valid
    if unknown:
        raise ContractError(f"article_id вне корпуса: {sorted(unknown)[:5]}")

    # Монотонность rank внутри запроса: 0, 1, 2, ... без пропусков.
    for qid, grp in df.groupby("query_id"):
        ranks = np.sort(grp["rank"].to_numpy())
        expected = np.arange(len(ranks))
        if not np.array_equal(ranks, expected):
            raise ContractError(f"query_id={qid}: ранги не образуют 0..{len(ranks) - 1}")
    return df


def check_answer_frame(
    df: pd.DataFrame,
    *,
    expected_query_ids: Iterable[int],
    valid_article_ids: Iterable[int],
    max_k: int = 10,
) -> pd.DataFrame:
    """Файл ответа (план §8): все запросы ровно один раз, ≤max_k уникальных
    существующих article_id, без пустых ответов, корректные колонки."""
    if list(df.columns) != ANSWER_COLUMNS:
        raise ContractError(f"ожидались колонки {ANSWER_COLUMNS}, получены {list(df.columns)}")

    expected = set(int(q) for q in expected_query_ids)
    got = df["query_id"].tolist()
    if len(got) != len(set(got)):
        raise ContractError("повторяющиеся query_id в answer.csv")
    if set(got) != expected:
        missing = sorted(expected - set(got))[:5]
        extra = sorted(set(got) - expected)[:5]
        raise ContractError(f"набор query_id не совпал с test.f (нет: {missing}, лишние: {extra})")

    valid = set(int(a) for a in valid_article_ids)
    for qid, ans in zip(df["query_id"], df["answer"]):
        if not isinstance(ans, str) or not ans.strip():
            raise ContractError(f"query_id={qid}: пустой ответ")
        ids = ans.split()
        if len(ids) > max_k:
            raise ContractError(f"query_id={qid}: больше {max_k} статей ({len(ids)})")
        int_ids = [int(x) for x in ids]
        if len(set(int_ids)) != len(int_ids):
            raise ContractError(f"query_id={qid}: повтор article_id в ответе")
        unknown = set(int_ids) - valid
        if unknown:
            raise ContractError(f"query_id={qid}: article_id вне корпуса: {sorted(unknown)}")
    return df
