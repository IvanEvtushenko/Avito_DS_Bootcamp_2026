"""Общий интерфейс ретриверов и корпус, который они индексируют.

`Corpus` держит статьи в виде, готовом для лексики: id, plain-заголовок и
очищенный от HTML текст тела — раздельно, потому что заголовок бустится
отдельным весом (BM25F-lite). `Retriever` фиксирует контракт всех ретриверов
(план §5.1.3): `fit` строит индекс по корпусу, `score_matrix` отдаёт полную
матрицу скоров запрос × статья (единый формат обмена, §5.1.2), `retrieve` —
её длинную топ-k проекцию, `save/load` — персистентность.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..contracts import ScoreMatrix
from ..preprocess import html_to_text


@dataclass
class Corpus:
    """Корпус статей для индексации лексическими ретриверами."""

    article_ids: np.ndarray  # [A] int64
    titles: list[str]        # plain-текст заголовков
    texts: list[str]         # очищенный от HTML текст тела

    def __post_init__(self) -> None:
        self.article_ids = np.asarray(self.article_ids, dtype=np.int64)
        if not (len(self.article_ids) == len(self.titles) == len(self.texts)):
            raise ValueError("Corpus: длины article_ids/titles/texts не совпадают")

    def __len__(self) -> int:
        return len(self.article_ids)


def build_corpus(articles: pd.DataFrame, *, drop_tags: Sequence[str] | None = None) -> Corpus:
    """Собрать `Corpus` из таблицы статей, очистив HTML тела.

    Логирует долю статей с пустым текстом — это наблюдаемость стадии (§5.1.8) и
    ранний сигнал, что очистка HTML сломалась.
    """
    kwargs = {"drop_tags": tuple(drop_tags)} if drop_tags is not None else {}
    texts = [html_to_text(b, **kwargs) for b in articles["body"]]
    return Corpus(
        article_ids=articles["article_id"].to_numpy(dtype=np.int64),
        titles=[str(t) for t in articles["title"]],
        texts=texts,
    )


class Retriever(ABC):
    """Базовый интерфейс ретривера.

    Атрибут `name` совпадает с ключом в конфиге и полем `source` матрицы скоров.
    """

    name: str = "retriever"

    @abstractmethod
    def fit(self, corpus: Corpus, train_queries: Sequence[str] | None = None) -> "Retriever":
        """Построить индекс по корпусу. `train_queries` не нужен лексике (без обучения)."""

    @abstractmethod
    def score_matrix(self, queries: Sequence[str], query_ids: Sequence[int]) -> ScoreMatrix:
        """Полная матрица скоров запрос × статья."""

    def retrieve(self, queries: Sequence[str], query_ids: Sequence[int], top_k: int) -> pd.DataFrame:
        """Длинная таблица RETRIEVAL_COLUMNS с топ-k статей на запрос."""
        return self.score_matrix(queries, query_ids).to_retrieval_frame(top_k)

    @abstractmethod
    def save(self, path: str | Path) -> Path:
        """Сохранить индекс в каталог."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "Retriever":
        """Загрузить индекс из каталога."""
