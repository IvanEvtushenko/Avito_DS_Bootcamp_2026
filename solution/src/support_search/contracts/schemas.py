"""Схемы таблиц и матрица скоров — единый формат обмена между стадиями.

Ключевая идея плана (§5.1.2): корпус крошечный (500 запросов × 793 статьи),
поэтому любой скоринговый компонент отдаёт **полную матрицу скоров
(запрос × статья)**. Из неё за миллисекунды получается и топ-k кандидатов, и
арифметика fusion. `ScoreMatrix` — этот формат; `rank_indices` — единая,
детерминированная логика «отсортировать статьи по убыванию скора».
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Длинный формат результата любого ретривера.
RETRIEVAL_COLUMNS = ["query_id", "article_id", "score", "rank", "source"]
# Итоговый файл ответа.
ANSWER_COLUMNS = ["query_id", "answer"]


def rank_indices(scores: np.ndarray, article_ids: np.ndarray, k: int) -> np.ndarray:
    """Индексы топ-k статей по убыванию скора; ничьи — по возрастанию article_id.

    Детерминированность обязательна: одинаковый вход всегда даёт одинаковый
    порядок (иначе повторный запуск изменит answer.csv). Поэтому ничьи по скору
    разрешаются вторичным ключом `article_id`, а не порядком в памяти.

    Параметры
    ---------
    scores       : [A] float — скор каждой статьи для одного запроса.
    article_ids  : [A] int   — идентификаторы статей (в том же порядке).
    k            : сколько вернуть (усечётся до числа статей).
    """
    n = len(scores)
    k = min(k, n)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    # lexsort: последний ключ — главный. Главный ключ -scores (по убыванию),
    # вторичный article_ids (по возрастанию) — так ничьи упорядочены стабильно.
    order = np.lexsort((article_ids, -scores))
    return order[:k]


@dataclass
class ScoreMatrix:
    """Плотная матрица скоров запрос × статья.

    query_ids   : [Q] int64 — идентификаторы запросов (порядок строк).
    article_ids : [A] int64 — идентификаторы статей (порядок столбцов).
    scores      : [Q, A] float32 — скор пары (запрос, статья).
    source      : имя компонента-источника ("bm25", "char_tfidf", ...).
    """

    query_ids: np.ndarray
    article_ids: np.ndarray
    scores: np.ndarray
    source: str

    def __post_init__(self) -> None:
        self.query_ids = np.asarray(self.query_ids, dtype=np.int64)
        self.article_ids = np.asarray(self.article_ids, dtype=np.int64)
        self.scores = np.asarray(self.scores, dtype=np.float32)
        q, a = self.scores.shape
        if q != len(self.query_ids) or a != len(self.article_ids):
            raise ValueError(
                f"форма scores {self.scores.shape} не согласована с "
                f"query_ids ({len(self.query_ids)}) / article_ids ({len(self.article_ids)})"
            )

    @property
    def shape(self) -> tuple[int, int]:
        return self.scores.shape  # type: ignore[return-value]

    def rankings(self, k: int) -> dict[int, list[int]]:
        """{query_id: [article_id, ...]} — топ-k статей на каждый запрос."""
        out: dict[int, list[int]] = {}
        for qi, qid in enumerate(self.query_ids):
            idx = rank_indices(self.scores[qi], self.article_ids, k)
            out[int(qid)] = self.article_ids[idx].tolist()
        return out

    def to_retrieval_frame(self, k: int) -> pd.DataFrame:
        """Длинный формат RETRIEVAL_COLUMNS с топ-k статей на запрос."""
        rows_q: list[int] = []
        rows_a: list[int] = []
        rows_s: list[float] = []
        rows_r: list[int] = []
        for qi, qid in enumerate(self.query_ids):
            s = self.scores[qi]
            idx = rank_indices(s, self.article_ids, k)
            rows_q.extend([int(qid)] * len(idx))
            rows_a.extend(self.article_ids[idx].tolist())
            rows_s.extend(s[idx].tolist())
            rows_r.extend(range(len(idx)))
        return pd.DataFrame(
            {
                "query_id": np.asarray(rows_q, dtype=np.int64),
                "article_id": np.asarray(rows_a, dtype=np.int64),
                "score": np.asarray(rows_s, dtype=np.float32),
                "rank": np.asarray(rows_r, dtype=np.int64),
                "source": self.source,
            }
        )

    def save(self, path: str | Path) -> Path:
        """Сохранить в .npz (без manifest — его пишет pipeline/artifacts)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            query_ids=self.query_ids,
            article_ids=self.article_ids,
            scores=self.scores,
            source=np.asarray(self.source),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ScoreMatrix":
        data = np.load(Path(path), allow_pickle=False)
        return cls(
            query_ids=data["query_ids"],
            article_ids=data["article_ids"],
            scores=data["scores"],
            source=str(data["source"]),
        )
