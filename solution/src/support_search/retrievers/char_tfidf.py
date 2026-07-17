"""Char-n-gram TF-IDF ретривер — дешёвая устойчивость к опечаткам.

Запросы поддержки полны опечаток («Москлвской», «прийдут», «возрат»), где BM25 по
леммам слепнет. Символьные n-граммы 3–5 с границами слов (`char_wb`) ловят
пересечение по подстрокам и потому устойчивы к морфологии и опечаткам без
лемматизации.

Реализация без sklearn (его нет в целевом окружении): свой словарь n-грамм с
`min_df`, сглаженный IDF как в sklearn `TfidfVectorizer(smooth_idf=True)`,
sublinear TF и L2-нормализация — тогда скор пары = косинус между разреженными
tf-idf векторами, а вся матрица скоров = одно умножение Q · Dᵀ. Заголовок бустится
тем же весом `title_weight`, что и в BM25.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix, diags, load_npz, save_npz

from ..contracts import ScoreMatrix
from ..data.io import read_json, write_json
from ..logging_utils import get_logger
from .base import Corpus, Retriever

logger = get_logger("retrievers.char_tfidf")


def _l2_normalize_rows(matrix: csr_matrix) -> csr_matrix:
    """Поделить каждую строку на её L2-норму (нулевые строки оставить нулевыми)."""
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    norms[norms == 0.0] = 1.0
    return (diags(1.0 / norms) @ matrix).tocsr()


class CharTfidfRetriever(Retriever):
    """TF-IDF по символьным n-граммам (char_wb) + косинусная близость."""

    name = "char_tfidf"

    def __init__(
        self,
        *,
        ngram_min: int = 3,
        ngram_max: int = 5,
        min_df: int = 2,
        sublinear_tf: bool = True,
        title_weight: float = 2.0,
    ) -> None:
        if not 1 <= ngram_min <= ngram_max:
            raise ValueError(f"ожидалось 1 <= ngram_min <= ngram_max, дано {ngram_min}..{ngram_max}")
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.title_weight = float(title_weight)
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.empty(0, dtype=np.float32)
        self.article_ids: np.ndarray = np.empty(0, dtype=np.int64)
        self._doc_matrix: csr_matrix | None = None  # [A × V], L2-нормированный tf-idf

    # ─── извлечение n-грамм ──────────────────────────────────────────────
    def _char_wb_ngrams(self, text: str) -> list[str]:
        """Символьные n-граммы внутри границ слов; края дополняются пробелом."""
        grams: list[str] = []
        for word in text.lower().split():
            padded = f" {word} "
            length = len(padded)
            for n in range(self.ngram_min, self.ngram_max + 1):
                if length < n:
                    continue
                grams.extend(padded[i : i + n] for i in range(length - n + 1))
        return grams

    def _doc_counts(self, title: str, body: str) -> Counter:
        """Взвешенные частоты n-грамм документа: тело + title_weight · заголовок."""
        counts: Counter = Counter(self._char_wb_ngrams(body))
        if title and title.strip():
            for gram, cnt in Counter(self._char_wb_ngrams(title)).items():
                counts[gram] += self.title_weight * cnt
        return counts

    # ─── обучение ────────────────────────────────────────────────────────
    def fit(self, corpus: Corpus, train_queries: Sequence[str] | None = None) -> "CharTfidfRetriever":
        doc_counts = [self._doc_counts(t, b) for t, b in zip(corpus.titles, corpus.texts)]

        n_docs = len(corpus)
        document_freq: Counter = Counter()
        for counts in doc_counts:
            document_freq.update(counts.keys())  # присутствие n-граммы в документе

        # Словарь — n-граммы, встретившиеся минимум в min_df документах.
        self.vocab = {gram: i for i, gram in enumerate(g for g, d in document_freq.items() if d >= self.min_df)}
        self.idf = np.zeros(len(self.vocab), dtype=np.float32)
        for gram, idx in self.vocab.items():
            self.idf[idx] = math.log((1.0 + n_docs) / (1.0 + document_freq[gram])) + 1.0  # smooth idf

        self._doc_matrix = self._vectorize(doc_counts)
        self.article_ids = np.asarray(corpus.article_ids, dtype=np.int64)

        logger.info(
            "char_tfidf fit: docs=%d vocab=%d (ngrams %d-%d, min_df=%d, w_title=%.2f)",
            n_docs, len(self.vocab), self.ngram_min, self.ngram_max, self.min_df, self.title_weight,
        )
        return self

    def _vectorize(self, counts_list: list[Counter]) -> csr_matrix:
        """Список частот n-грамм → L2-нормированная разреженная tf-idf матрица."""
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for i, counts in enumerate(counts_list):
            for gram, cnt in counts.items():
                idx = self.vocab.get(gram)
                if idx is None:
                    continue
                tf = 1.0 + math.log(cnt) if (self.sublinear_tf and cnt > 0) else float(cnt)
                rows.append(i)
                cols.append(idx)
                vals.append(tf * float(self.idf[idx]))
        matrix = csr_matrix(
            (np.asarray(vals, dtype=np.float32), (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
            shape=(len(counts_list), len(self.vocab)),
            dtype=np.float32,
        )
        return _l2_normalize_rows(matrix)

    # ─── инференс ────────────────────────────────────────────────────────
    def score_matrix(self, queries: Sequence[str], query_ids: Sequence[int]) -> ScoreMatrix:
        if self._doc_matrix is None:
            raise RuntimeError("CharTfidfRetriever.score_matrix вызван до fit()")
        query_counts = [Counter(self._char_wb_ngrams(q)) for q in queries]
        query_matrix = self._vectorize(query_counts)  # [Q × V], нормирован
        scores = (query_matrix @ self._doc_matrix.T).toarray().astype(np.float32)  # косинус [Q × A]
        return ScoreMatrix(query_ids=np.asarray(query_ids), article_ids=self.article_ids, scores=scores, source=self.name)

    # ─── персистентность ─────────────────────────────────────────────────
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        assert self._doc_matrix is not None, "нечего сохранять: вызовите fit()"
        save_npz(path / "doc_matrix.npz", self._doc_matrix)
        np.save(path / "idf.npy", self.idf)
        np.save(path / "article_ids.npy", self.article_ids)
        write_json(path / "vocab.json", self.vocab)
        write_json(
            path / "meta.json",
            {"name": self.name, "ngram_min": self.ngram_min, "ngram_max": self.ngram_max,
             "min_df": self.min_df, "sublinear_tf": self.sublinear_tf, "title_weight": self.title_weight,
             "n_docs": int(len(self.article_ids)), "vocab_size": len(self.vocab)},
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CharTfidfRetriever":
        path = Path(path)
        meta = read_json(path / "meta.json")
        obj = cls(
            ngram_min=meta["ngram_min"], ngram_max=meta["ngram_max"], min_df=meta["min_df"],
            sublinear_tf=meta["sublinear_tf"], title_weight=meta["title_weight"],
        )
        obj._doc_matrix = load_npz(path / "doc_matrix.npz").tocsr()
        obj.idf = np.load(path / "idf.npy")
        obj.article_ids = np.load(path / "article_ids.npy")
        obj.vocab = {k: int(v) for k, v in read_json(path / "vocab.json").items()}
        return obj
