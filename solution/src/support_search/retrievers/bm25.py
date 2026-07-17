"""BM25-ретривер (адаптация BM25-ядра автора под матрицу скоров запрос × статья).

Формула — Okapi BM25, как в исходном проекте:

                     idf(t) · tf(t,d) · (k1 + 1)
    score(t, d) = ───────────────────────────────────────
                  tf(t,d) + k1 · (1 − b + b · |d|/avgdl)

    idf(t) = log( (N − df + 0.5) / (df + 0.5) + 1 )

Здесь два отличия от исходника:

1. **Title boost (BM25F-lite).** Документ — не единый текст, а пара
   (заголовок, тело). Взвешенная частота термина = tf_body + w_title · tf_title;
   длина документа считается по той же взвешенной частоте. `title_weight` —
   дешёвый параметр совместного random search (план §4.3).
2. **Векторизация по всем запросам.** Корпус крошечный (500×793), поэтому
   BM25-веса термин–документ считаются один раз как разреженная матрица
   W[d,t], а скоры всех запросов — одним разреженным умножением Q_tf · Wᵀ.
   Результат идентичен посуммному скорингу по терминам запроса.

`scipy.sparse`, а не плотный numpy: матрица tf имеет форму [N_docs × V_vocab] и
на 99% состоит из нулей — CSR экономит память в сотни раз.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz

from ..contracts import ScoreMatrix
from ..data.io import read_json, write_json
from ..logging_utils import get_logger
from .base import Corpus, Retriever

logger = get_logger("retrievers.bm25")

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75

Tokenize = Callable[[str], list[str]]


class BM25Retriever(Retriever):
    """BM25 по леммам с бустом заголовка. `tokenizer` — общий на корпус и запросы."""

    name = "bm25"

    def __init__(
        self,
        tokenizer: Tokenize,
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        title_weight: float = 2.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.k1 = float(k1)
        self.b = float(b)
        self.title_weight = float(title_weight)
        self.vocab: dict[str, int] = {}
        self.article_ids: np.ndarray = np.empty(0, dtype=np.int64)
        self._weight: csr_matrix | None = None  # BM25-веса W[d,t], форма [A × V]

    # ─── обучение ────────────────────────────────────────────────────────
    def fit(self, corpus: Corpus, train_queries: Sequence[str] | None = None) -> "BM25Retriever":
        n_docs = len(corpus)
        vocab: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        doc_lens = np.zeros(n_docs, dtype=np.float32)

        for i, (title, text) in enumerate(zip(corpus.titles, corpus.texts)):
            combined: dict[str, float] = Counter(self.tokenizer(text))
            for term, cnt in Counter(self.tokenizer(title)).items():
                combined[term] = combined.get(term, 0.0) + self.title_weight * cnt
            for term, val in combined.items():
                idx = vocab.setdefault(term, len(vocab))
                rows.append(i)
                cols.append(idx)
                vals.append(float(val))
            doc_lens[i] = float(sum(combined.values()))

        vocab_size = len(vocab)
        tf = csr_matrix(
            (np.asarray(vals, dtype=np.float32), (np.asarray(rows), np.asarray(cols))),
            shape=(n_docs, vocab_size),
            dtype=np.float32,
        )
        df = np.asarray((tf > 0).sum(axis=0)).ravel().astype(np.int64)
        avgdl = float(doc_lens.mean()) or 1.0
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        length_norm = (1.0 - self.b) + self.b * (doc_lens / avgdl)  # [A]

        # W[d,t] = idf[t] · tf[d,t] · (k1+1) / (tf[d,t] + k1·length_norm[d]).
        coo = tf.tocoo()
        data = coo.data
        denom = data + self.k1 * length_norm[coo.row]
        w_data = (idf[coo.col] * data * (self.k1 + 1.0) / denom).astype(np.float32)
        self._weight = csr_matrix((w_data, (coo.row, coo.col)), shape=(n_docs, vocab_size))
        self.vocab = vocab
        self.article_ids = np.asarray(corpus.article_ids, dtype=np.int64)

        logger.info(
            "BM25 fit: docs=%d vocab=%d avgdl=%.1f nnz=%d (k1=%.2f b=%.2f w_title=%.2f)",
            n_docs, vocab_size, avgdl, tf.nnz, self.k1, self.b, self.title_weight,
        )
        return self

    # ─── инференс ────────────────────────────────────────────────────────
    def _query_tf_matrix(self, queries: Sequence[str]) -> csr_matrix:
        """Разреженная матрица частот терминов запросов [Q × V] (OOV отброшены)."""
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for qi, query in enumerate(queries):
            counts = Counter(t for t in self.tokenizer(query) if t in self.vocab)
            for term, cnt in counts.items():
                rows.append(qi)
                cols.append(self.vocab[term])
                vals.append(float(cnt))
        return csr_matrix(
            (np.asarray(vals, dtype=np.float32), (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
            shape=(len(queries), len(self.vocab)),
            dtype=np.float32,
        )

    def score_matrix(self, queries: Sequence[str], query_ids: Sequence[int]) -> ScoreMatrix:
        if self._weight is None:
            raise RuntimeError("BM25Retriever.score_matrix вызван до fit()")
        query_tf = self._query_tf_matrix(queries)
        scores = (query_tf @ self._weight.T).toarray().astype(np.float32)  # [Q × A]
        return ScoreMatrix(query_ids=np.asarray(query_ids), article_ids=self.article_ids, scores=scores, source=self.name)

    # ─── персистентность ─────────────────────────────────────────────────
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        assert self._weight is not None, "нечего сохранять: вызовите fit()"
        save_npz(path / "weight.npz", self._weight)
        np.save(path / "article_ids.npy", self.article_ids)
        write_json(path / "vocab.json", self.vocab)
        write_json(
            path / "meta.json",
            {"name": self.name, "k1": self.k1, "b": self.b, "title_weight": self.title_weight,
             "n_docs": int(len(self.article_ids)), "vocab_size": len(self.vocab)},
        )
        return path

    @classmethod
    def load(cls, path: str | Path, tokenizer: Tokenize | None = None) -> "BM25Retriever":
        path = Path(path)
        meta = read_json(path / "meta.json")
        if tokenizer is None:
            raise ValueError("BM25Retriever.load требует tokenizer (общий с индексацией)")
        obj = cls(tokenizer, k1=meta["k1"], b=meta["b"], title_weight=meta["title_weight"])
        obj._weight = load_npz(path / "weight.npz").tocsr()
        obj.article_ids = np.load(path / "article_ids.npy")
        obj.vocab = {k: int(v) for k, v in read_json(path / "vocab.json").items()}
        return obj
