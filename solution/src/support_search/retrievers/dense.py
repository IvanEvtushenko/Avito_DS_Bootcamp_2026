"""Dense-ретривер (bi-encoder) — закрывает лексический разрыв.

BM25 и char-TF-IDF слепы к перефразировкам («где мои деньги» vs «Сроки возврата
денежных средств»). Bi-encoder кодирует запрос и статью в общее семантическое
пространство, где такие пары близки.

Устройство (план §6, этап 4):

- статья представляется чанками (~400 токенов, перекрытие, префикс заголовка) и
  отдельно заголовком; всё кодируется энкодером как пассажи и кэшируется;
- скор статьи = **max по косинусам чанков** + `w_title` · косинус заголовка;
  max-pool берёт лучший фрагмент длинной статьи и не страдает от статьи-выброса;
- энкодер внедряется (`Encoder`): в проде E5, в тестах — заглушка (без GPU/сети).

Косинус = скалярное произведение L2-нормированных векторов (нормализацию даёт
энкодер), поэтому вся матрица скоров — одно плотное умножение.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ..contracts import ScoreMatrix
from ..data.io import read_json, write_json
from ..logging_utils import get_logger
from ..preprocess import chunk_text
from .base import Corpus, Retriever
from .encoders import Encoder

logger = get_logger("retrievers.dense")


class DenseRetriever(Retriever):
    """Bi-encoder ретривер: max по чанкам + вклад заголовка."""

    name = "dense"

    def __init__(
        self,
        encoder: Encoder,
        *,
        max_tokens: int = 400,
        overlap: int = 100,
        max_chunks_per_article: int = 40,
        title_prefix: bool = True,
        w_title: float = 0.5,
    ) -> None:
        self.encoder = encoder
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.max_chunks_per_article = max_chunks_per_article
        self.title_prefix = title_prefix
        self.w_title = float(w_title)
        self.article_ids: np.ndarray = np.empty(0, dtype=np.int64)
        self.chunk_emb: np.ndarray = np.empty((0, 0), dtype=np.float32)   # [C, D]
        self.title_emb: np.ndarray = np.empty((0, 0), dtype=np.float32)   # [A, D]
        self.chunk_offsets: np.ndarray = np.empty(0, dtype=np.int64)      # [A] старт чанков статьи

    def _article_chunks(self, title: str, text: str) -> list[str]:
        chunks = chunk_text(
            text, title=title, max_tokens=self.max_tokens, overlap=self.overlap,
            max_chunks=self.max_chunks_per_article, title_prefix=self.title_prefix,
        )
        return chunks or [title.strip() or " "]  # гарантируем ≥1 чанк на статью

    def fit(self, corpus: Corpus, train_queries: Sequence[str] | None = None) -> "DenseRetriever":
        all_chunks: list[str] = []
        offsets: list[int] = []
        for title, text in zip(corpus.titles, corpus.texts):
            offsets.append(len(all_chunks))
            all_chunks.extend(self._article_chunks(title, text))

        self.chunk_offsets = np.asarray(offsets, dtype=np.int64)
        self.chunk_emb = self.encoder.encode_passages(all_chunks)
        self.title_emb = self.encoder.encode_passages([str(t) for t in corpus.titles])
        self.article_ids = np.asarray(corpus.article_ids, dtype=np.int64)

        logger.info(
            "dense fit: articles=%d chunks=%d dim=%d (w_title=%.2f, max_tokens=%d)",
            len(corpus), len(all_chunks), self.chunk_emb.shape[1], self.w_title, self.max_tokens,
        )
        return self

    def score_matrix(self, queries: Sequence[str], query_ids: Sequence[int]) -> ScoreMatrix:
        if self.chunk_emb.size == 0:
            raise RuntimeError("DenseRetriever.score_matrix вызван до fit()")
        q_emb = self.encoder.encode_queries(list(queries))          # [Q, D]
        chunk_sim = q_emb @ self.chunk_emb.T                        # [Q, C]
        # max по чанкам каждой статьи: reduceat режет по стартовым офсетам статей.
        article_max = np.maximum.reduceat(chunk_sim, self.chunk_offsets, axis=1)  # [Q, A]
        title_sim = q_emb @ self.title_emb.T                       # [Q, A]
        scores = (article_max + self.w_title * title_sim).astype(np.float32)
        return ScoreMatrix(query_ids=np.asarray(query_ids), article_ids=self.article_ids, scores=scores, source=self.name)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path / "embeddings.npz",
            chunk_emb=self.chunk_emb, title_emb=self.title_emb,
            chunk_offsets=self.chunk_offsets, article_ids=self.article_ids,
        )
        write_json(
            path / "meta.json",
            {"name": self.name, "w_title": self.w_title, "max_tokens": self.max_tokens,
             "overlap": self.overlap, "max_chunks_per_article": self.max_chunks_per_article,
             "title_prefix": self.title_prefix, "encoder": self.encoder.info(),
             "n_articles": int(len(self.article_ids)), "n_chunks": int(self.chunk_emb.shape[0])},
        )
        return path

    @classmethod
    def load(cls, path: str | Path, encoder: Encoder | None = None) -> "DenseRetriever":
        path = Path(path)
        meta = read_json(path / "meta.json")
        if encoder is None:
            raise ValueError("DenseRetriever.load требует encoder (общий с индексацией)")
        obj = cls(
            encoder, max_tokens=meta["max_tokens"], overlap=meta["overlap"],
            max_chunks_per_article=meta["max_chunks_per_article"],
            title_prefix=meta["title_prefix"], w_title=meta["w_title"],
        )
        data = np.load(path / "embeddings.npz")
        obj.chunk_emb = data["chunk_emb"]
        obj.title_emb = data["title_emb"]
        obj.chunk_offsets = data["chunk_offsets"]
        obj.article_ids = data["article_ids"]
        return obj
