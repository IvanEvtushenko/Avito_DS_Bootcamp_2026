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
        self.chunk_texts: list[str] = []                                  # [C] тексты чанков

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
        self.chunk_texts = all_chunks
        self.chunk_emb = self.encoder.encode_passages(all_chunks)
        self.title_emb = self.encoder.encode_passages([str(t) for t in corpus.titles])
        self.article_ids = np.asarray(corpus.article_ids, dtype=np.int64)

        logger.info(
            "dense fit: articles=%d chunks=%d dim=%d (w_title=%.2f, max_tokens=%d)",
            len(corpus), len(all_chunks), self.chunk_emb.shape[1], self.w_title, self.max_tokens,
        )
        return self

    def score_from_query_embeddings(self, query_embeddings: np.ndarray, query_ids: Sequence[int]) -> ScoreMatrix:
        """Матрица скоров из готовых эмбеддингов запросов (без повторного кодирования)."""
        if self.chunk_emb.size == 0:
            raise RuntimeError("DenseRetriever вызван до fit()")
        chunk_sim = query_embeddings @ self.chunk_emb.T             # [Q, C]
        # max по чанкам каждой статьи: reduceat режет по стартовым офсетам статей.
        article_max = np.maximum.reduceat(chunk_sim, self.chunk_offsets, axis=1)  # [Q, A]
        title_sim = query_embeddings @ self.title_emb.T            # [Q, A]
        scores = (article_max + self.w_title * title_sim).astype(np.float32)
        return ScoreMatrix(query_ids=np.asarray(query_ids), article_ids=self.article_ids, scores=scores, source=self.name)

    def score_matrix(self, queries: Sequence[str], query_ids: Sequence[int]) -> ScoreMatrix:
        return self.score_from_query_embeddings(self.encode_queries(queries), query_ids)

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        """L2-нормированные эмбеддинги запросов [Q, D] (для выбора лучшего чанка)."""
        return self.encoder.encode_queries(list(queries))

    def best_chunk_texts(
        self, query_embeddings: np.ndarray, article_id_lists: Sequence[Sequence[int]]
    ) -> list[list[str]]:
        """Для каждой пары (запрос, статья-кандидат) — текст лучшего по косинусу чанка.

        Именно этот текст (уже с префиксом заголовка) идёт на вход cross-encoder
        реранкера: «запрос × лучший чанк статьи» (план §6, этап 6).
        """
        id_to_idx = {int(a): i for i, a in enumerate(self.article_ids)}
        n_chunks = self.chunk_emb.shape[0]
        out: list[list[str]] = []
        for q_i, article_ids in enumerate(article_id_lists):
            q_vec = query_embeddings[q_i]
            row: list[str] = []
            for aid in article_ids:
                a = id_to_idx[int(aid)]
                start = int(self.chunk_offsets[a])
                end = int(self.chunk_offsets[a + 1]) if a + 1 < len(self.chunk_offsets) else n_chunks
                best = start + int(np.argmax(self.chunk_emb[start:end] @ q_vec))
                row.append(self.chunk_texts[best])
            out.append(row)
        return out

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path / "embeddings.npz",
            chunk_emb=self.chunk_emb, title_emb=self.title_emb,
            chunk_offsets=self.chunk_offsets, article_ids=self.article_ids,
        )
        write_json(path / "chunk_texts.json", self.chunk_texts)
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
        """Загрузить индекс. `encoder` нужен только для кодирования новых запросов;
        без него доступны сохранённые эмбеддинги/чанки (напр. для реранкера)."""
        path = Path(path)
        meta = read_json(path / "meta.json")
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
        texts_path = path / "chunk_texts.json"
        obj.chunk_texts = read_json(texts_path) if texts_path.exists() else []
        return obj
