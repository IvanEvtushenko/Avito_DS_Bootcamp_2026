"""Ретриверы: лексические (BM25, char-TF-IDF) и, позже, dense.

Все реализуют единый интерфейс `Retriever` (fit / score_matrix / retrieve /
save / load), поэтому включаются и сравниваются без изменений остального
пайплайна (план §5.1.3).
"""
from __future__ import annotations

from .base import Corpus, Retriever, build_corpus
from .bm25 import BM25Retriever
from .char_tfidf import CharTfidfRetriever

# Реестр по имени — CLI/pipeline включают ретривер одним ключом конфига (§5.1.5).
RETRIEVER_REGISTRY: dict[str, type[Retriever]] = {
    "bm25": BM25Retriever,
    "char_tfidf": CharTfidfRetriever,
}

__all__ = [
    "Corpus",
    "Retriever",
    "build_corpus",
    "BM25Retriever",
    "CharTfidfRetriever",
    "RETRIEVER_REGISTRY",
]
