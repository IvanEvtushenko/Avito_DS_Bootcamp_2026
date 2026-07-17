"""Препроцессинг: HTML → текст, нормализация/лемматизация, чанкинг."""
from __future__ import annotations

from .html import extract_article_links, html_to_text
from .normalize import Tokenizer
from .chunking import chunk_tokens, chunk_text

__all__ = [
    "extract_article_links",
    "html_to_text",
    "Tokenizer",
    "chunk_tokens",
    "chunk_text",
]
