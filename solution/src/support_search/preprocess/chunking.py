"""Разбиение длинной статьи на чанки ~400 токенов с перекрытием и префиксом.

Нужно двум потребителям: bi-encoder кодирует каждый чанк отдельно (скор статьи =
max по чанкам, план §4 этап 4), а лексика может индексировать чанки вместо
целого текста. Скользящее окно с перекрытием не режет мысль на границе, префикс
заголовка даёт каждому чанку контекст, а лимит числа чанков защищает пайплайн от
статьи на 901К символов (план §8).

Чанкинг работает на «единицах» (слова через пробел) — регистр и пунктуация
сохраняются, потому что энкодеру нужен естественный текст, а не леммы.
"""
from __future__ import annotations

from typing import Sequence


def chunk_tokens(
    tokens: Sequence[str],
    *,
    max_tokens: int = 400,
    overlap: int = 100,
    max_chunks: int | None = None,
) -> list[list[str]]:
    """Скользящее окно по токенам: окна длины `max_tokens`, шаг `max_tokens-overlap`.

    Последнее окно всегда покрывает хвост. `max_chunks` обрезает слишком длинные
    статьи (защита от выброса на 901К символов).
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens должен быть > 0, получено {max_tokens}")
    if not 0 <= overlap < max_tokens:
        raise ValueError(f"overlap должен быть в [0, max_tokens), получено {overlap}")

    n = len(tokens)
    if n == 0:
        return []

    step = max_tokens - overlap
    chunks: list[list[str]] = []
    for start in range(0, n, step):
        chunks.append(list(tokens[start : start + max_tokens]))
        if start + max_tokens >= n:  # хвост покрыт — дальше только дубли-перекрытия
            break
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
    return chunks


def chunk_text(
    text: str,
    *,
    title: str | None = None,
    max_tokens: int = 400,
    overlap: int = 100,
    max_chunks: int | None = None,
    title_prefix: bool = True,
) -> list[str]:
    """Текст статьи → список текстовых чанков; каждый с префиксом заголовка.

    Пустой текст со заголовком даёт один чанк из заголовка (статья не исчезает
    из индекса). Пустой текст без заголовка — пустой список.
    """
    prefix = f"{title.strip()}. " if (title_prefix and title and title.strip()) else ""
    words = text.split()
    token_chunks = chunk_tokens(words, max_tokens=max_tokens, overlap=overlap, max_chunks=max_chunks)

    if not token_chunks:
        return [prefix.strip()] if prefix else []
    return [prefix + " ".join(chunk) for chunk in token_chunks]
