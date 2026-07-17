"""HTML статьи → чистый текст.

Тело статьи хранится с разметкой: таблицы, спойлеры/вкладки (`<label>`), списки,
анкоры ссылок, а также служебные `<input>`, `<img>`, `<script>`, `<style>`.
Задача (план §6, этап 2): убрать теги без полезного текста и сохранить всё
остальное, схлопнув пробелы.

Проверено на данных: осмысленный текст несут в т.ч. `<label>` (заголовки
вкладок) и ячейки таблиц — их `get_text` сохраняет; `<input>`/`<img>` текста не
несут. Самая длинная статья (901К символов) чистится в ~506К символов текста —
её обрабатывает чанкинг с лимитом числа чанков.
"""
from __future__ import annotations

import re
from typing import Sequence

from bs4 import BeautifulSoup

# Ссылки статьи друг на друга: support.avito.ru/articles/<ID> (в т.ч. без домена).
_ARTICLE_LINK_RE = re.compile(r"/articles/(\d+)")
_WS_RE = re.compile(r"\s+")

DEFAULT_DROP_TAGS = ("script", "style", "input", "img", "svg", "button")


def _collapse_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text.replace("\xa0", " ")).strip()


def html_to_text(html: str, *, drop_tags: Sequence[str] = DEFAULT_DROP_TAGS) -> str:
    """HTML → одна строка чистого текста.

    Теги из `drop_tags` удаляются целиком; из остального `get_text` собирает
    текст, вставляя пробел между узлами (слова не склеиваются), после чего
    пробелы схлопываются. Пустой/None вход даёт пустую строку.
    """
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    for tag_name in drop_tags:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    return _collapse_whitespace(soup.get_text(separator=" "))


def extract_article_links(html: str) -> list[int]:
    """ID статей, на которые ссылается данная (для графа ссылок, TODO §7.1).

    Не входит в лексический бейзлайн, но дёшево извлекается здесь же, чтобы не
    парсить HTML дважды на следующих этапах.
    """
    if not html:
        return []
    # dict.fromkeys сохраняет порядок и убирает повторные ссылки на одну статью.
    return [int(x) for x in dict.fromkeys(_ARTICLE_LINK_RE.findall(html))]
