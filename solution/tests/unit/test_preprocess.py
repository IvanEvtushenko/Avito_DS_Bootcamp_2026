"""Препроцессинг: очистка HTML, извлечение ссылок, токенизация, чанкинг."""
from __future__ import annotations

import pathlib
import sys
import unittest

_SOL = pathlib.Path(__file__).resolve().parents[2]
for _p in (_SOL / "src", _SOL / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from support_search.preprocess import (  # noqa: E402
    Tokenizer,
    chunk_text,
    chunk_tokens,
    extract_article_links,
    html_to_text,
)


class TestHtmlToText(unittest.TestCase):
    def test_drops_script_keeps_text(self):
        html = "<p>Привет<script>evil()</script> мир</p>"
        self.assertEqual(html_to_text(html), "Привет мир")

    def test_keeps_table_and_label_text(self):
        # Спойлеры/вкладки в данных — это <label>; текст таблиц должен сохраниться.
        html = "<label>Вкладка</label><table><tr><td>Ячейка</td></tr></table>"
        text = html_to_text(html)
        self.assertIn("Вкладка", text)
        self.assertIn("Ячейка", text)

    def test_collapses_whitespace_and_nbsp(self):
        self.assertEqual(html_to_text("<p>a\xa0\n  b</p>"), "a b")

    def test_empty(self):
        self.assertEqual(html_to_text(""), "")
        self.assertEqual(html_to_text(None), "")


class TestExtractLinks(unittest.TestCase):
    def test_extracts_and_dedupes(self):
        html = (
            '<a href="https://support.avito.ru/articles/101">a</a>'
            '<a href="/articles/101">b</a><a href="/articles/202">c</a>'
        )
        self.assertEqual(extract_article_links(html), [101, 202])


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        # Плейсхолдеры и стоп-слова проверяются независимо от бэкенда лемматизации.
        self.tok = Tokenizer(
            lemmatizer="none", tokenizer="regex", min_len=2, use_stopwords=True,
            placeholders={"<MONEY>": "деньги", "<URL>": "ссылка", "<ID>": ""},
        )

    def test_placeholder_replacement(self):
        tokens = self.tok("Оплатил <MONEY> и перешёл по <URL>")
        self.assertIn("деньги", tokens)
        self.assertIn("ссылка", tokens)

    def test_placeholder_drop(self):
        tokens = self.tok("код <ID> получен")
        self.assertNotIn("id", tokens)
        self.assertTrue(all("<" not in t and ">" not in t for t in tokens))

    def test_stopwords_and_minlen_and_lowercase(self):
        tokens = self.tok("Я и ты")  # все — стоп-слова/короткие
        self.assertEqual(tokens, [])
        self.assertIn("товар", self.tok("Товар"))  # нижний регистр

    def test_punctuation_removed(self):
        self.assertEqual(self.tok("привет, мир!"), ["привет", "мир"])


class TestChunking(unittest.TestCase):
    def test_sliding_window_covers_tail(self):
        tokens = [str(i) for i in range(10)]
        chunks = chunk_tokens(tokens, max_tokens=4, overlap=1)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(len(c) == 4 for c in chunks))
        self.assertEqual(chunks[-1], ["6", "7", "8", "9"])

    def test_max_chunks_caps(self):
        tokens = [str(i) for i in range(100)]
        chunks = chunk_tokens(tokens, max_tokens=10, overlap=2, max_chunks=3)
        self.assertEqual(len(chunks), 3)

    def test_overlap_must_be_valid(self):
        with self.assertRaises(ValueError):
            chunk_tokens(["a", "b"], max_tokens=2, overlap=2)

    def test_chunk_text_title_prefix(self):
        chunks = chunk_text("один два три", title="Заголовок", max_tokens=2, overlap=0)
        self.assertTrue(all(c.startswith("Заголовок. ") for c in chunks))

    def test_empty_text_with_title_returns_title(self):
        # Пустой текст со заголовком → один чанк из заголовка (статья не исчезает).
        self.assertEqual(chunk_text("", title="Только заголовок"), ["Только заголовок."])
        self.assertEqual(chunk_text("", title=""), [])


if __name__ == "__main__":
    unittest.main()
