"""Ретриверы на игрушечном корпусе: релевантная статья наверху, save/load, опечатки."""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

_SOL = pathlib.Path(__file__).resolve().parents[2]
for _p in (_SOL / "src", _SOL / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

from mini_data import mini_articles  # noqa: E402
from support_search.preprocess import Tokenizer  # noqa: E402
from support_search.retrievers import BM25Retriever, CharTfidfRetriever, build_corpus  # noqa: E402


class _RetrieverContractMixin:
    """Общие проверки, одинаковые для всех ретриверов (симметричное сравнение)."""

    def _make(self):  # переопределяется
        raise NotImplementedError

    def setUp(self):
        self.corpus = build_corpus(mini_articles())
        self.retriever = self._make().fit(self.corpus)
        self.qids = [1, 2, 3]
        self.queries = [
            "как отправить товар доставкой",
            "вернуть деньги за заказ",
            "профиль заблокирован",
        ]

    def test_score_matrix_shape(self):
        sm = self.retriever.score_matrix(self.queries, self.qids)
        self.assertEqual(sm.shape, (3, len(self.corpus)))
        self.assertTrue(np.isfinite(sm.scores).all())

    def test_relevant_article_on_top(self):
        sm = self.retriever.score_matrix(self.queries, self.qids)
        rankings = sm.rankings(k=3)
        self.assertEqual(rankings[1][0], 101)  # отправка товара
        self.assertIn(104, rankings[3][:2])    # блокировка профиля

    def test_retrieve_frame_top_k(self):
        frame = self.retriever.retrieve(self.queries, self.qids, top_k=5)
        self.assertEqual(len(frame), 15)
        self.assertEqual(frame["source"].unique().tolist(), [self.retriever.name])


class TestBM25(_RetrieverContractMixin, unittest.TestCase):
    def _make(self):
        return BM25Retriever(Tokenizer(lemmatizer="none", tokenizer="regex"), title_weight=2.0)

    def test_save_load_roundtrip(self):
        sm_before = self.retriever.score_matrix(self.queries, self.qids)
        with tempfile.TemporaryDirectory() as d:
            self.retriever.save(d)
            loaded = BM25Retriever.load(d, tokenizer=Tokenizer(lemmatizer="none", tokenizer="regex"))
        sm_after = loaded.score_matrix(self.queries, self.qids)
        np.testing.assert_allclose(sm_before.scores, sm_after.scores, rtol=1e-5)


class TestCharTfidf(_RetrieverContractMixin, unittest.TestCase):
    def _make(self):
        return CharTfidfRetriever(ngram_min=3, ngram_max=5, min_df=1, title_weight=2.0)

    def test_typo_robustness(self):
        # «профиль» с опечаткой «профль» — char-n-граммы должны всё равно поднять 104.
        sm = self.retriever.score_matrix(["профль заблокирован"], [99])
        self.assertIn(104, sm.rankings(k=3)[99])

    def test_save_load_roundtrip(self):
        sm_before = self.retriever.score_matrix(self.queries, self.qids)
        with tempfile.TemporaryDirectory() as d:
            self.retriever.save(d)
            loaded = CharTfidfRetriever.load(d)
        sm_after = loaded.score_matrix(self.queries, self.qids)
        np.testing.assert_allclose(sm_before.scores, sm_after.scores, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
