"""Dense-ретривер на заглушке-энкодере (без GPU/сети): логика max-по-чанкам,
вклад заголовка, save/load."""
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
from support_search.retrievers import DenseRetriever, HashingStubEncoder, build_corpus  # noqa: E402


class TestDenseRetriever(unittest.TestCase):
    def setUp(self):
        self.corpus = build_corpus(mini_articles())
        self.encoder = HashingStubEncoder(dim=512, seed=1)
        self.retriever = DenseRetriever(
            self.encoder, max_tokens=40, overlap=10, max_chunks_per_article=8, w_title=0.5
        ).fit(self.corpus)
        self.queries = ["как отправить товар доставкой", "профиль заблокирован"]
        self.qids = [1, 2]

    def test_score_matrix_shape_and_finite(self):
        sm = self.retriever.score_matrix(self.queries, self.qids)
        self.assertEqual(sm.shape, (2, len(self.corpus)))
        self.assertTrue(np.isfinite(sm.scores).all())

    def test_relevant_article_ranked_high(self):
        # Заглушка лексическая → релевантная статья должна попасть в топ.
        sm = self.retriever.score_matrix(self.queries, self.qids)
        rankings = sm.rankings(k=3)
        self.assertIn(101, rankings[1][:2])   # отправка товара
        self.assertIn(104, rankings[2][:2])   # блокировка профиля

    def test_every_article_has_chunk(self):
        # chunk_offsets строго возрастают → у каждой статьи ≥1 чанк (reduceat корректен).
        offs = self.retriever.chunk_offsets
        self.assertEqual(len(offs), len(self.corpus))
        self.assertTrue(np.all(np.diff(offs) >= 1))

    def test_save_load_roundtrip(self):
        before = self.retriever.score_matrix(self.queries, self.qids)
        with tempfile.TemporaryDirectory() as d:
            self.retriever.save(d)
            loaded = DenseRetriever.load(d, encoder=self.encoder)
        after = loaded.score_matrix(self.queries, self.qids)
        np.testing.assert_allclose(before.scores, after.scores, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
