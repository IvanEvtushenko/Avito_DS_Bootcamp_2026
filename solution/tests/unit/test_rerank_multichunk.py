"""Мульти-чанковый реранк (идея №3): max по чанкам, listwise-путь, top_chunk_texts."""
from __future__ import annotations

import pathlib
import sys
import unittest

_SOL = pathlib.Path(__file__).resolve().parents[2]
for _p in (_SOL / "src", _SOL / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

from support_search.contracts import ScoreMatrix  # noqa: E402
from support_search.rerank import (  # noqa: E402
    NON_CANDIDATE,
    LexicalStubReranker,
    candidate_lists,
    rerank_to_matrix,
)
from support_search.retrievers.dense import DenseRetriever  # noqa: E402


class _ListwiseStub:
    """Листовая заглушка: скор = доля слов запроса в пассаже (как LexicalStubReranker)."""

    def score_listwise(self, query: str, passages):
        return LexicalStubReranker().score_pairs([query] * len(passages), list(passages))

    def score_pairs(self, queries, passages):  # не должен вызываться в листовом пути
        raise AssertionError("для листового реранкера rerank_to_matrix обязан звать score_listwise")

    def info(self):
        return {"type": "listwise_stub"}


def _fusion_1q() -> tuple[ScoreMatrix, list[list[int]]]:
    fusion = ScoreMatrix([1], [10, 20], np.array([[0.9, 0.8]], np.float32), "fusion")
    return fusion, candidate_lists(fusion, top_k=2)  # [10, 20]


class TestMultiChunkMaxPool(unittest.TestCase):
    def test_article_wins_by_second_chunk(self):
        """Релевантный ВТОРОЙ чанк должен поднять статью (max, а не первый чанк)."""
        fusion, cands = _fusion_1q()
        passages = [[
            ["про доставку товара", "про оплату картой"],          # статья 10: оба чанка мимо
            ["про профиль", "как вернуть деньги за заказ"],        # статья 20: второй чанк — попадание
        ]]
        m = rerank_to_matrix(LexicalStubReranker(), ["вернуть деньги"], [1], fusion.article_ids, cands, passages)
        self.assertEqual(m.rankings(2)[1][0], 20)

    def test_string_passages_backward_compatible(self):
        """Строковые пассажи работают как раньше (один чанк на статью)."""
        fusion, cands = _fusion_1q()
        m = rerank_to_matrix(
            LexicalStubReranker(), ["вернуть деньги"], [1], fusion.article_ids, cands,
            [["про доставку", "вернуть деньги"]],
        )
        self.assertEqual(m.rankings(2)[1][0], 20)

    def test_mixed_str_and_list(self):
        """Смесь форматов в одной строке допустима (str и список чанков)."""
        fusion, cands = _fusion_1q()
        m = rerank_to_matrix(
            LexicalStubReranker(), ["вернуть деньги"], [1], fusion.article_ids, cands,
            [["про доставку", ["мимо", "вернуть деньги мгновенно"]]],
        )
        self.assertEqual(m.rankings(2)[1][0], 20)

    def test_non_candidate_keeps_sentinel(self):
        """Не-кандидат остаётся NON_CANDIDATE и при мульти-чанках."""
        fusion = ScoreMatrix([1], [10, 20, 30], np.array([[0.1, 0.9, 0.5]], np.float32), "fusion")
        cands = candidate_lists(fusion, top_k=2)  # [20, 30]
        m = rerank_to_matrix(
            LexicalStubReranker(), ["вернуть деньги"], [1], fusion.article_ids, cands,
            [[["a", "b"], ["c"]]],
        )
        self.assertLess(m.scores[0, 0], NON_CANDIDATE / 2)

    def test_listwise_path_maxpools_per_article(self):
        """Листовой реранкер получает плоский список чанков, скор статьи = max её чанков."""
        fusion, cands = _fusion_1q()
        passages = [[
            ["про доставку товара", "про оплату картой"],
            ["про профиль", "как вернуть деньги за заказ"],
        ]]
        m = rerank_to_matrix(_ListwiseStub(), ["вернуть деньги"], [1], fusion.article_ids, cands, passages)
        self.assertEqual(m.rankings(2)[1][0], 20)
        # и оба кандидата получили реальные скоры (не сентинел)
        self.assertTrue(np.all(m.scores[0] > NON_CANDIDATE / 2))


class TestTopChunkTexts(unittest.TestCase):
    def _retriever(self) -> DenseRetriever:
        dr = DenseRetriever(encoder=None)
        dr.article_ids = np.array([10, 20])
        dr.chunk_offsets = np.array([0, 3])          # статья 10: чанки 0..2; статья 20: чанк 3
        dr.chunk_texts = ["a0", "a1", "a2", "b0"]
        dr.chunk_emb = np.array([[0.1], [0.9], [0.5], [1.0]], np.float32)
        return dr

    def test_orders_by_cosine_and_truncates(self):
        out = self._retriever().top_chunk_texts(np.array([[1.0]], np.float32), [[10, 20]], m=2)
        self.assertEqual(out[0][0], ["a1", "a2"])    # топ-2 чанка статьи 10 по косинусу
        self.assertEqual(out[0][1], ["b0"])          # у статьи 20 один чанк < m — не падает

    def test_m1_matches_best_chunk(self):
        dr = self._retriever()
        q = np.array([[1.0]], np.float32)
        top1 = dr.top_chunk_texts(q, [[10, 20]], m=1)
        best = dr.best_chunk_texts(q, [[10, 20]])
        self.assertEqual([[c[0] for c in row] for row in top1], best)


if __name__ == "__main__":
    unittest.main()
