"""Логистическая регрессия и LR-варианты слияния/блэнда."""
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
from support_search.data.splits import make_splits  # noqa: E402
from support_search.fusion import oof_fusion_ltr  # noqa: E402
from support_search.ltr import LogisticRegressionLTR, reciprocal_ranks  # noqa: E402
from support_search.ranking import oof_blend_ltr  # noqa: E402
from support_search.rerank.apply import NON_CANDIDATE  # noqa: E402


class TestLogisticRegression(unittest.TestCase):
    def test_separates_and_weights_informative_feature(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 2))
        y = (X[:, 0] + 0.1 * rng.normal(size=300) > 0).astype(float)  # зависит от признака 0
        lr = LogisticRegressionLTR(l2=0.1).fit(X, y)
        acc = ((lr.predict_proba(X) > 0.5) == y).mean()
        self.assertGreater(acc, 0.85)
        coef = lr.coefficients(["a", "b"])
        self.assertGreater(coef["a"], abs(coef["b"]))  # признак 0 важнее

    def test_class_balancing_handles_imbalance(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(500, 1))
        y = np.zeros(500)
        y[X[:, 0] > 1.5] = 1.0  # ~7% позитивов
        lr = LogisticRegressionLTR(l2=0.1, class_weight="balanced").fit(X, y)
        # сбалансированный LR ловит большинство редких позитивов (recall高)
        recall = (lr.predict_proba(X[y == 1]) > 0.5).mean()
        self.assertGreater(recall, 0.7)


class TestReciprocalRanks(unittest.TestCase):
    def test_values(self):
        rr = reciprocal_ranks(np.array([[0.1, 0.9, 0.5]]))
        self.assertEqual(rr[0, 1], 1.0)          # 0.9 — ранг 0
        self.assertAlmostEqual(rr[0, 2], 0.5, places=6)  # 0.5 — ранг 1
        self.assertAlmostEqual(rr[0, 0], 1 / 3, places=6)


def _splits(n):
    return make_splits({q: 1 for q in range(1, n + 1)}, n_splits=5, holdout_frac=0.2, seed=0)


class TestFusionLTR(unittest.TestCase):
    def test_oof_covers_dev(self):
        n, aids = 25, [10, 20, 30, 40]
        rng = np.random.default_rng(0)
        gt = {q: {aids[q % 4]} for q in range(1, n + 1)}
        # источник a "знает" ответ (высокий скор у GT), b — шум.
        a = np.full((n, 4), 0.1, dtype=np.float32)
        for i, q in enumerate(range(1, n + 1)):
            a[i, aids.index(aids[q % 4])] = 0.9
        sm_a = ScoreMatrix(list(range(1, n + 1)), aids, a, "a")
        sm_b = ScoreMatrix(list(range(1, n + 1)), aids, rng.random((n, 4)).astype(np.float32), "b")
        oof, feats, oof_m = oof_fusion_ltr({"a": sm_a, "b": sm_b}, gt, _splits(n), names=["a", "b"], depth=4)
        self.assertEqual(set(oof), set(_splits(n).dev))
        self.assertIn("a_norm", feats)
        self.assertIn("a_rr", feats)
        self.assertEqual(oof_m.shape, (n, 4))


class TestBlendLTR(unittest.TestCase):
    def test_oof_covers_dev_and_candidates_on_top(self):
        n, aids = 25, [10, 20, 30, 40]
        rng = np.random.default_rng(2)
        qids = list(range(1, n + 1))
        # кандидаты — статьи 10,20 (столбцы 0,1); остальное — сентинел.
        rer = np.full((n, 4), NON_CANDIDATE, dtype=np.float32)
        rer[:, :2] = rng.random((n, 2))
        reranker = ScoreMatrix(qids, aids, rer, "reranker")
        src = {"bm25": ScoreMatrix(qids, aids, rng.random((n, 4)).astype(np.float32), "bm25")}
        gt = {q: {aids[q % 2]} for q in qids}  # GT среди кандидатов
        oof, feats = oof_blend_ltr(src, reranker, gt, _splits(n), names=["bm25"], depth=4)
        self.assertEqual(set(oof), set(_splits(n).dev))
        self.assertIn("reranker_norm", feats)
        for ranking in oof.values():
            self.assertLessEqual(set(ranking[:2]), {10, 20})  # кандидаты — сверху


if __name__ == "__main__":
    unittest.main()
