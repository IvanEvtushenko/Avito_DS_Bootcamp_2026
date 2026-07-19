"""Граф-фичи кандидатов: сборка графов, LOO-протокол, listwise MLP-голова."""
from __future__ import annotations

import pathlib
import sys
import unittest

_SOL = pathlib.Path(__file__).resolve().parents[2]
for _p in (_SOL / "src", _SOL / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

from support_search.features import GraphFeaturizer, article_vectors, co_label_weights, link_adjacency  # noqa: E402
from support_search.ranking import MLPRankerLTR  # noqa: E402

AIDS = [10, 20, 30]


def _featurizer(*, link=(), vecs=None, top_m=2, interaction=False, co_weight="raw") -> GraphFeaturizer:
    adj = np.zeros((3, 3), dtype=bool)
    col = {a: j for j, a in enumerate(AIDS)}
    for a, b in link:
        adj[col[a], col[b]] = adj[col[b], col[a]] = True
    if vecs is None:
        vecs = np.eye(3, dtype=np.float32)  # ортогональны: sim=0 между разными
    return GraphFeaturizer(AIDS, link_adj=adj, article_vec=vecs, top_m=top_m,
                           interaction=interaction, co_weight=co_weight)


class TestGraphBuilders(unittest.TestCase):
    def test_link_adjacency_symmetric_and_ignores_unknown(self):
        bodies = ['<a href="/articles/20">см.</a> и /articles/999', "", ""]
        adj = link_adjacency(AIDS, bodies)
        self.assertTrue(adj[0, 1] and adj[1, 0])       # 10↔20, симметрично
        self.assertEqual(int(adj.sum()), 2)            # 999 нет в корпусе

    def test_co_label_weights_and_freq(self):
        gt = {1: {10, 20}, 2: {10, 20}, 3: {10}}
        weights, freq = co_label_weights(gt, AIDS)
        self.assertEqual(weights[0, 1], 2.0)           # пара (10,20) в двух запросах
        self.assertEqual(weights[0, 0], 0.0)           # диагональ пустая
        self.assertEqual(freq.tolist(), [3.0, 2.0, 0.0])

    def test_article_vectors_normalized_mean(self):
        chunk = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)  # статья A: 2 чанка, B: 1
        title = np.zeros((2, 2), dtype=np.float32)
        vec = article_vectors(chunk, title, np.array([0, 2]))
        self.assertEqual(vec.shape, (2, 2))
        np.testing.assert_allclose(np.linalg.norm(vec, axis=1), 1.0, atol=1e-6)


class TestGraphFeaturizer(unittest.TestCase):
    # Скоры: топ-2 кандидата запроса — статьи 10 и 20; кандидат 30 — «хвост».
    scores = np.array([[3.0, 2.0, 1.0]], dtype=np.float32)
    mask = np.ones((1, 3), dtype=bool)

    def test_val_query_uses_only_train_labels(self):
        # Запрос 99 (val) не в gt_train → его фичи зависят лишь от чужой разметки.
        feats = _featurizer().features(self.scores, self.mask, [99], gt_train={1: {10, 30}})
        co_w, link, sim, freq = feats[0, 2]            # кандидат 30 против лидеров {10, 20}
        self.assertAlmostEqual(co_w, np.log1p(1.0), places=6)   # пара (30,10) из запроса 1
        self.assertEqual(link, 0.0)
        self.assertEqual(sim, 0.0)                      # орт-векторы
        self.assertAlmostEqual(freq, np.log1p(1.0), places=6)

    def test_train_query_loo_removes_own_contribution(self):
        # Единственный источник пары (30,10) — сам запрос 1 → после LOO ноль.
        feats = _featurizer().features(self.scores, self.mask, [1], gt_train={1: {10, 30}})
        co_w, _, _, freq = feats[0, 2]
        self.assertEqual(co_w, 0.0)
        self.assertEqual(freq, 0.0)

    def test_link_feature_against_leaders(self):
        feats = _featurizer(link=[(30, 20)]).features(self.scores, self.mask, [99], gt_train={})
        self.assertEqual(feats[0, 2, 1], 1.0)          # 30 связан с лидером 20
        self.assertEqual(feats[0, 0, 1], 0.0)          # 10 ни с кем из лидеров

    def test_leader_excludes_itself_from_anchors(self):
        vecs = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)  # 10 и 20 совпадают
        feats = _featurizer(vecs=vecs).features(self.scores, self.mask, [99], gt_train={})
        self.assertAlmostEqual(feats[0, 0, 2], 1.0, places=6)  # sim топ-1 → топ-2, не к себе

    def test_interaction_is_co_w_times_margin(self):
        f = _featurizer(interaction=True)
        self.assertEqual(f.feature_names()[-1], "co_x_margin")
        feats = f.features(self.scores, self.mask, [99], gt_train={1: {10, 30}})
        # margin = (3-2)/(3-1) = 0.5; co_top_w(30→{10,20}) = log1p(1)
        self.assertAlmostEqual(feats[0, 2, 4], np.log1p(1.0) * 0.5, places=6)

    def test_cond_weight_is_conditional_on_anchor(self):
        f = _featurizer(co_weight="cond")
        self.assertEqual(f.feature_names()[0], "co_top_p")
        # Пара (30,10) в 1 из 2 запросов якоря 10 → P(30|10) = 0.5; частота ≠ вес.
        gt_train = {1: {10, 30}, 2: {10}}
        feats = f.features(self.scores, self.mask, [99], gt_train=gt_train)
        self.assertAlmostEqual(feats[0, 2, 0], 0.5, places=6)

    def test_cond_weight_loo_on_train_query(self):
        # Единственный источник и пары, и частоты якоря — сам запрос 1 → после LOO ноль.
        f = _featurizer(co_weight="cond")
        feats = f.features(self.scores, self.mask, [1], gt_train={1: {10, 30}})
        self.assertEqual(feats[0, 2, 0], 0.0)


class TestMLPRankerLTR(unittest.TestCase):
    def _toy(self, n_groups=40, cands=6, seed=0):
        rng = np.random.default_rng(seed)
        X, y, groups = [], [], []
        for g in range(n_groups):
            pos = rng.integers(cands)
            for c in range(cands):
                X.append([1.0 if c == pos else 0.0, rng.normal()])  # первый признак — идеальный
                y.append(1.0 if c == pos else 0.0)
                groups.append(g)
        return np.array(X, np.float32), np.array(y, np.float32), np.array(groups)

    def test_learns_informative_feature(self):
        X, y, groups = self._toy()
        model = MLPRankerLTR(hidden=[8], epochs=150, lr=0.05, seed=0).fit(X, y, groups)
        proba = model.predict_proba(X)
        self.assertGreater(proba[y == 1].mean(), proba[y == 0].mean() + 0.2)

    def test_deterministic(self):
        X, y, groups = self._toy()
        p1 = MLPRankerLTR(hidden=[8], epochs=50, seed=7).fit(X, y, groups).predict_proba(X)
        p2 = MLPRankerLTR(hidden=[8], epochs=50, seed=7).fit(X, y, groups).predict_proba(X)
        np.testing.assert_allclose(p1, p2)

    def test_group_without_positive_is_skipped(self):
        X, y, groups = self._toy(n_groups=10)
        y[groups == 0] = 0.0                            # запрос без позитива (GT вне кандидатов)
        MLPRankerLTR(hidden=[4], epochs=20, seed=0).fit(X, y, groups)  # не должен упасть/дать NaN


if __name__ == "__main__":
    unittest.main()
