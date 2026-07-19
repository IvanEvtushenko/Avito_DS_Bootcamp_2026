"""Логистическая регрессия для learning-to-rank (mini-LTR, план §7.2).

Сквозная утилита (как `config`/`logging_utils`): не зависит от ступеней пайплайна,
поэтому её могут использовать и `fusion`, и `ranking`. `scikit-learn` в окружении
нет, поэтому LR реализована на numpy + scipy (L-BFGS) — тот же принцип, что и с
BM25/TF-IDF: минимальный понятный блок вместо тяжёлой зависимости.

Pointwise-LTR: для пары (запрос, статья) с признаками x предсказываем
P(релевантна) = σ(wᵀx + b). Ранжируем по этой вероятности. Классы сильно
несбалансированы (1–3 релевантных на сотни кандидатов), поэтому по умолчанию
включено балансирующее взвешивание классов. Признаки стандартизуются по
статистикам train (без утечки), что даёт устойчивую оптимизацию и сравнимые
коэффициенты (интерпретируемость — в отчёт, план §7.2).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def reciprocal_ranks(scores: np.ndarray) -> np.ndarray:
    """1/(ранг+1) для каждой статьи по убыванию скора; форма как у `scores` [Q, A].

    Ранговый признак: даёт LR информацию о позиции, а не только о значении скора
    (план §7.2 — «ранги по каждому источнику»). Ничьи разрешаются порядком
    столбцов (детерминированно).
    """
    q, a = scores.shape
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty((q, a), dtype=np.int64)
    ranks[np.arange(q)[:, None], order] = np.arange(a)[None, :]
    return (1.0 / (ranks + 1.0)).astype(np.float32)


class LogisticRegressionLTR:
    """L2-регуляризованная логистическая регрессия (numpy/scipy L-BFGS).

    Параметры
    ---------
    l2            : сила L2-регуляризации весов (bias не штрафуется).
    class_weight  : "balanced" — веса классов ∝ 1/частоте (борьба с дисбалансом);
                    None — без взвешивания.
    standardize   : стандартизовать признаки по train-статистикам.
    max_iter      : лимит итераций L-BFGS.
    """

    def __init__(
        self,
        *,
        l2: float = 1.0,
        class_weight: str | None = "balanced",
        standardize: bool = True,
        max_iter: int = 300,
    ) -> None:
        self.l2 = float(l2)
        self.class_weight = class_weight
        self.standardize = standardize
        self.max_iter = max_iter
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.w_: np.ndarray | None = None
        self.b_: float = 0.0

    def _prepare(self, X: np.ndarray, *, fit: bool) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if not self.standardize:
            return X
        if fit:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0)
            self.std_[self.std_ == 0] = 1.0
        return (X - self.mean_) / self.std_

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> "LogisticRegressionLTR":
        # `groups` (индекс запроса строки) не используется: LR — pointwise. Параметр
        # нужен для единого интерфейса со listwise-головами (ranking/mlp.py).
        del groups
        Xs = self._prepare(X, fit=True)
        y = np.asarray(y, dtype=np.float64)
        n, d = Xs.shape

        if self.class_weight == "balanced":
            n_pos = float(y.sum())
            n_neg = float(n - n_pos)
            w_pos = n / (2.0 * n_pos) if n_pos > 0 else 1.0
            w_neg = n / (2.0 * n_neg) if n_neg > 0 else 1.0
            sample_w = np.where(y == 1.0, w_pos, w_neg)
        else:
            sample_w = np.ones(n)

        def objective(theta: np.ndarray):
            w = theta[:d]
            b = theta[d]
            z = Xs @ w + b
            # Численно устойчивый weighted BCE-with-logits.
            loss = sample_w * (np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z))))
            f = float(loss.sum()) + 0.5 * self.l2 * float(w @ w)
            p = 1.0 / (1.0 + np.exp(-z))
            residual = sample_w * (p - y)
            grad_w = Xs.T @ residual + self.l2 * w
            grad_b = float(residual.sum())
            return f, np.concatenate([grad_w, [grad_b]])

        result = minimize(
            objective, np.zeros(d + 1), jac=True, method="L-BFGS-B",
            options={"maxiter": self.max_iter},
        )
        self.w_ = result.x[:d]
        self.b_ = float(result.x[d])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w_ is None:
            raise RuntimeError("LogisticRegressionLTR.predict_proba вызван до fit()")
        Xs = self._prepare(X, fit=False)
        z = Xs @ self.w_ + self.b_
        return 1.0 / (1.0 + np.exp(-z))

    def coefficients(self, feature_names: list[str] | None = None) -> dict[str, float]:
        """Коэффициенты (в стандартизованном пространстве) + bias — для отчёта."""
        if self.w_ is None:
            raise RuntimeError("нет обученной модели")
        names = feature_names or [f"f{i}" for i in range(len(self.w_))]
        out = {name: round(float(w), 4) for name, w in zip(names, self.w_)}
        out["bias"] = round(self.b_, 4)
        return out
