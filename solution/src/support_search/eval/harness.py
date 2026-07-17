"""Eval-harness: единая оценка любого ранжировщика.

`evaluate()` принимает готовые ранжирования (dict query_id -> список article_id)
и ground truth и возвращает `EvalResult` с MAP@k, per-query AP (для анализа
ошибок) и recall@k (потолок для реранкера). Один и тот же harness применяется к
BM25, char-TF-IDF, fusion и реранкеру — так сравнение честное (codestyle:
«метрики считаются одинаково для всех сравниваемых моделей»).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .metrics import per_query_ap, recall_at_k
from .stats import bootstrap_ci


@dataclass
class EvalResult:
    """Результат оценки одной конфигурации на одном наборе запросов."""

    name: str
    k: int
    map_at_k: float
    per_query_ap: dict[int, float]
    recall_at_k: dict[int, float]
    n_queries: int
    ci: tuple[float, float] | None = field(default=None)

    @property
    def ap_array(self) -> np.ndarray:
        """AP@k по запросам в порядке возрастания query_id (для stats-тестов)."""
        return np.asarray([self.per_query_ap[q] for q in sorted(self.per_query_ap)], dtype=np.float64)

    def compute_ci(self, *, n_resamples: int = 10_000, ci: float = 0.95, seed: int = 42) -> "EvalResult":
        self.ci = bootstrap_ci(self.ap_array, n_resamples=n_resamples, ci=ci, seed=seed)
        return self

    def to_row(self) -> dict[str, object]:
        """Строка для таблицы экспериментов (план §4.4)."""
        row: dict[str, object] = {
            "name": self.name,
            "n_queries": self.n_queries,
            f"map@{self.k}": round(self.map_at_k, 4),
        }
        if self.ci is not None:
            row["ci_low"] = round(self.ci[0], 4)
            row["ci_high"] = round(self.ci[1], 4)
        for rk, val in sorted(self.recall_at_k.items()):
            row[f"recall@{rk}"] = round(val, 4)
        return row

    def summary(self) -> str:
        ci = f" CI[{self.ci[0]:.4f}, {self.ci[1]:.4f}]" if self.ci else ""
        rec = " ".join(f"R@{rk}={v:.3f}" for rk, v in sorted(self.recall_at_k.items()))
        return f"{self.name}: MAP@{self.k}={self.map_at_k:.4f}{ci} | {rec} | n={self.n_queries}"


def evaluate(
    rankings: Mapping[int, Sequence[int]],
    ground_truth: Mapping[int, set[int]],
    *,
    name: str = "model",
    k: int = 10,
    recall_ks: Sequence[int] = (10, 20, 50, 100),
) -> EvalResult:
    """Оценить ранжирования на запросах из `ground_truth`.

    Параметры
    ---------
    rankings     : query_id -> ранжированный список article_id (длиннее k для recall@k).
    ground_truth : query_id -> множество правильных article_id.
    name         : имя конфигурации (для отчёта).
    k            : отсечка MAP@k.
    recall_ks    : на каких k считать средний recall.
    """
    aps = per_query_ap(rankings, ground_truth, k)
    map_score = float(np.mean(list(aps.values()))) if aps else 0.0

    recall: dict[int, float] = {}
    for rk in recall_ks:
        vals = [recall_at_k(rankings.get(q, []), rel, rk) for q, rel in ground_truth.items()]
        recall[int(rk)] = float(np.mean(vals)) if vals else 0.0

    return EvalResult(
        name=name,
        k=k,
        map_at_k=map_score,
        per_query_ap=aps,
        recall_at_k=recall,
        n_queries=len(ground_truth),
    )
