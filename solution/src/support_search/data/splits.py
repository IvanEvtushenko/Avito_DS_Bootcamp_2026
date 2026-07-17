"""Разбиение calibration на dev/holdout и 5-fold — центр валидации (план §4.1).

На 500 примерах переобучиться легко, поэтому разбиение фиксируется ДО
экспериментов и переиспользуется всеми стадиями:

- **holdout** — 20% запросов, не участвуют ни в каком обучении/тюнинге;
- **dev** — остальные 80%, на них стратифицированный 5-fold CV.

Стратификация — по числу GT-статей (1 / 2 / 3+): распределение перекошено (56% —
одна статья), и без стратификации holdout мог бы случайно собрать нетипичные
запросы. Реализация без sklearn — чистый numpy, полностью детерминирована сидом.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


def stratum_of(gt_count: int) -> str:
    """Бакет стратификации по числу правильных статей: '1' / '2' / '3+'."""
    if gt_count <= 1:
        return "1"
    if gt_count == 2:
        return "2"
    return "3+"


@dataclass
class Splits:
    """Готовое разбiение: holdout, dev и назначение фолдов внутри dev."""

    seed: int
    n_splits: int
    holdout_frac: float
    holdout: list[int]
    dev: list[int]
    fold_of: dict[int, int]  # query_id -> индекс фолда (только dev)

    def fold(self, i: int) -> list[int]:
        """query_id валидационной части фолда i."""
        return sorted(q for q, f in self.fold_of.items() if f == i)

    def train_ids(self, i: int) -> list[int]:
        """query_id обучающей части фолда i (все dev, кроме фолда i)."""
        return sorted(q for q, f in self.fold_of.items() if f != i)

    def to_json(self) -> dict:
        return {
            "seed": self.seed,
            "n_splits": self.n_splits,
            "holdout_frac": self.holdout_frac,
            "holdout": sorted(self.holdout),
            "dev": sorted(self.dev),
            "folds": {str(i): self.fold(i) for i in range(self.n_splits)},
            "fold_of": {str(q): f for q, f in sorted(self.fold_of.items())},
        }

    @classmethod
    def from_json(cls, data: dict) -> "Splits":
        return cls(
            seed=int(data["seed"]),
            n_splits=int(data["n_splits"]),
            holdout_frac=float(data["holdout_frac"]),
            holdout=[int(q) for q in data["holdout"]],
            dev=[int(q) for q in data["dev"]],
            fold_of={int(q): int(f) for q, f in data["fold_of"].items()},
        )


def make_splits(
    gt_counts: Mapping[int, int],
    *,
    n_splits: int = 5,
    holdout_frac: float = 0.2,
    seed: int = 42,
) -> Splits:
    """Собрать стратифицированные holdout + 5-fold по числу GT-статей.

    Параметры
    ---------
    gt_counts    : query_id -> число правильных статей.
    n_splits     : число фолдов внутри dev.
    holdout_frac : доля запросов в holdout.
    seed         : фиксирует и перемешивание, и назначение фолдов.

    Внутри каждого страта запросы перемешиваются одним сидом, первые
    `holdout_frac` уходят в holdout, остальные раскладываются по фолдам
    round-robin — так и holdout, и каждый фолд сохраняют пропорции классов.
    """
    rng = np.random.default_rng(seed)

    strata: dict[str, list[int]] = {}
    for qid, cnt in gt_counts.items():
        strata.setdefault(stratum_of(cnt), []).append(int(qid))

    holdout: list[int] = []
    dev: list[int] = []
    fold_of: dict[int, int] = {}

    for name in sorted(strata):
        ids = np.array(sorted(strata[name]), dtype=np.int64)
        rng.shuffle(ids)
        n_holdout = int(round(holdout_frac * len(ids)))
        holdout.extend(ids[:n_holdout].tolist())
        dev_ids = ids[n_holdout:]
        dev.extend(dev_ids.tolist())
        # round-robin по фолдам: i-й dev-запрос страта → фолд i % n_splits.
        for i, qid in enumerate(dev_ids.tolist()):
            fold_of[qid] = i % n_splits

    return Splits(
        seed=seed,
        n_splits=n_splits,
        holdout_frac=holdout_frac,
        holdout=sorted(holdout),
        dev=sorted(dev),
        fold_of=fold_of,
    )
