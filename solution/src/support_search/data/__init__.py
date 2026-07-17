"""Ввод-вывод данных и разбиение на dev/holdout + фолды."""
from __future__ import annotations

from .io import (
    load_articles,
    load_calibration,
    load_test,
    parse_ground_truth,
    read_json,
    write_json,
)
from .splits import Splits, make_splits

__all__ = [
    "load_articles",
    "load_calibration",
    "load_test",
    "parse_ground_truth",
    "read_json",
    "write_json",
    "Splits",
    "make_splits",
]
