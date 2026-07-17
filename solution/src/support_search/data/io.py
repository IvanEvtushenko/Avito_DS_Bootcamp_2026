"""Чтение исходных данных (feather) и артефактов (json).

Тонкий слой: без бизнес-логики, только загрузка/сохранение с фиксированными
контрактами колонок. `parse_ground_truth` — единственное преобразование:
строка "1909 4396" → множество int, потому что метрика работает с множествами.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Config

ARTICLE_COLUMNS = ["article_id", "title", "body"]
CALIBRATION_COLUMNS = ["query_id", "query_text", "ground_truth"]
TEST_COLUMNS = ["query_id", "query_text"]


def _read_feather(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"файл данных не найден: {path}")
    with warnings.catch_warnings():
        # pandas 3.x предупреждает о внутреннем pyarrow.feather — нам всё равно.
        warnings.simplefilter("ignore", FutureWarning)
        return pd.read_feather(path)


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> pd.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: нет колонок {missing}, есть {list(df.columns)}")
    return df


def load_articles(cfg: Config) -> pd.DataFrame:
    """Корпус статей: article_id (int), title (str), body (HTML str)."""
    data = cfg.section("data")
    df = _read_feather(cfg.resolve(data.dir) / data.articles)
    df = _require_columns(df, ARTICLE_COLUMNS, "articles")
    df = df.copy()
    df["article_id"] = df["article_id"].astype("int64")
    df["title"] = df["title"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    return df.reset_index(drop=True)


def load_calibration(cfg: Config) -> pd.DataFrame:
    """Размеченные запросы: query_id, query_text, ground_truth (строка id)."""
    data = cfg.section("data")
    df = _read_feather(cfg.resolve(data.dir) / data.calibration)
    df = _require_columns(df, CALIBRATION_COLUMNS, "calibration")
    df = df.copy()
    df["query_id"] = df["query_id"].astype("int64")
    df["query_text"] = df["query_text"].fillna("").astype(str)
    df["ground_truth"] = df["ground_truth"].fillna("").astype(str)
    return df.reset_index(drop=True)


def load_test(cfg: Config) -> pd.DataFrame:
    """Тестовые запросы без разметки: query_id, query_text."""
    data = cfg.section("data")
    df = _read_feather(cfg.resolve(data.dir) / data.test)
    df = _require_columns(df, TEST_COLUMNS, "test")
    df = df.copy()
    df["query_id"] = df["query_id"].astype("int64")
    df["query_text"] = df["query_text"].fillna("").astype(str)
    return df.reset_index(drop=True)


def parse_ground_truth(calibration: pd.DataFrame) -> dict[int, set[int]]:
    """{query_id: {article_id, ...}} из строкового поля ground_truth."""
    out: dict[int, set[int]] = {}
    for qid, gt in zip(calibration["query_id"], calibration["ground_truth"]):
        ids = {int(x) for x in str(gt).split()} if str(gt).strip() else set()
        out[int(qid)] = ids
    return out


def write_json(path: str | Path, payload: Any) -> Path:
    """Записать JSON (utf-8, с отступами, стабильный порядок ключей)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)
