"""Артефакты вместо скрытого состояния (план §5.1.6).

Каждая стадия пишет результат в свой каталог `artifacts/<stage>/` и рядом —
`manifest.json`: версия схемы, хэши входных файлов, хэш релевантной секции
конфига, seed, версии библиотек, время. По манифесту стадия понимает, свеж ли
кэш; несовместимый кэш не используется молча, а пересчитывается с логом.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import scipy

from .. import __version__
from ..config import Config
from ..data.io import read_json, write_json
from ..logging_utils import get_logger

logger = get_logger("pipeline.artifacts")

MANIFEST_NAME = "manifest.json"


def _library_versions() -> dict[str, str]:
    return {
        "support_search": __version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
    }


def hash_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """SHA-256 содержимого файла (усечён до 16 символов) — для инвалидации кэша."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()[:16]


def stage_dir(cfg: Config, name: str, *, subdir: str | None = None) -> Path:
    """Каталог артефактов стадии; создаётся при обращении."""
    path = cfg.artifacts_dir / name
    if subdir:
        path = path / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class StageDir:
    """Каталог стадии + удобная запись манифеста и проверка свежести."""

    cfg: Config
    name: str
    path: Path
    schema_version: int
    config_sections: tuple[str, ...]
    input_hashes: dict[str, str]

    def build_manifest(self, **extra: object) -> dict:
        return {
            "stage": self.name,
            "schema_version": self.schema_version,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": self.cfg.seed,
            "config_hash": self.cfg.hash_section(*self.config_sections),
            "inputs": dict(self.input_hashes),
            "versions": _library_versions(),
            **extra,
        }

    def write_manifest(self, **extra: object) -> Path:
        return write_manifest(self.path, self.build_manifest(**extra))

    def is_fresh(self) -> bool:
        return is_fresh(
            self.path,
            schema_version=self.schema_version,
            config_hash=self.cfg.hash_section(*self.config_sections),
            input_hashes=self.input_hashes,
        )


def write_manifest(directory: str | Path, manifest: Mapping[str, object]) -> Path:
    return write_json(Path(directory) / MANIFEST_NAME, dict(manifest))


def read_manifest(directory: str | Path) -> dict | None:
    path = Path(directory) / MANIFEST_NAME
    return read_json(path) if path.exists() else None


def is_fresh(
    directory: str | Path,
    *,
    schema_version: int,
    config_hash: str,
    input_hashes: Mapping[str, str],
) -> bool:
    """Свеж ли кэш: манифест есть и совпал по схеме, конфигу и входам."""
    manifest = read_manifest(directory)
    if manifest is None:
        return False
    if manifest.get("schema_version") != schema_version:
        logger.info("кэш %s: другая версия схемы — пересчёт", directory)
        return False
    if manifest.get("config_hash") != config_hash:
        logger.info("кэш %s: изменилась секция конфига — пересчёт", directory)
        return False
    if manifest.get("inputs") != dict(input_hashes):
        logger.info("кэш %s: изменились входные файлы — пересчёт", directory)
        return False
    return True
