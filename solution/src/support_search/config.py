"""Загрузка, валидация и выдача секций конфига.

Конфиг — единый источник правды об эксперименте (план §5.1, codestyle: «вся
конфигурация видна целиком»). Здесь он загружается из YAML, поверх него
накладываются ablation-переопределения (`configs/experiments/*.yaml`) и точечные
CLI-оверрайды (`section.key=value`). Компонентам отдаётся только их секция, а не
весь конфиг.

Пути в конфиге относительны корню проекта — каталогу, содержащему `configs/`
(то есть `solution/`). Это делает запуск независимым от текущего рабочего
каталога.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from omegaconf import DictConfig, OmegaConf

# Секции, без которых пайплайн не соберётся. Проверяются при загрузке, чтобы
# ошибка конфига падала сразу, а не через десять стадий.
REQUIRED_SECTIONS = (
    "data", "preprocess", "folds", "retrievers", "fusion", "reranker", "ranking", "evaluation", "export",
)


@dataclass(frozen=True)
class Config:
    """Обёртка над `DictConfig` с корнем проекта и удобными аксессорами."""

    raw: DictConfig
    project_root: Path
    config_path: Path

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))

    def section(self, name: str) -> DictConfig:
        """Вернуть одну секцию конфига (например, `retrievers`)."""
        if name not in self.raw:
            raise KeyError(f"в конфиге нет секции {name!r}")
        return self.raw[name]

    def resolve(self, path: str | Path) -> Path:
        """Абсолютный путь: относительные трактуются от корня проекта."""
        p = Path(path)
        return p if p.is_absolute() else (self.project_root / p).resolve()

    @property
    def artifacts_dir(self) -> Path:
        return self.resolve(self.raw.get("artifacts_dir", "artifacts"))

    def hash_section(self, *names: str) -> str:
        """Устойчивый хэш одной или нескольких секций — для manifest.json.

        Хэш зависит только от переданных секций, поэтому изменение чужого
        параметра не инвалидирует кэш стадии (план §5.1.6).
        """
        payload = {n: OmegaConf.to_container(self.section(n), resolve=True) for n in names}
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return OmegaConf.to_container(self.raw, resolve=True)  # type: ignore[return-value]


def load_config(
    config_path: str | Path = "configs/default.yaml",
    *,
    experiment: str | Path | None = None,
    overrides: Sequence[str] | None = None,
) -> Config:
    """Собрать конфиг: default → experiment-override → CLI-override.

    Параметры
    ---------
    config_path : базовый YAML (обычно `configs/default.yaml`).
    experiment  : необязательный YAML с переопределениями (ablation-конфиг).
    overrides   : список строк `section.key=value` (dotlist OmegaConf).

    Возвращает `Config` с корнем проекта = родитель каталога `configs/`.
    """
    base_path = Path(config_path).resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"конфиг не найден: {base_path}")

    cfg = OmegaConf.load(base_path)

    if experiment is not None:
        exp_path = Path(experiment)
        if not exp_path.is_absolute():
            exp_path = base_path.parent / "experiments" / exp_path
        if not exp_path.exists():
            raise FileNotFoundError(f"experiment-конфиг не найден: {exp_path}")
        cfg = OmegaConf.merge(cfg, OmegaConf.load(exp_path))

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    assert isinstance(cfg, DictConfig)
    _validate(cfg)

    # Корень проекта — родитель каталога configs/ (то есть solution/).
    project_root = base_path.parent.parent
    return Config(raw=cfg, project_root=project_root, config_path=base_path)


def _validate(cfg: DictConfig) -> None:
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f"в конфиге отсутствуют обязательные секции: {missing}")
