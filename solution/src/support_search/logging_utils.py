"""Единая настройка логирования.

Наблюдаемость — часть контракта стадии (план §5.1.8): каждая стадия логирует
число входных/выходных строк, долю пустых текстов, время и метрики. Чтобы эти
сообщения выглядели одинаково и не дублировались при повторном импорте, весь код
берёт логгер только отсюда.
"""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    """Настроить корневой логгер один раз за процесс (идемпотентно)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger("support_search")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Логгер для модуля; гарантирует, что базовая настройка уже применена."""
    configure_logging()
    return logging.getLogger(f"support_search.{name}")
