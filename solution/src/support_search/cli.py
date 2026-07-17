"""CLI: подкоманда = стадия пайплайна (план §5.1.4).

    support-search <command> [--config C] [--experiment E] [--set k=v ...] [--force]

Команды: preprocess, make-folds, retrieve, evaluate, make-answer,
validate-answer и all (полный прогон по порядку). Полный прогон только
оркеструет те же стадии — отдельной логики в нём нет.
"""
from __future__ import annotations

import argparse
from typing import Sequence

from .config import load_config
from .logging_utils import get_logger
from .pipeline import (
    stage_evaluate,
    stage_fuse,
    stage_make_answer,
    stage_make_folds,
    stage_preprocess,
    stage_retrieve,
    stage_validate_answer,
)

logger = get_logger("cli")

# Стадии, поддерживающие кэш (принимают force=...).
_CACHEABLE = {"preprocess", "make-folds", "retrieve"}


def _common_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/default.yaml", help="базовый YAML-конфиг")
    common.add_argument("--experiment", default=None, help="ablation-конфиг из configs/experiments/")
    common.add_argument("--set", nargs="*", dest="overrides", default=[],
                        metavar="section.key=value", help="точечные оверрайды конфига")
    common.add_argument("--force", action="store_true", help="пересчитать, игнорируя кэш")
    return common


def _run_stage(command: str, cfg, force: bool) -> None:
    if command == "preprocess":
        stage_preprocess(cfg, force=force)
    elif command == "make-folds":
        stage_make_folds(cfg, force=force)
    elif command == "retrieve":
        stage_retrieve(cfg, force=force)
    elif command == "evaluate":
        stage_evaluate(cfg)
    elif command == "fuse":
        stage_fuse(cfg)
    elif command == "make-answer":
        stage_make_answer(cfg)
    elif command == "validate-answer":
        stage_validate_answer(cfg)
    elif command == "all":
        stage_preprocess(cfg, force=force)
        stage_make_folds(cfg, force=force)
        stage_retrieve(cfg, force=force)
        stage_evaluate(cfg)
        stage_fuse(cfg)
        stage_make_answer(cfg)
        stage_validate_answer(cfg)
    else:  # pragma: no cover - argparse не пропустит
        raise ValueError(f"неизвестная команда: {command}")


def main(argv: Sequence[str] | None = None) -> int:
    common = _common_parser()
    parser = argparse.ArgumentParser(prog="support-search", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for cmd in ["preprocess", "make-folds", "retrieve", "evaluate", "fuse", "make-answer", "validate-answer", "all"]:
        sub.add_parser(cmd, parents=[common], help=f"стадия: {cmd}")

    args = parser.parse_args(argv)
    cfg = load_config(args.config, experiment=args.experiment, overrides=args.overrides)
    logger.info("команда=%s config=%s experiment=%s", args.command, args.config, args.experiment)
    _run_stage(args.command, cfg, force=bool(args.force))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
