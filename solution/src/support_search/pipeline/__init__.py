"""Оркестрация: сборка ступеней в стадии + управление артефактами.

Единственное место, где ступени пайплайна собираются вместе (план §5.1.1).
CLI знает только про `pipeline` и `config`.
"""
from __future__ import annotations

from .artifacts import StageDir, hash_file, is_fresh, stage_dir, write_manifest
from .stages import (
    stage_evaluate,
    stage_fuse,
    stage_make_answer,
    stage_make_folds,
    stage_preprocess,
    stage_retrieve,
    stage_validate_answer,
)

__all__ = [
    "StageDir",
    "hash_file",
    "is_fresh",
    "stage_dir",
    "write_manifest",
    "stage_preprocess",
    "stage_make_folds",
    "stage_retrieve",
    "stage_evaluate",
    "stage_fuse",
    "stage_make_answer",
    "stage_validate_answer",
]
