#!/usr/bin/env python3
"""Сравнение реранкеров: собрать полные таблицы метрик всех прогонов и ранжировать по целевой.

Каждый прогон `rerank_*.yaml` уже пишет ПОЛНУЮ таблицу метрик (все retriever×split:
MAP@k, recall@ks, CI) в свой `<artifacts_dir>/runs/experiments.csv`. Этот скрипт
ничего не пересчитывает — он лишь:

  1) читает experiments.csv каждого эксперимента, помечает строки моделью/backend'ом;
  2) склеивает их в одну общую таблицу `rerank_compare/comparison_full.csv`
     (полные метрики каждого результата сохранены);
  3) печатает и сохраняет сравнение по ЦЕЛЕВОЙ метрике (по умолчанию
     reranker_zs / dev_oof / MAP@10) — по ней и выбираем лучший.

Запуск (из solution/):
    python3 compare_rerankers.py                     # все configs/experiments/rerank_*.yaml
    python3 compare_rerankers.py --experiments rerank_bge_v2_m3.yaml rerank_jina_v3.yaml
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

# Позволяет запускать как `python3 compare_rerankers.py` без PYTHONPATH=src (как в тестах).
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from support_search.config import load_config  # noqa: E402


def _experiment_files(names: list[str], configs_dir: Path) -> list[Path]:
    if names:
        return [(configs_dir / "experiments" / n) if not Path(n).is_absolute() else Path(n) for n in names]
    return sorted(Path(p) for p in glob.glob(str(configs_dir / "experiments" / "rerank_*.yaml")))


def _label(cfg) -> dict[str, str]:
    rr = cfg.section("reranker")
    return {"backend": str(rr.get("backend", "sequence_classification")), "model": str(rr.get("model_name", ""))}


def collect(experiments: list[Path], base_config: str) -> tuple[pd.DataFrame, list[str]]:
    """Склеить experiments.csv всех прогонов в одну таблицу; вернуть (df, пропущенные)."""
    frames, missing = [], []
    for exp in experiments:
        cfg = load_config(base_config, experiment=str(exp))
        csv = cfg.artifacts_dir / "runs" / "experiments.csv"
        if not csv.exists():
            missing.append(f"{exp.name} → нет {csv} (эксперимент ещё не прогнан)")
            continue
        df = pd.read_csv(csv)
        df.insert(0, "experiment", exp.stem)
        for key, val in _label(cfg).items():
            df.insert(1, key, val)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, missing


def rank(combined: pd.DataFrame, retriever: str, split: str, metric: str) -> pd.DataFrame:
    """Одна строка на эксперимент по целевой (retriever/split), отсортировано по metric ↓."""
    tgt = combined[(combined["retriever"] == retriever) & (combined["split"] == split)].copy()
    if tgt.empty:
        return tgt
    # при повторных прогонах одного конфига — берём самый свежий
    tgt = tgt.sort_values("created_at").drop_duplicates("experiment", keep="last")
    cols = ["experiment", "backend", "model", metric, "ci_low", "ci_high", "n_queries"]
    cols = [c for c in cols if c in tgt.columns]
    return tgt.sort_values(metric, ascending=False)[cols].reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml", help="базовый YAML")
    ap.add_argument("--experiments", nargs="*", default=[], metavar="rerank_X.yaml",
                    help="конфиги экспериментов (по умолчанию все rerank_*.yaml)")
    ap.add_argument("--out", default="rerank_compare", help="каталог для сводных CSV")
    ap.add_argument("--target-retriever", default="reranker_zs", help="целевой источник (строка метрик)")
    ap.add_argument("--target-split", default="dev_oof", help="целевой split")
    ap.add_argument("--metric", default="map@10", help="целевая метрика для ранжирования")
    args = ap.parse_args(argv)

    configs_dir = Path(args.config).resolve().parent
    experiments = _experiment_files(args.experiments, configs_dir)
    combined, missing = collect(experiments, args.config)

    for m in missing:
        print("[skip]", m)
    if combined.empty:
        print("нет прогнанных экспериментов — сначала запустите `cli all --experiment rerank_*.yaml`")
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    full_path = out / "comparison_full.csv"
    combined.to_csv(full_path, index=False)

    ranked = rank(combined, args.target_retriever, args.target_split, args.metric)
    if ranked.empty:
        print(f"в собранных таблицах нет строки {args.target_retriever}/{args.target_split}")
        return 1
    rank_path = out / "comparison_target.csv"
    ranked.to_csv(rank_path, index=False)

    print(f"\nПолная таблица метрик всех прогонов: {full_path}  ({len(combined)} строк)")
    print(f"Сравнение по целевой ({args.target_retriever}/{args.target_split}/{args.metric}) → {rank_path}\n")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(ranked.to_string(index=False))
    best = ranked.iloc[0]
    print(f"\nЛучший по {args.metric}: {best['experiment']}  ({best.get('model','')})  "
          f"{args.metric}={best[args.metric]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
