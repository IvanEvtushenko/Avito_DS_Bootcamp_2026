#!/usr/bin/env python3
"""GPU-скрининг вариантов запросов одной фолд-моделью (контракт — ноутбук
robustness_error_analysis §7, режим A).

Режим A (--fixed-candidates, приоритетный): у варианта меняется ТОЛЬКО текст
запроса, который видит cross-encoder. Кандидаты — исходные fusion-OOF
(`<artifact>/candidates/dev_oof.parquet`), пассажи зафиксированы выбором
ОРИГИНАЛЬНОГО запроса (кэш q_emb): изолируется чувствительность реранкера к
тексту, dense-энкодер не загружается вовсе. Скорится одна модель
`<artifact>/models/reranker_ft/fold_<N>` на val-запросах своего фолда — модель
их разметку не видела, сравнение честное.

Режим B (full query path: пересчёт BM25/char/dense/fusion для варианта) по
протоколу ноутбука нужен только вариантам, прошедшим режим A, и здесь
намеренно не реализован — запуск без --fixed-candidates завершается ошибкой.

Выход — parquet `query_id, variant, article_id, score` (50 строк на пару),
его автоматически подхватывает ячейка §7 ноутбука.

  cd solution && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src HF_HUB_OFFLINE=1 \\
  python3 experiments/score_query_variants.py \\
    --variants experiments/generated/query_variants_fold0.csv \\
    --artifact artifacts_rr_ft_k50_ep3 --fold 0 --fixed-candidates \\
    --output experiments/generated/variant_scores_fold0.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOL = Path(__file__).resolve().parent.parent
if str(SOL / "src") not in sys.path:
    sys.path.insert(0, str(SOL / "src"))

import numpy as np
import pandas as pd

from support_search.config import load_config
from support_search.data.io import read_json
from support_search.data.splits import Splits
from support_search.logging_utils import get_logger

logger = get_logger("experiments.score_variants")


def score_variants(
    variants: pd.DataFrame,
    *,
    artifact: Path,
    fold: int,
    limit: int | None = None,
    reranker=None,
) -> pd.DataFrame:
    """Оценить все (query, variant) моделью фолда на фиксированных кандидатах.

    `reranker=None` → загрузить CrossEncoderReranker фолда (GPU); тесты могут
    передать заглушку с интерфейсом `score_pairs(queries, passages)`.
    """
    from support_search.rerank import rerank_to_matrix
    from support_search.retrievers import DenseRetriever

    splits = Splits.from_json(read_json(artifact / "splits" / "folds.json"))
    val_ids = set(splits.fold(fold))
    qids = sorted(set(variants["query_id"].astype(int)) & val_ids)
    dropped = sorted(set(variants["query_id"].astype(int)) - val_ids)
    if dropped:
        logger.warning("вне val-фолда %d и пропущены: %s", fold, dropped[:10])
    if limit:
        qids = qids[:limit]
    if not qids:
        raise ValueError(f"в variants нет запросов val-фолда {fold}")
    if "original" not in set(variants["variant"]):
        raise ValueError("variants обязан содержать строки variant='original' (paired-сравнение)")

    cand = pd.read_parquet(artifact / "candidates" / "dev_oof.parquet")
    cand_by_qid = {
        int(q): g.sort_values("rank")["article_id"].astype(int).tolist()
        for q, g in cand.groupby("query_id") if int(q) in set(qids)
    }
    missing = [q for q in qids if q not in cand_by_qid]
    if missing:
        raise ValueError(f"нет кандидатов dev_oof для запросов: {missing[:10]}")
    cand_lists = [cand_by_qid[q] for q in qids]

    # Пассажи — по кэшированным эмбеддингам ОРИГИНАЛЬНЫХ запросов (фиксируем
    # выбор чанков: вариант меняет только текст на входе cross-encoder).
    cfg = load_config(SOL / "configs" / "default.yaml")
    cal = pd.read_feather(cfg.resolve(cfg.section("data").dir) / cfg.section("data").calibration)
    row_of = {int(q): i for i, q in enumerate(cal["query_id"])}
    q_emb = np.load(artifact / "scores" / "dense" / "q_emb_calibration.npy")[[row_of[q] for q in qids]]
    dense = DenseRetriever.load(artifact / "scores" / "dense" / "index", encoder=None)
    manifest = read_json(artifact / "scores" / "reranker" / "manifest.json")
    m_chunks = int(manifest.get("chunks_per_article", 1) or 1)
    passages = (dense.top_chunk_texts(q_emb, cand_lists, m_chunks) if m_chunks > 1
                else dense.best_chunk_texts(q_emb, cand_lists))

    if reranker is None:
        from support_search.rerank import CrossEncoderReranker
        rr = cfg.section("reranker")
        reranker = CrossEncoderReranker(
            artifact / "models" / "reranker_ft" / f"fold_{fold}",
            max_length=int(rr.max_length), batch_size=int(rr.batch_size), use_fp16=bool(rr.use_fp16),
        )

    text_of = {(int(r.query_id), str(r.variant)): str(r.query_text) for r in variants.itertuples()}
    col_of = {int(a): j for j, a in enumerate(dense.article_ids)}
    variant_names = list(dict.fromkeys(variants["variant"]))  # порядок появления, original первым
    rows: list[dict] = []
    for name in variant_names:
        texts = [text_of.get((q, name)) for q in qids]
        if any(t is None for t in texts):
            logger.warning("variant=%s: не для всех запросов есть текст — пропуск", name)
            continue
        matrix = rerank_to_matrix(reranker, texts, qids, dense.article_ids, cand_lists, passages, source=name)
        for i, qid in enumerate(qids):
            for aid in cand_lists[i]:
                rows.append({"query_id": qid, "variant": name, "article_id": aid,
                             "score": float(matrix.scores[i, col_of[aid]])})
        logger.info("variant=%s: %d запросов оценено", name, len(qids))
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", required=True, help="CSV: query_id, variant, query_text")
    parser.add_argument("--artifact", required=True, help="artifacts_dir с моделями/кандидатами/индексом")
    parser.add_argument("--fold", type=int, default=0, help="фолд: модель fold_N, запросы val(N)")
    parser.add_argument("--fixed-candidates", action="store_true", help="режим A (обязателен)")
    parser.add_argument("--output", required=True, help="путь выходного parquet")
    parser.add_argument("--limit", type=int, default=None, help="только первые N запросов (smoke)")
    args = parser.parse_args(argv)
    if not args.fixed_candidates:
        raise SystemExit("режим B (full query path) не реализован — запускайте с --fixed-candidates; "
                         "по протоколу ноутбука §7 он нужен только вариантам, прошедшим режим A")

    artifact = (SOL / args.artifact) if not Path(args.artifact).is_absolute() else Path(args.artifact)
    variants = pd.read_csv(SOL / args.variants if not Path(args.variants).is_absolute() else args.variants)
    result = score_variants(variants, artifact=artifact, fold=args.fold, limit=args.limit)
    out = SOL / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)
    logger.info("готово: %d строк (%d вариантов × %d запросов × кандидаты) → %s",
                len(result), result["variant"].nunique(), result["query_id"].nunique(), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
