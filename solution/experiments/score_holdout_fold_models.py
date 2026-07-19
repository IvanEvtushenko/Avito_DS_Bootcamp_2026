#!/usr/bin/env python3
"""Score the same holdout queries with five saved OOF fold models.

This is the GPU producer for section 11 of
``article_query_graph_diagnostics.ipynb``.  It deliberately does not average
fold logits: the output keeps a matrix ``scores[fold, query, article]`` so all
ensemble sizes and aggregation rules can be studied later on CPU.

The script writes only under ``solution/experiments/``.  A partial NPZ is
updated after every completed fold, so inference can be resumed without
rescoring finished models.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOLUTION = Path(__file__).resolve().parents[1]
SRC = SOLUTION / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from support_search.config import load_config  # noqa: E402
from support_search.contracts import ScoreMatrix  # noqa: E402
from support_search.data.io import parse_ground_truth  # noqa: E402
from support_search.data.splits import Splits  # noqa: E402
from support_search.eval.harness import evaluate  # noqa: E402
from support_search.pipeline.stages import (  # noqa: E402
    _reorder_rows,
    _texts_in_order,
    load_calibration,
)
from support_search.rerank import (  # noqa: E402
    CrossEncoderReranker,
    candidate_lists,
    rerank_to_matrix,
)
from support_search.rerank.apply import NON_CANDIDATE  # noqa: E402
from support_search.retrievers import DenseRetriever  # noqa: E402


def resolve_from_solution(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (SOLUTION / path).resolve()


def metric_summary(scores: np.ndarray, query_ids: np.ndarray, article_ids: np.ndarray, gt: dict, name: str) -> str:
    matrix = ScoreMatrix(query_ids, article_ids, scores, name)
    ground_truth = {int(qid): gt[int(qid)] for qid in query_ids}
    result = evaluate(
        matrix.rankings(50), ground_truth, name=name, k=10, recall_ks=[10, 20, 50]
    ).compute_ci(n_resamples=10_000, ci=0.95, seed=42)
    return result.summary()


def save_progress(
    path: Path,
    query_ids: np.ndarray,
    article_ids: np.ndarray,
    scores: np.ndarray,
    completed_folds: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        query_ids=query_ids,
        article_ids=article_ids,
        scores=scores,
        completed_folds=np.asarray(completed_folds, dtype=np.int64),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default="artifacts_rr_ft_k50_ep3")
    parser.add_argument(
        "--output",
        default="experiments/generated/holdout_fold_scores_ep3.npz",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    artifact = resolve_from_solution(args.artifact)
    output = resolve_from_solution(args.output)
    partial = output.with_name(output.stem + ".partial.npz")
    cfg = load_config(
        SOLUTION / "configs/default.yaml",
        overrides=[f"artifacts_dir={artifact}"],
    )

    manifest = json.loads((artifact / "scores/reranker/manifest.json").read_text())
    if manifest.get("chosen") != "fine_tune":
        raise RuntimeError(f"ожидали fine_tune, manifest={manifest}")
    top_k = int(manifest["candidate_top_k"])
    chunks_per_article = int(manifest["chunks_per_article"])

    calibration = load_calibration(cfg)
    gt = parse_ground_truth(calibration)
    splits = Splits.from_json(json.loads((artifact / "splits/folds.json").read_text()))
    holdout_ids = np.asarray([int(qid) for qid in splits.holdout], dtype=np.int64)
    if len(holdout_ids) != 100:
        raise RuntimeError(f"ожидали общий holdout из 100 запросов, получили {len(holdout_ids)}")

    fusion = ScoreMatrix.load(artifact / "scores/fusion/calibration.npz")
    article_ids = fusion.article_ids.copy()
    row_of = {int(qid): row for row, qid in enumerate(fusion.query_ids)}
    holdout_rows = [row_of[int(qid)] for qid in holdout_ids]

    dense = DenseRetriever.load(artifact / "scores/dense/index", encoder=None)
    query_embeddings = np.load(artifact / "scores/dense/q_emb_calibration.npy")
    query_embeddings = _reorder_rows(
        query_embeddings,
        calibration["query_id"].to_numpy(),
        fusion.query_ids,
    )
    query_texts = _texts_in_order(calibration, fusion.query_ids)
    all_candidates = candidate_lists(fusion, top_k)
    holdout_candidates = [all_candidates[row] for row in holdout_rows]
    holdout_passages = dense.top_chunk_texts(
        query_embeddings[holdout_rows], holdout_candidates, chunks_per_article
    )
    holdout_texts = [query_texts[row] for row in holdout_rows]

    model_root = artifact / "models/reranker_ft"
    model_dirs = [model_root / f"fold_{fold}" for fold in range(splits.n_splits)]
    missing = [path for path in model_dirs if not (path / "model.safetensors").exists()]
    if missing:
        raise FileNotFoundError(f"не найдены полные fold-модели: {missing}")

    print(
        f"artifact={artifact.name}; holdout={len(holdout_ids)}; top_k={top_k}; "
        f"chunks={chunks_per_article}; pairs/model≈{len(holdout_ids) * top_k * chunks_per_article}"
    )
    print(f"output={output}")
    if args.prepare_only:
        print("prepare-only: входы и пять моделей валидны; CUDA не запускалась")
        return 0

    shape = (splits.n_splits, len(holdout_ids), len(article_ids))
    fold_scores = np.full(shape, NON_CANDIDATE, dtype=np.float32)
    completed: list[int] = []
    if partial.exists():
        with np.load(partial, allow_pickle=False) as data:
            if not np.array_equal(data["query_ids"], holdout_ids):
                raise RuntimeError("partial: изменился порядок query_ids")
            if not np.array_equal(data["article_ids"], article_ids):
                raise RuntimeError("partial: изменился порядок article_ids")
            if data["scores"].shape != shape:
                raise RuntimeError(f"partial: неожиданная форма {data['scores'].shape}, ожидали {shape}")
            fold_scores = data["scores"].astype(np.float32)
            completed = data["completed_folds"].astype(int).tolist()
        print(f"resume: уже готовы folds={completed}")

    rr_cfg = cfg.section("reranker")
    for fold, model_dir in enumerate(model_dirs):
        if fold in completed:
            print(f"fold {fold}: skip (есть в partial)")
            continue
        model = CrossEncoderReranker(
            model_dir,
            max_length=int(rr_cfg.max_length),
            batch_size=int(rr_cfg.batch_size),
            use_fp16=bool(rr_cfg.use_fp16),
        )
        matrix = rerank_to_matrix(
            model,
            holdout_texts,
            holdout_ids,
            article_ids,
            holdout_candidates,
            holdout_passages,
            source=f"reranker_ft_fold_{fold}",
        )
        fold_scores[fold] = matrix.scores
        completed.append(fold)
        completed.sort()
        save_progress(partial, holdout_ids, article_ids, fold_scores, completed)
        print(metric_summary(fold_scores[fold], holdout_ids, article_ids, gt, f"fold_{fold}/holdout"))
        print(f"fold {fold}: сохранён; progress={len(completed)}/{splits.n_splits}")
        del model
        import torch

        torch.cuda.empty_cache()

    if completed != list(range(splits.n_splits)):
        raise RuntimeError(f"не все модели посчитаны: {completed}")
    save_progress(output, holdout_ids, article_ids, fold_scores, completed)
    print(metric_summary(fold_scores.mean(axis=0), holdout_ids, article_ids, gt, "mean5/holdout"))
    print(f"готово: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
