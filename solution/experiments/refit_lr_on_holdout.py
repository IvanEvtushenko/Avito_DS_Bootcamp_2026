#!/usr/bin/env python3
"""Рефит финальной LR-головы на holdout (100 з.) с mean-of-5 реранкер-фичами.

Финальная голова stage_blend обучена на OOF-скорах (одна фолд-модель на
запрос), а применяется к среднему пяти — рефит на holdout убирает это
несоответствие: и fit, и predict видят mean-of-5. Цена: holdout расходуется
как обучающая выборка и перестаёт быть честной оценкой — правду о качестве
дальше говорит только test (метрика на holdout ниже — train, оптимистична).

Протокол — тот же, что у финального fit в stage_blend (§7.2):
- реранкер-фича holdout = среднее 5 фолд-моделей из
  ``experiments/generated/holdout_fold_scores_ep3.npz`` (ни одна не видела
  holdout) — то же распределение, что у ``scores/reranker/test.npz``;
- граф строится по всей размеченной выборке (dev + holdout, 500 з.);
  собственный вклад holdout-запроса при fit вычитается leave-one-out внутри
  featurizer — как для train-строк штатного протокола; у test-запросов меток
  в графе нет по построению;
- голова и фичи: LogisticRegressionLTR(l2), norm+rr, граф-фичи — из конфига.

Пишет только под ``experiments/generated/``: answer CSV + JSON с
коэффициентами. CPU-only.
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
from support_search.export import build_answer, validate_answer_file, write_answer  # noqa: E402
from support_search.ltr import LogisticRegressionLTR  # noqa: E402
from support_search.pipeline.stages import (  # noqa: E402
    _build_graph_featurizer,
    _fusion_sources,
    _load_matrices,
    load_articles,
    load_calibration,
    load_test,
)
from support_search.ranking.ltr import (  # noqa: E402
    _fit_head,
    _labels,
    _with_graph_features,
    build_blend_features,
    predict_blend_ltr,
)

CI_KW = dict(n_resamples=10_000, ci=0.95, seed=42)


def resolve_from_solution(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (SOLUTION / path).resolve()


def slice_rows(matrix: ScoreMatrix, query_ids: np.ndarray) -> ScoreMatrix:
    row_of = {int(q): i for i, q in enumerate(matrix.query_ids)}
    rows = [row_of[int(q)] for q in query_ids]
    return ScoreMatrix(query_ids, matrix.article_ids, matrix.scores[rows], matrix.source)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default="artifacts_rr_ft_k50_ep3")
    parser.add_argument("--fold-scores", default="experiments/generated/holdout_fold_scores_ep3.npz")
    parser.add_argument("--output", default="experiments/generated/answer_lr_holdout_refit.csv")
    args = parser.parse_args()

    artifact = resolve_from_solution(args.artifact)
    output = resolve_from_solution(args.output)
    cfg = load_config(
        SOLUTION / "configs/default.yaml",
        overrides=[f"artifacts_dir={artifact}"],
    )
    ltr_cfg = cfg.section("ranking").ltr
    l2 = float(ltr_cfg.l2)
    use_ranks = bool(ltr_cfg.use_ranks)

    calibration = load_calibration(cfg)
    gt = parse_ground_truth(calibration)
    splits = Splits.from_json(json.loads((artifact / "splits/folds.json").read_text()))

    with np.load(resolve_from_solution(args.fold_scores), allow_pickle=False) as data:
        if data["completed_folds"].tolist() != list(range(splits.n_splits)):
            raise RuntimeError(f"NPZ неполный: folds={data['completed_folds'].tolist()}")
        holdout_ids = data["query_ids"].astype(np.int64)
        article_ids = data["article_ids"].astype(np.int64)
        mean5 = data["scores"].astype(np.float32).mean(axis=0)
    if sorted(int(q) for q in holdout_ids) != sorted(splits.holdout):
        raise RuntimeError("query_ids NPZ не совпадают с holdout из splits")

    names = _fusion_sources(cfg)
    src_cal = _load_matrices(cfg, names, "calibration")
    src_test = _load_matrices(cfg, names, "test")
    reranker_test = ScoreMatrix.load(artifact / "scores/reranker/test.npz")
    for m in (*src_cal.values(), *src_test.values(), reranker_test):
        if not np.array_equal(m.article_ids, article_ids):
            raise RuntimeError(f"порядок article_ids расходится: {m.source}")

    src_hold = {name: slice_rows(m, holdout_ids) for name, m in src_cal.items()}
    reranker_hold = ScoreMatrix(holdout_ids, article_ids, mean5, "reranker_ft_mean5")
    featurizer = _build_graph_featurizer(cfg, article_ids)
    gt_all = {int(q): gt[int(q)] for q in (*splits.dev, *splits.holdout) if int(q) in gt}

    base, base_names, mask = build_blend_features(src_hold, reranker_hold, names=names, use_ranks=use_ranks)
    features, feature_names = _with_graph_features(base, base_names, reranker_hold, mask, featurizer, gt_all)
    labels = _labels(holdout_ids, article_ids, gt)
    model = _fit_head(LogisticRegressionLTR(l2=l2), features, labels, list(range(len(holdout_ids))), mask)
    coefficients = model.coefficients(feature_names)
    print(f"коэффициенты (fit на holdout, n={len(holdout_ids)}): {coefficients}")

    predict_kw = dict(names=names, use_ranks=use_ranks, featurizer=featurizer, gt_train=gt_all)
    blend_hold = predict_blend_ltr(model, src_hold, reranker_hold, **predict_kw)
    gt_holdout = {int(q): gt[int(q)] for q in splits.holdout}
    res = evaluate(blend_hold.rankings(50), gt_holdout, name="lr_holdout_refit/train",
                   k=10, recall_ks=[10, 20, 50]).compute_ci(**CI_KW)
    print(f"{res.summary()}  <- train-метрика (fit на этих же запросах), оптимистична")

    final_test = predict_blend_ltr(model, src_test, reranker_test, **predict_kw)
    test = load_test(cfg)
    articles = load_articles(cfg)
    top_k = int(cfg.section("export").top_k)
    answer = build_answer(final_test.rankings(top_k), test["query_id"].tolist(), top_k=top_k)
    write_answer(answer, output)
    validate_answer_file(output, test=test, articles=articles, max_k=top_k)
    print(f"answer: {output} ({len(answer)} строк, валиден)")

    report = {
        "artifact": artifact.name,
        "fit_on": "holdout",
        "n_fit_queries": int(len(holdout_ids)),
        "reranker_feature": "mean of 5 fold models (as on test)",
        "graph_gt": "dev + holdout (500), LOO at fit",
        "l2": l2,
        "use_ranks": use_ranks,
        "coefficients": coefficients,
        "holdout_train_map@10": round(res.map_at_k, 4),
        "answer": str(output),
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"отчёт: {report_path}")

    # Holdout израсходован как обучающая выборка — фиксируем в логе обращений
    # (формат holdout_check._log_access; идемпотентно при повторных запусках).
    log_path = artifact / "splits/holdout_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"budget": 3, "accesses": []}
    what = "FIT: LR-голова обучена на holdout (mean-of-5) — holdout исчерпан как оценка"
    if not any(entry.get("what") == what for entry in log["accesses"]):
        from datetime import datetime, timezone

        log["accesses"].append({
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "what": what,
            "results": {"holdout_train_map@10": round(res.map_at_k, 4)},
        })
        log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2))
        print(f"обращение-FIT записано в {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
