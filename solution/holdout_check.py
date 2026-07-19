#!/usr/bin/env python3
"""Holdout-проверка финальных моделей (бюджет обращений ≤3, план §4.1).

Каждый запуск — ОДНО учтённое обращение к holdout (логируется в
<artifacts>/splits/holdout_log.json). Протокол — ровно тот, что в архивных
прогонах 2026-07-18: fusion top-50, chunks_per_article=3 (из манифеста), никакого
подбора по holdout — только финальная оценка уже зафиксированных моделей.

  python3 holdout_check.py zs   # исходная модель: zero-shot bge + её blend
                                # (artifacts_rr_bge_ft_synth, chosen=zero_shot; матрицы
                                #  уже покрывают holdout — GPU не нужен)
  python3 holdout_check.py ft   # дообученная: усреднение 5 фолд-моделей (как на test,
                                #  ни одна не видела holdout) + blend, перефитченный на dev
                                #  (коэффициенты сверяются с blend_report) — нужен GPU
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from support_search.config import load_config  # noqa: E402
from support_search.contracts import ScoreMatrix  # noqa: E402
from support_search.data.io import parse_ground_truth  # noqa: E402
from support_search.data.splits import Splits  # noqa: E402
from support_search.eval.harness import evaluate  # noqa: E402

DEPTH = 100
CI_KW = dict(n_resamples=10_000, ci=0.95, seed=42)


def _holdout_eval(matrix: ScoreMatrix, gt_holdout: dict, name: str):
    rankings = matrix.rankings(DEPTH)
    res = evaluate(rankings, gt_holdout, name=name, k=10, recall_ks=[10, 20, 50, 100]).compute_ci(**CI_KW)
    print(res.summary())
    return res


def _log_access(artifacts: Path, what: str, results: dict) -> None:
    path = artifacts / "splits" / "holdout_log.json"
    log = json.loads(path.read_text()) if path.exists() else {"budget": 3, "accesses": []}
    log["accesses"].append({
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "what": what,
        "results": results,
    })
    path.write_text(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"обращение записано в {path} (использовано {len(log['accesses'])} из {log['budget']})")


def check_zs() -> None:
    """Исходная модель: zero-shot bge и её blend — по матрицам СВЕЖЕГО полного
    прогона rerank_bge_zs_c40 (весь пайплайн, GPU; holdout скорится штатно,
    т.к. zero-shot оценивает все 500 калибровочных запросов)."""
    cfg = load_config("configs/default.yaml", experiment="rerank_bge_zs_c40.yaml")
    art = cfg.artifacts_dir
    manifest = json.loads((art / "scores" / "reranker" / "manifest.json").read_text())
    assert manifest["chosen"] == "zero_shot", "ожидали chosen=zero_shot (FT выключен)"

    splits = Splits.from_json(json.loads((art / "splits" / "folds.json").read_text()))
    from support_search.pipeline.stages import load_calibration
    gt = parse_ground_truth(load_calibration(cfg))
    gt_holdout = {q: gt[q] for q in splits.holdout}

    rr = ScoreMatrix.load(art / "scores" / "reranker" / "calibration.npz")
    blend = ScoreMatrix.load(art / "scores" / "blend" / "calibration.npz")
    print(f"— holdout n={len(gt_holdout)}, протокол: top-50, chunks={manifest.get('chunks_per_article')}")
    res_rr = _holdout_eval(rr, gt_holdout, "zero_shot_reranker/holdout")
    res_blend = _holdout_eval(blend, gt_holdout, "zero_shot_blend/holdout")
    _log_access(art, "final zero-shot: raw reranker + blend (одно обращение)", {
        "reranker_map10": round(res_rr.map_at_k, 4), "reranker_ci": [round(x, 4) for x in res_rr.ci],
        "blend_map10": round(res_blend.map_at_k, 4), "blend_ci": [round(x, 4) for x in res_blend.ci],
    })


def check_ft(cfg=None) -> None:
    """Дообученная модель: среднее 5 фолд-моделей на holdout (как на test) + blend.

    cfg=None → архивный labeled-прогон; иначе — переданный конфиг (напр. победитель
    свипа: load_config(..., overrides=["artifacts_dir=artifacts_rr_ft_k50"])). Блэнд —
    plain (без граф-фич): быстрая проверка переноса реранкера на holdout.
    """
    if cfg is None:
        cfg = load_config("configs/default.yaml", experiment="rerank_bge_ft_labeled.yaml")
    art = cfg.artifacts_dir
    manifest = json.loads((art / "scores" / "reranker" / "manifest.json").read_text())
    assert manifest["chosen"] == "fine_tune", "ожидали chosen=fine_tune в labeled-прогоне"
    m_chunks = int(manifest["chunks_per_article"])  # протокол архивного прогона (=3), не текущий конфиг
    top_k = int(manifest["candidate_top_k"])

    from support_search.pipeline.stages import _load_matrices, _reorder_rows, _texts_in_order, load_calibration
    from support_search.rerank import CrossEncoderReranker, candidate_lists, rerank_to_matrix
    from support_search.retrievers import DenseRetriever

    cal = load_calibration(cfg)
    gt = parse_ground_truth(cal)
    splits = Splits.from_json(json.loads((art / "splits" / "folds.json").read_text()))
    gt_holdout = {q: gt[q] for q in splits.holdout}

    fusion_cal = ScoreMatrix.load(art / "scores" / "fusion" / "calibration.npz")
    article_ids = fusion_cal.article_ids
    row_of = {int(q): i for i, q in enumerate(fusion_cal.query_ids)}
    h_ids = [int(q) for q in splits.holdout]
    h_rows = [row_of[q] for q in h_ids]

    dense = DenseRetriever.load(art / "scores" / "dense" / "index", encoder=None)
    q_emb = np.load(art / "scores" / "dense" / "q_emb_calibration.npy")
    q_emb_f = _reorder_rows(q_emb, cal["query_id"].to_numpy(), fusion_cal.query_ids)
    qtext_f = _texts_in_order(cal, fusion_cal.query_ids)

    cand_all = candidate_lists(fusion_cal, top_k)
    h_cand = [cand_all[i] for i in h_rows]
    h_pass = dense.top_chunk_texts(q_emb_f[h_rows], h_cand, m_chunks)
    h_text = [qtext_f[i] for i in h_rows]
    print(f"— holdout n={len(h_ids)}, протокол: top-{top_k}, chunks={m_chunks}; скорим 5 фолд-моделей…")

    accum = None
    for fold in range(splits.n_splits):
        model = CrossEncoderReranker(
            art / "models" / "reranker_ft" / f"fold_{fold}",
            max_length=512, batch_size=64, use_fp16=True,
        )
        m = rerank_to_matrix(model, h_text, h_ids, article_ids, h_cand, h_pass, source="reranker_ft")
        accum = m.scores if accum is None else accum + m.scores
        del model
        import torch; torch.cuda.empty_cache()
        print(f"  fold {fold}: готов")
    h_scores = accum / splits.n_splits

    # Матрица реранкера с заполненными holdout-строками (dev-строки — из прогона, OOF).
    rr_cal = ScoreMatrix.load(art / "scores" / "reranker" / "calibration.npz")
    fixed = rr_cal.scores.copy()
    for i, row in enumerate(h_rows):
        fixed[row] = h_scores[i]
    rr_fixed = ScoreMatrix(rr_cal.query_ids, rr_cal.article_ids, fixed, rr_cal.source)

    # Blend: рефит mini-LTR на dev (holdout в фите не участвует → коэффициенты те же,
    # что в blend_report; сверяем как санити) и предсказание для holdout-строк.
    from support_search.ranking import fit_blend_ltr, predict_blend_ltr
    names = ["bm25", "char_tfidf", "dense"]
    src_cal = _load_matrices(cfg, names, "calibration")
    ltr_cfg = cfg.section("ranking").ltr
    lr, feat_names = fit_blend_ltr(src_cal, rr_fixed, gt, splits.dev, names=names,
                                   l2=float(ltr_cfg.l2), use_ranks=bool(ltr_cfg.use_ranks))
    saved = json.loads((art / "runs" / "blend_report.json").read_text())["coefficients"]
    refit = lr.coefficients(feat_names)
    drift = max(abs(float(refit.get(k, 0.0)) - float(v)) for k, v in saved.items())
    print(f"  сверка коэффициентов LTR с blend_report: max|Δ|={drift:.4f} (ожидаем ~0)")
    blend_cal = predict_blend_ltr(lr, src_cal, rr_fixed, names=names, use_ranks=bool(ltr_cfg.use_ranks))

    res_rr = _holdout_eval(rr_fixed, gt_holdout, "ft_reranker(avg5)/holdout")
    res_blend = _holdout_eval(blend_cal, gt_holdout, "ft_blend/holdout")
    _log_access(art, "final fine-tuned: avg-5-fold reranker + blend (одно обращение)", {
        "reranker_map10": round(res_rr.map_at_k, 4), "reranker_ci": [round(x, 4) for x in res_rr.ci],
        "blend_map10": round(res_blend.map_at_k, 4), "blend_ci": [round(x, 4) for x in res_blend.ci],
        "ltr_coef_max_drift_vs_report": round(drift, 6),
    })


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else ""
    if variant == "zs":
        check_zs()
    elif variant == "ft":
        check_ft()
    elif variant == "ft_dir" and len(sys.argv) > 2:
        # Подтверждение произвольного FT-прогона (напр. победителя свипа) на его
        # holdout: plain blend, обращение логируется в holdout_log.json этого дира.
        check_ft(load_config("configs/default.yaml", overrides=[f"artifacts_dir={sys.argv[2]}"]))
    else:
        raise SystemExit("usage: holdout_check.py zs|ft|ft_dir <artifacts_dir>  "
                         "(каждый запуск = одно обращение к holdout!)")
