"""Стадии пайплайна: каждая — функция «артефакт → артефакт».

Стадии независимы (план §5.1.4): у каждой явные входы (артефакты предыдущих
стадий) и выходы, и каждая падает с понятной ошибкой, если вход отсутствует
(«запустите X»). Полный прогон (`cli all`) лишь вызывает их по порядку:

    preprocess → make-folds → retrieve → evaluate → make-answer → validate-answer
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from ..config import Config
from ..contracts import ScoreMatrix, check_answer_frame, check_score_matrix
from ..data.io import (
    load_articles,
    load_calibration,
    load_test,
    parse_ground_truth,
    read_json,
    write_json,
)
from ..data.splits import Splits, make_splits
from ..eval import evaluate, paired_permutation_test
from ..export import build_answer, validate_answer_file, write_answer
from ..fusion import oof_fusion_search, reciprocal_rank_fusion, search_weights, weighted_sum
from ..logging_utils import get_logger
from ..preprocess import Tokenizer, chunk_text
from ..retrievers import (
    RETRIEVER_REGISTRY,
    BM25Retriever,
    Corpus,
    DenseRetriever,
    E5Encoder,
    build_corpus,
)
from .artifacts import StageDir, hash_file, stage_dir

logger = get_logger("pipeline.stages")

# Версии схем артефактов — рост версии инвалидирует старый кэш (§5.1.6).
SCHEMA_CORPUS = 2  # +chunks.parquet
SCHEMA_SPLITS = 1
SCHEMA_SCORES = 1
SCHEMA_FUSION = 1


# ─── общие помощники ─────────────────────────────────────────────────────
def _data_path(cfg: Config, key: str) -> Path:
    data = cfg.section("data")
    return cfg.resolve(data.dir) / data[key]


def _data_hashes(cfg: Config, *keys: str) -> dict[str, str]:
    return {key: hash_file(_data_path(cfg, key)) for key in keys}


def build_tokenizer(cfg: Config) -> Tokenizer:
    """Единый токенизатор для корпуса и запросов из секции preprocess.normalize."""
    nrm = cfg.section("preprocess").normalize
    placeholders = OmegaConf.to_container(nrm.get("placeholders", {}), resolve=True)
    return Tokenizer(
        lemmatizer=str(nrm.get("lemmatizer", "auto")),
        tokenizer=str(nrm.get("tokenizer", "auto")),
        min_len=int(nrm.get("min_len", 2)),
        use_stopwords=bool(nrm.get("use_stopwords", True)),
        placeholders=placeholders,  # type: ignore[arg-type]
    )


def _enabled_retrievers(cfg: Config) -> list[str]:
    retr = cfg.section("retrievers")
    return [name for name in RETRIEVER_REGISTRY if name in retr and bool(retr[name].get("enabled", False))]


def _build_encoder(cfg: Config) -> E5Encoder:
    """E5-энкодер из секции retrievers.dense (грузит модель — только когда нужен)."""
    d = cfg.section("retrievers").dense
    device = d.get("device", None)
    return E5Encoder(
        str(d.model_name),
        device=str(device) if device else None,
        max_seq_len=int(d.max_seq_len), batch_size=int(d.batch_size),
        use_fp16=bool(d.use_fp16), query_prefix=str(d.query_prefix), passage_prefix=str(d.passage_prefix),
    )


def _build_retriever(cfg: Config, name: str, tokenizer: Tokenizer):
    sub = cfg.section("retrievers")[name]
    if name == "bm25":
        return BM25Retriever(
            tokenizer, k1=float(sub.k1), b=float(sub.b), title_weight=float(sub.title_weight)
        )
    if name == "char_tfidf":
        cls = RETRIEVER_REGISTRY[name]
        return cls(
            ngram_min=int(sub.ngram_min), ngram_max=int(sub.ngram_max), min_df=int(sub.min_df),
            sublinear_tf=bool(sub.sublinear_tf), title_weight=float(sub.title_weight),
        )
    if name == "dense":
        chunking = cfg.section("preprocess").chunking  # чанкинг — из препроцессинга
        return DenseRetriever(
            _build_encoder(cfg),
            max_tokens=int(chunking.max_tokens), overlap=int(chunking.overlap),
            max_chunks_per_article=int(chunking.max_chunks_per_article),
            title_prefix=bool(chunking.title_prefix), w_title=float(sub.w_title),
        )
    raise ValueError(f"неизвестный ретривер: {name}")


# ─── стадия: препроцессинг ───────────────────────────────────────────────
def stage_preprocess(cfg: Config, *, force: bool = False) -> Path:
    """articles.f → artifacts/corpus/corpus.parquet (article_id, title, clean_text)."""
    sd = StageDir(
        cfg, "corpus", stage_dir(cfg, "corpus"), SCHEMA_CORPUS,
        config_sections=("preprocess", "data"), input_hashes=_data_hashes(cfg, "articles"),
    )
    out_path = sd.path / "corpus.parquet"
    if out_path.exists() and sd.is_fresh() and not force:
        logger.info("corpus свеж — пропуск (%s)", out_path)
        return sd.path

    articles = load_articles(cfg)
    drop_tags = list(cfg.section("preprocess").html.drop_tags)
    corpus = build_corpus(articles, drop_tags=drop_tags)

    lengths = np.asarray([len(t) for t in corpus.texts])
    empty_frac = float((lengths == 0).mean())
    df = pd.DataFrame(
        {"article_id": corpus.article_ids, "title": corpus.titles, "clean_text": corpus.texts}
    )
    df.to_parquet(out_path, index=False)

    # Чанки статьи (для dense-ретривера) — отдельный артефакт того же корпуса.
    chunk_rows = _build_chunks(cfg, corpus)
    pd.DataFrame(chunk_rows, columns=["article_id", "chunk_id", "text"]).to_parquet(
        sd.path / "chunks.parquet", index=False
    )

    sd.write_manifest(
        n_articles=len(corpus),
        n_chunks=len(chunk_rows),
        empty_text_frac=round(empty_frac, 4),
        text_len_median=int(np.median(lengths)),
        text_len_max=int(lengths.max()),
    )
    logger.info(
        "preprocess: %d статей, %d чанков, пустой текст %.1f%%, медиана %d симв., макс %d симв.",
        len(corpus), len(chunk_rows), 100 * empty_frac, int(np.median(lengths)), int(lengths.max()),
    )
    return sd.path


def _build_chunks(cfg: Config, corpus: Corpus) -> list[tuple[int, int, str]]:
    """Разбить каждую статью на чанки по секции preprocess.chunking."""
    ch = cfg.section("preprocess").chunking
    rows: list[tuple[int, int, str]] = []
    for article_id, title, text in zip(corpus.article_ids, corpus.titles, corpus.texts):
        chunks = chunk_text(
            text, title=title, max_tokens=int(ch.max_tokens), overlap=int(ch.overlap),
            max_chunks=int(ch.max_chunks_per_article), title_prefix=bool(ch.title_prefix),
        ) or [str(title).strip() or " "]
        for chunk_id, chunk in enumerate(chunks):
            rows.append((int(article_id), chunk_id, chunk))
    return rows


def _load_corpus(cfg: Config) -> Corpus:
    path = stage_dir(cfg, "corpus") / "corpus.parquet"
    if not path.exists():
        raise FileNotFoundError("корпус не найден — сначала запустите `preprocess`")
    df = pd.read_parquet(path)
    return Corpus(
        article_ids=df["article_id"].to_numpy(dtype=np.int64),
        titles=df["title"].astype(str).tolist(),
        texts=df["clean_text"].astype(str).tolist(),
    )


# ─── стадия: разбиение ───────────────────────────────────────────────────
def stage_make_folds(cfg: Config, *, force: bool = False) -> Path:
    """calibration.f → artifacts/splits/folds.json (dev/holdout + 5-fold)."""
    sd = StageDir(
        cfg, "splits", stage_dir(cfg, "splits"), SCHEMA_SPLITS,
        config_sections=("folds",), input_hashes=_data_hashes(cfg, "calibration"),
    )
    folds_path = sd.path / "folds.json"
    if folds_path.exists() and sd.is_fresh() and not force:
        logger.info("splits свежи — пропуск (%s)", folds_path)
        return sd.path

    calibration = load_calibration(cfg)
    gt = parse_ground_truth(calibration)
    gt_counts = {qid: len(rel) for qid, rel in gt.items()}
    folds_cfg = cfg.section("folds")
    splits = make_splits(
        gt_counts, n_splits=int(folds_cfg.n_splits),
        holdout_frac=float(folds_cfg.holdout_frac), seed=int(folds_cfg.seed),
    )
    write_json(folds_path, splits.to_json())
    # Журнал обращений к holdout (бюджет ≤3, план §4.1) — создаём пустым один раз.
    holdout_log = sd.path / "holdout_log.json"
    if not holdout_log.exists():
        write_json(holdout_log, {"budget": 3, "accesses": []})

    sizes = {i: len(splits.fold(i)) for i in range(splits.n_splits)}
    sd.write_manifest(n_holdout=len(splits.holdout), n_dev=len(splits.dev), fold_sizes=sizes)
    logger.info(
        "make-folds: holdout=%d dev=%d, размеры фолдов=%s",
        len(splits.holdout), len(splits.dev), sizes,
    )
    return sd.path


def _load_splits(cfg: Config) -> Splits:
    path = stage_dir(cfg, "splits") / "folds.json"
    if not path.exists():
        raise FileNotFoundError("folds.json не найден — сначала запустите `make-folds`")
    return Splits.from_json(read_json(path))


# ─── стадия: ретрив ──────────────────────────────────────────────────────
def stage_retrieve(cfg: Config, *, force: bool = False) -> Path:
    """Матрицы скоров каждого включённого ретривера для calibration и test."""
    names = _enabled_retrievers(cfg)
    if not names:
        raise ValueError("в конфиге не включён ни один ретривер (retrievers.*.enabled)")

    corpus = _load_corpus(cfg)
    calibration = load_calibration(cfg)
    test = load_test(cfg)
    tokenizer = build_tokenizer(cfg)
    valid_ids = corpus.article_ids

    base = stage_dir(cfg, "scores")
    input_hashes = _data_hashes(cfg, "articles", "calibration", "test")
    for name in names:
        sd = StageDir(
            cfg, "scores", base / name, SCHEMA_SCORES,
            config_sections=("retrievers", "preprocess"), input_hashes=input_hashes,
        )
        if (sd.path / "calibration.npz").exists() and sd.is_fresh() and not force:
            logger.info("scores/%s свежи — пропуск", name)
            continue

        retriever = _build_retriever(cfg, name, tokenizer)
        retriever.fit(corpus)
        cal_scores = retriever.score_matrix(calibration["query_text"].tolist(), calibration["query_id"].tolist())
        test_scores = retriever.score_matrix(test["query_text"].tolist(), test["query_id"].tolist())
        check_score_matrix(cal_scores, valid_ids)
        check_score_matrix(test_scores, valid_ids)
        cal_scores.save(sd.path / "calibration.npz")
        test_scores.save(sd.path / "test.npz")
        retriever.save(sd.path / "index")
        sd.write_manifest(
            retriever=name, n_calibration=int(cal_scores.shape[0]),
            n_test=int(test_scores.shape[0]), n_articles=int(cal_scores.shape[1]),
            tokenizer=tokenizer.backend_info(),
        )
        logger.info("retrieve[%s]: calibration %s, test %s сохранены", name, cal_scores.shape, test_scores.shape)
    return base


# ─── стадия: оценка ──────────────────────────────────────────────────────
def stage_evaluate(cfg: Config) -> Path:
    """Оценить матрицы скоров на dev/holdout/all → experiments.csv + отчёт."""
    names = _enabled_retrievers(cfg)
    calibration = load_calibration(cfg)
    gt = parse_ground_truth(calibration)
    splits = _load_splits(cfg)
    eval_cfg = cfg.section("evaluation")
    k = int(eval_cfg.k)
    recall_ks = [int(x) for x in eval_cfg.recall_ks]
    depth = max(recall_ks + [k])

    subsets = {"dev": splits.dev, "holdout": splits.holdout, "all": sorted(gt)}
    rows: list[dict] = []
    dev_results = {}
    scores_dir = stage_dir(cfg, "scores")

    for name in names:
        matrix_path = scores_dir / name / "calibration.npz"
        if not matrix_path.exists():
            raise FileNotFoundError(f"нет матрицы {matrix_path} — сначала запустите `retrieve`")
        sm = ScoreMatrix.load(matrix_path)
        rankings = sm.rankings(depth)

        for split_name, qids in subsets.items():
            gt_subset = {q: gt[q] for q in qids}
            result = evaluate(rankings, gt_subset, name=f"{name}/{split_name}", k=k, recall_ks=recall_ks)
            result.compute_ci(n_resamples=int(eval_cfg.bootstrap_resamples), ci=float(eval_cfg.ci), seed=cfg.seed)
            rows.append({"retriever": name, "split": split_name, **result.to_row()})
            logger.info(result.summary())
            if split_name == "dev":
                dev_results[name] = result

    # Значимость различий на dev: каждый ретривер против BM25 (парный permutation).
    significance = {}
    if "bm25" in dev_results:
        base_ap = dev_results["bm25"].ap_array
        for name, res in dev_results.items():
            if name == "bm25":
                continue
            p = paired_permutation_test(res.ap_array, base_ap, seed=cfg.seed)
            significance[f"{name}_vs_bm25"] = round(p, 4)
            logger.info("значимость dev %s vs bm25: p=%.4f", name, p)

    runs_dir = stage_dir(cfg, "runs")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_hash": cfg.hash_section("retrievers", "preprocess", "folds"),
        "k": k,
        "rows": rows,
        "significance_vs_bm25_on_dev": significance,
    }
    write_json(runs_dir / "eval_report.json", report)
    _append_experiments(runs_dir / "experiments.csv", rows, report["config_hash"], report["created_at"])
    logger.info("evaluate: отчёт в %s", runs_dir / "eval_report.json")
    return runs_dir


def _append_experiments(path: Path, rows: list[dict], config_hash: str, created_at: str) -> None:
    """Обновить накопительную таблицу экспериментов.

    Каждая уникальная (config_hash, retriever, split) хранится один раз —
    повторный прогон того же конфига обновляет строку, а не плодит дубли.
    """
    stamped = [{"created_at": created_at, "config_hash": config_hash, **r} for r in rows]
    new = pd.DataFrame(stamped)
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new = new.drop_duplicates(subset=["config_hash", "retriever", "split"], keep="last").reset_index(drop=True)
    new.to_csv(path, index=False)


# ─── стадия: слияние (fusion) ────────────────────────────────────────────
def _fusion_sources(cfg: Config) -> list[str]:
    srcs = cfg.section("fusion").get("sources", None)
    return [str(s) for s in srcs] if srcs else _enabled_retrievers(cfg)


def _load_matrices(cfg: Config, names: Sequence[str], split: str) -> dict[str, ScoreMatrix]:
    scores_dir = stage_dir(cfg, "scores")
    out: dict[str, ScoreMatrix] = {}
    for name in names:
        path = scores_dir / name / f"{split}.npz"
        if not path.exists():
            raise FileNotFoundError(f"нет матрицы {path} — сначала запустите `retrieve`")
        out[name] = ScoreMatrix.load(path)
    return out


def _oof_candidates_frame(oof_rankings: dict[int, list[int]], top_k: int) -> pd.DataFrame:
    """OOF-ранжирования dev → таблица кандидатов (query_id, article_id, rank)."""
    rows_q, rows_a, rows_r = [], [], []
    for qid, ranking in sorted(oof_rankings.items()):
        for rank, aid in enumerate(ranking[:top_k]):
            rows_q.append(int(qid)); rows_a.append(int(aid)); rows_r.append(rank)
    return pd.DataFrame({"query_id": rows_q, "article_id": rows_a, "rank": rows_r})


def stage_fuse(cfg: Config) -> Path:
    """Слить источники (min-max weighted sum + RRF), честно оценить OOF, сохранить
    слитые матрицы и топ-кандидатов (потолок реранкера)."""
    fusion_cfg = cfg.section("fusion")
    names = _fusion_sources(cfg)
    if not names:
        raise ValueError("нет источников для fusion")

    cal = _load_matrices(cfg, names, "calibration")
    test = _load_matrices(cfg, names, "test")
    splits = _load_splits(cfg)
    gt = parse_ground_truth(load_calibration(cfg))
    gt_dev = {q: gt[q] for q in splits.dev}

    eval_cfg = cfg.section("evaluation")
    k = int(eval_cfg.k)
    recall_ks = [int(x) for x in eval_cfg.recall_ks]
    top_c = int(fusion_cfg.candidate_top_k)
    depth = max(recall_ks + [k, top_c])
    n_samples = int(fusion_cfg.random_search.n_samples)
    rrf_k = int(fusion_cfg.rrf_k)
    ci_kw = dict(n_resamples=int(eval_cfg.bootstrap_resamples), ci=float(eval_cfg.ci), seed=cfg.seed)

    rows: list[dict] = []
    # OOF: weighted sum (веса — random search на train(f)) и RRF (без параметров).
    oof_ws, fold_weights = oof_fusion_search(
        cal, gt, splits, names=names, method="weighted_sum", n_samples=n_samples, k=k, seed=cfg.seed, depth=depth
    )
    res_ws = evaluate(oof_ws, gt_dev, name="fusion_weighted/dev_oof", k=k, recall_ks=recall_ks).compute_ci(**ci_kw)
    rows.append({"retriever": "fusion_weighted", "split": "dev_oof", **res_ws.to_row()})
    logger.info(res_ws.summary())

    oof_rrf, _ = oof_fusion_search(
        cal, gt, splits, names=names, method="rrf", rrf_k=rrf_k, k=k, seed=cfg.seed, depth=depth
    )
    res_rrf = evaluate(oof_rrf, gt_dev, name="fusion_rrf/dev_oof", k=k, recall_ks=recall_ks).compute_ci(**ci_kw)
    rows.append({"retriever": "fusion_rrf", "split": "dev_oof", **res_rrf.to_row()})
    logger.info(res_rrf.summary())

    # Значимость fusion → BM25 на dev (парный permutation).
    significance: dict[str, float] = {}
    if "bm25" in cal:
        bm25_dev = evaluate(cal["bm25"].rankings(depth), gt_dev, name="bm25/dev", k=k, recall_ks=recall_ks)
        p = paired_permutation_test(res_ws.ap_array, bm25_dev.ap_array, seed=cfg.seed)
        significance["fusion_weighted_vs_bm25"] = round(p, 4)
        logger.info("значимость dev fusion_weighted vs bm25: p=%.4f", p)

    # Финал: веса на всех dev → применяем к cal и test.
    method = str(fusion_cfg.method)
    if method == "weighted_sum":
        final_weights = search_weights(cal, gt, splits.dev, names=names, n_samples=n_samples, k=k, seed=cfg.seed)
        fused_cal = weighted_sum(cal, final_weights)
        fused_test = weighted_sum(test, final_weights)
        logger.info("итоговые веса fusion (dev): %s", {n: round(w, 3) for n, w in final_weights.items()})
    else:
        final_weights = {}
        fused_cal = reciprocal_rank_fusion(cal, k=rrf_k)
        fused_test = reciprocal_rank_fusion(test, k=rrf_k)

    valid_ids = cal[names[0]].article_ids
    check_score_matrix(fused_cal, valid_ids)
    check_score_matrix(fused_test, valid_ids)
    sd = StageDir(
        cfg, "scores", stage_dir(cfg, "scores", subdir="fusion"), SCHEMA_FUSION,
        config_sections=("fusion", "retrievers", "preprocess"),
        input_hashes=_data_hashes(cfg, "articles", "calibration", "test"),
    )
    fused_cal.save(sd.path / "calibration.npz")
    fused_test.save(sd.path / "test.npz")
    sd.write_manifest(
        method=method, sources=list(names), final_weights=final_weights,
        fold_weights={str(f): w for f, w in fold_weights.items()},
        oof_map=round(res_ws.map_at_k, 4), recall_at_50_oof=round(res_ws.recall_at_k.get(50, 0.0), 4),
    )

    # Кандидаты для реранкера (этап 6): test (финальные веса) и dev OOF (честные).
    cand_dir = stage_dir(cfg, "candidates")
    fused_test.to_retrieval_frame(top_c).to_parquet(cand_dir / "test.parquet", index=False)
    _oof_candidates_frame(oof_ws if method == "weighted_sum" else oof_rrf, top_c).to_parquet(
        cand_dir / "dev_oof.parquet", index=False
    )

    runs_dir = stage_dir(cfg, "runs")
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config_hash = cfg.hash_section("fusion", "retrievers", "preprocess")
    write_json(
        runs_dir / "fusion_report.json",
        {"created_at": created_at, "config_hash": config_hash, "method": method, "sources": list(names),
         "final_weights": final_weights, "fold_weights": {str(f): w for f, w in fold_weights.items()},
         "significance": significance, "rows": rows, "recall_at_50_oof": round(res_ws.recall_at_k.get(50, 0.0), 4)},
    )
    _append_experiments(runs_dir / "experiments.csv", rows, config_hash, created_at)

    r50 = res_ws.recall_at_k.get(50, 0.0)
    logger.info(
        "fuse: OOF recall@%d=%.3f (потолок реранкера) — %s",
        top_c, r50, "OK ≥0.97" if r50 >= 0.97 else "ниже 0.97, см. план §5 (расширить кандидатов)",
    )
    return sd.path


# ─── стадия: ответ ───────────────────────────────────────────────────────
def stage_make_answer(cfg: Config) -> Path:
    """Матрица скоров primary_source на test → artifacts/answers/answer.csv."""
    export_cfg = cfg.section("export")
    primary = str(export_cfg.primary_source)
    top_k = int(export_cfg.top_k)

    matrix_path = stage_dir(cfg, "scores") / primary / "test.npz"
    if not matrix_path.exists():
        raise FileNotFoundError(f"нет матрицы {matrix_path} — сначала запустите `retrieve`")

    test = load_test(cfg)
    articles = load_articles(cfg)
    sm = ScoreMatrix.load(matrix_path)
    rankings = sm.rankings(top_k)
    answer = build_answer(rankings, test["query_id"].tolist(), top_k=top_k)
    check_answer_frame(
        answer, expected_query_ids=test["query_id"].tolist(),
        valid_article_ids=articles["article_id"].tolist(), max_k=top_k,
    )

    answers_dir = stage_dir(cfg, "answers")
    out_path = answers_dir / "answer.csv"
    write_answer(answer, out_path)
    OmegaConf.save(cfg.raw, answers_dir / "config_snapshot.yaml")
    logger.info("make-answer[%s]: %d ответов → %s", primary, len(answer), out_path)
    return out_path


def stage_validate_answer(cfg: Config) -> Path:
    """Проверить artifacts/answers/answer.csv против test.f и корпуса."""
    out_path = stage_dir(cfg, "answers") / "answer.csv"
    test = load_test(cfg)
    articles = load_articles(cfg)
    validate_answer_file(out_path, test=test, articles=articles, max_k=int(cfg.section("export").top_k))
    return out_path
