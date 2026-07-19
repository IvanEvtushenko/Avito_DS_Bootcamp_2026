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
from ..features import GraphFeaturizer, article_vectors, link_adjacency
from ..fusion import (
    fit_fusion_ltr,
    oof_fusion_ltr,
    oof_fusion_search,
    predict_fusion_ltr,
    reciprocal_rank_fusion,
    search_weights,
    weighted_sum,
)
from ..logging_utils import get_logger
from ..preprocess import Tokenizer, chunk_text
from ..ranking import (
    MLPRankerLTR,
    blend_matrix,
    fit_blend_ltr,
    oof_blend,
    oof_blend_ltr,
    predict_blend_ltr,
    search_blend_weight,
)
from ..rerank import (
    CrossEncoderReranker,
    build_reranker,
    build_training_groups,
    candidate_lists,
    fine_tune_reranker,
    rerank_to_matrix,
)
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
SCHEMA_RERANK = 1
SCHEMA_BLEND = 1


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
        pooling=str(d.get("pooling", "mean")), model_class=str(d.get("model_class", "auto")),
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
        sd.path.mkdir(parents=True, exist_ok=True)  # np.save (dense q_emb) не создаёт каталог

        retriever = _build_retriever(cfg, name, tokenizer)
        retriever.fit(corpus)
        if isinstance(retriever, DenseRetriever):
            # Кэшируем эмбеддинги запросов — их же использует реранкер для выбора
            # лучшего чанка, без повторной загрузки E5 (этап 6).
            q_cal = retriever.encode_queries(calibration["query_text"].tolist())
            q_test = retriever.encode_queries(test["query_text"].tolist())
            cal_scores = retriever.score_from_query_embeddings(q_cal, calibration["query_id"].tolist())
            test_scores = retriever.score_from_query_embeddings(q_test, test["query_id"].tolist())
            np.save(sd.path / "q_emb_calibration.npy", q_cal)
            np.save(sd.path / "q_emb_test.npy", q_test)
        else:
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

    ltr_cfg = fusion_cfg.ltr
    l2 = float(ltr_cfg.l2)
    use_ranks = bool(ltr_cfg.use_ranks)
    method = str(fusion_cfg.method)
    objective = str(fusion_cfg.get("objective", "recall"))  # recall@top_c (tie MAP) | map

    rows: list[dict] = []
    # OOF всех трёх способов для сравнения; финал берёт настроенный method.
    # Цель подбора весов для кандидатов — recall@candidate_top_k (tie MAP@k).
    oof_ws, fold_weights, oof_ws_m = oof_fusion_search(
        cal, gt, splits, names=names, method="weighted_sum",
        n_samples=n_samples, k=k, recall_k=top_c, objective=objective, seed=cfg.seed, depth=depth,
    )
    res_ws = evaluate(oof_ws, gt_dev, name="fusion_weighted/dev_oof", k=k, recall_ks=recall_ks).compute_ci(**ci_kw)
    rows.append({"retriever": "fusion_weighted", "split": "dev_oof", **res_ws.to_row()})
    logger.info(res_ws.summary())

    oof_rrf, _, oof_rrf_m = oof_fusion_search(
        cal, gt, splits, names=names, method="rrf", rrf_k=rrf_k, k=k, recall_k=top_c, seed=cfg.seed, depth=depth
    )
    res_rrf = evaluate(oof_rrf, gt_dev, name="fusion_rrf/dev_oof", k=k, recall_ks=recall_ks).compute_ci(**ci_kw)
    rows.append({"retriever": "fusion_rrf", "split": "dev_oof", **res_rrf.to_row()})
    logger.info(res_rrf.summary())

    oof_lr, feature_names, oof_lr_m = oof_fusion_ltr(cal, gt, splits, names=names, l2=l2, use_ranks=use_ranks, depth=depth)
    res_lr = evaluate(oof_lr, gt_dev, name="fusion_lr/dev_oof", k=k, recall_ks=recall_ks).compute_ci(**ci_kw)
    rows.append({"retriever": "fusion_lr", "split": "dev_oof", **res_lr.to_row()})
    logger.info(res_lr.summary())

    oof_by_method = {
        "weighted_sum": (oof_ws, res_ws, oof_ws_m),
        "rrf": (oof_rrf, res_rrf, oof_rrf_m),
        "lr": (oof_lr, res_lr, oof_lr_m),
    }
    oof_primary, res_primary, oof_primary_m = oof_by_method[method]

    # Значимость fusion(method) → BM25 на dev (парный permutation).
    significance: dict[str, float] = {}
    if "bm25" in cal:
        bm25_dev = evaluate(cal["bm25"].rankings(depth), gt_dev, name="bm25/dev", k=k, recall_ks=recall_ks)
        p = paired_permutation_test(res_primary.ap_array, bm25_dev.ap_array, seed=cfg.seed)
        significance[f"fusion_{method}_vs_bm25"] = round(p, 4)
        logger.info("значимость dev fusion_%s vs bm25: p=%.4f", method, p)

    # Финал: обучаем/подбираем на всех dev → all-dev fusion для cal и test.
    final_weights: dict = {}
    coefficients: dict = {}
    if method == "weighted_sum":
        final_weights = search_weights(cal, gt, splits.dev, names=names, n_samples=n_samples, k=k, recall_k=top_c, objective=objective, seed=cfg.seed)
        fused_cal = weighted_sum(cal, final_weights)
        fused_test = weighted_sum(test, final_weights)
        logger.info("итоговые веса fusion (dev): %s", {n: round(w, 3) for n, w in final_weights.items()})
    elif method == "lr":
        lr, feature_names = fit_fusion_ltr(cal, gt, splits.dev, names=names, l2=l2, use_ranks=use_ranks)
        fused_cal = predict_fusion_ltr(lr, cal, names=names, use_ranks=use_ranks)
        fused_test = predict_fusion_ltr(lr, test, names=names, use_ranks=use_ranks)
        coefficients = lr.coefficients(feature_names)
        logger.info("итоговые коэффициенты LR fusion (dev): %s", coefficients)
    else:  # rrf
        fused_cal = reciprocal_rank_fusion(cal, k=rrf_k)
        fused_test = reciprocal_rank_fusion(test, k=rrf_k)

    # БЕЗ УТЕЧКИ (§4.2): строки dev в calibration.npz — из OOF (веса не видели свой
    # запрос), holdout — из all-dev. Реранкер и блэнд берут кандидаты/признаки dev
    # именно отсюда, поэтому их OOF-оценка честная.
    assert np.array_equal(fused_cal.query_ids, oof_primary_m.query_ids)
    row_of_cal = {int(q): i for i, q in enumerate(fused_cal.query_ids)}
    for q in splits.dev:
        fused_cal.scores[row_of_cal[q]] = oof_primary_m.scores[row_of_cal[q]]

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
    recall_ceiling = round(res_primary.recall_at_k.get(top_c, 0.0), 4)
    sd.write_manifest(
        method=method, sources=list(names), final_weights=final_weights, coefficients=coefficients,
        fold_weights={str(f): w for f, w in fold_weights.items()},
        oof_map=round(res_primary.map_at_k, 4), recall_at_ceiling_oof=recall_ceiling, ceiling_k=top_c,
    )

    # Кандидаты для реранкера (этап 6): test (финал method) и dev OOF (честные).
    cand_dir = stage_dir(cfg, "candidates")
    fused_test.to_retrieval_frame(top_c).to_parquet(cand_dir / "test.parquet", index=False)
    _oof_candidates_frame(oof_primary, top_c).to_parquet(cand_dir / "dev_oof.parquet", index=False)

    runs_dir = stage_dir(cfg, "runs")
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config_hash = cfg.hash_section("fusion", "retrievers", "preprocess")
    write_json(
        runs_dir / "fusion_report.json",
        {"created_at": created_at, "config_hash": config_hash, "method": method, "sources": list(names),
         "final_weights": final_weights, "coefficients": coefficients,
         "fold_weights": {str(f): w for f, w in fold_weights.items()},
         "significance": significance, "rows": rows, "recall_at_ceiling_oof": recall_ceiling, "ceiling_k": top_c},
    )
    # OOF per-query AP гибрида (method) — для проверки значимости hybrid→rerank (этап 7).
    write_json(runs_dir / "fusion_oof_ap.json", {str(q): ap for q, ap in res_primary.per_query_ap.items()})
    _append_experiments(runs_dir / "experiments.csv", rows, config_hash, created_at)

    logger.info(
        "fuse[%s]: OOF recall@%d=%.3f (потолок реранкера) — %s",
        method, top_c, recall_ceiling, "OK ≥0.95" if recall_ceiling >= 0.95 else "ниже 0.95, расширить кандидатов",
    )
    return sd.path


# ─── стадия: реранкер (cross-encoder) ────────────────────────────────────
def _texts_in_order(df: pd.DataFrame, query_ids) -> list[str]:
    lookup = dict(zip(df["query_id"].tolist(), df["query_text"].tolist()))
    return [str(lookup[int(q)]) for q in query_ids]


def _reorder_rows(rows: np.ndarray, source_ids, target_ids) -> np.ndarray:
    row_of = {int(q): i for i, q in enumerate(source_ids)}
    return rows[[row_of[int(q)] for q in target_ids]]


def _free_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - torch может отсутствовать
        pass


def _synthetic_training_groups(cfg: Config, ft_cfg, *, n_negatives: int) -> list:
    """Обучающие группы реранкера из синтетических запросов (идея №1 MAP_IMPROVEMENT_IDEAS).

    CSV формата calibration (query_id, query_text, ground_truth). Позитив — GT-статья
    синтетического запроса; хард-негативы — верх dense-выдачи по этому запросу
    (GT исключается внутри build_training_groups). Пассаж — лучший dense-чанк.
    Разметка calibration в обучении не участвует → одна модель честна для dev OOF.
    """
    path = cfg.resolve(str(ft_cfg.synthetic_path))
    df = pd.read_csv(path)
    gt = parse_ground_truth(df)
    qids = [int(q) for q in df["query_id"]]
    qtext_of = {int(q): str(t) for q, t in zip(df["query_id"], df["query_text"])}

    encoder = _build_encoder(cfg)
    dense = DenseRetriever.load(stage_dir(cfg, "scores") / "dense" / "index", encoder=encoder)
    # GT вне корпуса (не должно случаться) не роняет пайплайн, а выпадает из групп.
    known = {int(a) for a in dense.article_ids}
    gt = {q: {a for a in arts if int(a) in known} for q, arts in gt.items()}

    q_emb = dense.encode_queries([qtext_of[q] for q in qids])
    mine_top_k = int(ft_cfg.get("mine_top_k", 30))
    ranked = dense.score_from_query_embeddings(q_emb, qids).rankings(mine_top_k)

    union_lists = [sorted(set(ranked[q]) | gt.get(q, set())) for q in qids]
    union_pass = dense.best_chunk_texts(q_emb, union_lists)
    passage_of = {
        (q, int(a)): union_pass[i][j]
        for i, q in enumerate(qids) for j, a in enumerate(union_lists[i])
    }
    groups = build_training_groups(qids, qtext_of, gt, {q: ranked[q] for q in qids}, passage_of, n_negatives=n_negatives)
    del dense, encoder
    _free_cuda()
    return groups


def stage_rerank(cfg: Config, *, force: bool = False) -> Path:
    """Cross-encoder reranker над топ-K кандидатов: zero-shot и (опц.) fine-tune."""
    rr = cfg.section("reranker")
    if not bool(rr.enabled):
        raise ValueError("reranker.enabled=false — стадия rerank отключена")
    top_k = int(rr.candidate_top_k)
    scores_dir = stage_dir(cfg, "scores")

    fusion_cal = ScoreMatrix.load(_require(scores_dir / "fusion" / "calibration.npz", "fuse"))
    fusion_test = ScoreMatrix.load(_require(scores_dir / "fusion" / "test.npz", "fuse"))
    dense_index = _require(scores_dir / "dense" / "index" / "meta.json", "retrieve").parent
    dense = DenseRetriever.load(dense_index, encoder=None)
    q_cal = np.load(_require(scores_dir / "dense" / "q_emb_calibration.npy", "retrieve"))
    q_test = np.load(scores_dir / "dense" / "q_emb_test.npy")

    cal = load_calibration(cfg)
    test = load_test(cfg)
    gt = parse_ground_truth(cal)
    article_ids = fusion_cal.article_ids
    splits = _load_splits(cfg)
    gt_dev = {q: gt[q] for q in splits.dev}
    eval_cfg = cfg.section("evaluation")
    k = int(eval_cfg.k)
    recall_ks = [int(x) for x in eval_cfg.recall_ks]
    depth = max(recall_ks + [k])
    ci_kw = dict(n_resamples=int(eval_cfg.bootstrap_resamples), ci=float(eval_cfg.ci), seed=cfg.seed)

    # Всё — в порядке строк fusion-матрицы (канонический порядок запросов).
    qtext_cal = _texts_in_order(cal, fusion_cal.query_ids)
    qtext_test = _texts_in_order(test, fusion_test.query_ids)
    q_cal_f = _reorder_rows(q_cal, cal["query_id"].to_numpy(), fusion_cal.query_ids)
    q_test_f = _reorder_rows(q_test, test["query_id"].to_numpy(), fusion_test.query_ids)
    cand_cal = candidate_lists(fusion_cal, top_k)
    cand_test = candidate_lists(fusion_test, top_k)
    m_chunks = int(rr.get("chunks_per_article", 1))
    if m_chunks > 1:
        # Идея №3 (MAP_IMPROVEMENT_IDEAS): реранкер скорит топ-m чанков статьи,
        # скор статьи = max — не наследуем ошибку единственного dense-чанка.
        pass_cal = dense.top_chunk_texts(q_cal_f, cand_cal, m_chunks)
        pass_test = dense.top_chunk_texts(q_test_f, cand_test, m_chunks)
    else:
        pass_cal = dense.best_chunk_texts(q_cal_f, cand_cal)
        pass_test = dense.best_chunk_texts(q_test_f, cand_test)

    # --- Zero-shot ---
    device = rr.get("device", None)
    backend = str(rr.get("backend", "sequence_classification"))
    reranker = build_reranker(rr)  # выбор модели по reranker.backend (дефолт — bge cross-encoder)
    zs_cal = rerank_to_matrix(reranker, qtext_cal, fusion_cal.query_ids, article_ids, cand_cal, pass_cal, source="reranker_zs")
    zs_test = rerank_to_matrix(reranker, qtext_test, fusion_test.query_ids, article_ids, cand_test, pass_test, source="reranker_zs")
    del reranker
    _free_cuda()

    rows: list[dict] = []
    res_zs = evaluate(zs_cal.rankings(depth), gt_dev, name="reranker_zs/dev_oof", k=k, recall_ks=recall_ks).compute_ci(**ci_kw)
    rows.append({"retriever": "reranker_zs", "split": "dev_oof", **res_zs.to_row()})
    logger.info(res_zs.summary())

    chosen_cal, chosen_test, chosen = zs_cal, zs_test, "zero_shot"
    res_ft = None

    # --- Fine-tune: data=labeled (по фолдам, честный OOF) | data=synthetic (одна модель) ---
    ft_cfg = rr.fine_tune
    if bool(ft_cfg.enabled):
        # FT реализован только для cross-encoder head (bge). Для causal-LM/listwise/GGUF
        # backend-ов дообучение не поддержано — падаем сразу, а не молча инференсим не то.
        if backend != "sequence_classification":
            raise ValueError(f"reranker.fine_tune не поддержан для backend={backend!r} (только sequence_classification)")
        n_neg = int(ft_cfg.n_negatives)
        ft_data = str(ft_cfg.get("data", "labeled"))  # labeled | synthetic
        tune_kw = dict(
            epochs=int(ft_cfg.epochs), lr=float(ft_cfg.lr), batch_groups=int(ft_cfg.batch_groups),
            max_length=int(ft_cfg.max_length), use_fp16=bool(rr.use_fp16),
            freeze_embeddings=bool(ft_cfg.get("freeze_embeddings", True)),
            schedule=str(ft_cfg.get("schedule", "constant")),
            warmup_ratio=float(ft_cfg.get("warmup_ratio", 0.0)),
            device=str(device) if device else None,
        )
        models_dir = stage_dir(cfg, "models", subdir="reranker_ft")
        # reuse_models: путь к готовым фолд-моделям (обученным раньше). Если задан —
        # НЕ обучаем, а грузим и только прогоняем инференс. Так одну ночную тренировку
        # можно переиспользовать под любую правку ретрива/препроцессинга (напр. лемматизация
        # BM25): пассажи реранкера от неё не зависят, а фолды детерминированы (seed) →
        # dev OOF остаётся честным. Веса не смешиваются: инференс их не меняет.
        reuse = ft_cfg.get("reuse_models", None)
        reuse_dir = cfg.resolve(str(reuse)) if reuse else None
        if reuse_dir is not None and not reuse_dir.exists():
            raise FileNotFoundError(f"reranker.fine_tune.reuse_models: нет каталога {reuse_dir}")
        # resume_folds: добивка убитого прогона — полные фолд-модели ЭТОГО artifacts_dir
        # переиспользуются, обучаются только недостающие. Опт-ин через
        # `--set reranker.fine_tune.resume_folds=true`: молча не реюзаем, потому что
        # совместимость весов со сменившимся конфигом здесь не проверяется.
        resume_folds = bool(ft_cfg.get("resume_folds", False))
        n_resumed = 0

        if ft_data == "synthetic":
            # Обучение не видит разметку calibration → модель честно OOF для dev и test,
            # фолды не нужны: одна модель, один инференс-проход.
            if reuse_dir is not None:
                model_dir = reuse_dir / "synthetic"
                if not model_dir.exists():
                    raise FileNotFoundError(f"reuse_models: нет {model_dir}")
                logger.info("fine-tune (synthetic): reuse %s (без обучения)", model_dir)
            else:
                groups = _synthetic_training_groups(cfg, ft_cfg, n_negatives=n_neg)
                logger.info("fine-tune (synthetic): %d обучающих групп", len(groups))
                model_dir = fine_tune_reranker(str(rr.model_name), groups, models_dir / "synthetic", seed=cfg.seed, **tune_kw)
            ft_model = CrossEncoderReranker(
                model_dir, device=str(device) if device else None,
                max_length=int(rr.max_length), batch_size=int(rr.batch_size), use_fp16=bool(rr.use_fp16),
            )
            ft_cal = rerank_to_matrix(ft_model, qtext_cal, fusion_cal.query_ids, article_ids, cand_cal, pass_cal, source="reranker_ft")
            ft_test = rerank_to_matrix(ft_model, qtext_test, fusion_test.query_ids, article_ids, cand_test, pass_test, source="reranker_ft")
            del ft_model
            _free_cuda()
        else:
            qtext_by_qid = {int(q): qtext_cal[i] for i, q in enumerate(fusion_cal.query_ids)}
            cand_by_qid = {int(q): cand_cal[i] for i, q in enumerate(fusion_cal.query_ids)}
            # Обучающие пассажи (single best chunk) нужны только при реальном обучении.
            if reuse_dir is None:
                union_lists = [sorted(set(cand_cal[i]) | gt.get(int(q), set())) for i, q in enumerate(fusion_cal.query_ids)]
                union_pass = dense.best_chunk_texts(q_cal_f, union_lists)
                passage_of = {
                    (int(q), int(a)): union_pass[i][j]
                    for i, q in enumerate(fusion_cal.query_ids) for j, a in enumerate(union_lists[i])
                }

            row_of = {int(q): i for i, q in enumerate(fusion_cal.query_ids)}
            oof_scores = np.full_like(zs_cal.scores, fill_value=zs_cal.scores.min())
            test_accum = np.zeros_like(zs_test.scores)
            for fold in range(splits.n_splits):
                if reuse_dir is not None:
                    fold_dir = reuse_dir / f"fold_{fold}"
                    if not fold_dir.exists():
                        raise FileNotFoundError(f"reuse_models: нет {fold_dir}")
                    logger.info("fine-tune fold %d: reuse %s (без обучения)", fold, fold_dir)
                else:
                    fold_dir = models_dir / f"fold_{fold}"
                    # Полнота сохранённой модели: save_pretrained пишет config+веса,
                    # затем токенайзер — проверяем последний записываемый файл тоже.
                    complete = (fold_dir / "config.json").exists() and (fold_dir / "tokenizer_config.json").exists()
                    if resume_folds and complete:
                        n_resumed += 1
                        logger.info("fine-tune fold %d: resume %s (обучение пропущено)", fold, fold_dir)
                    else:
                        groups = build_training_groups(
                            splits.train_ids(fold), qtext_by_qid, gt, cand_by_qid, passage_of, n_negatives=n_neg
                        )
                        logger.info("fine-tune fold %d: %d обучающих групп", fold, len(groups))
                        fold_dir = fine_tune_reranker(
                            str(rr.model_name), groups, fold_dir, seed=cfg.seed + fold, **tune_kw
                        )
                fold_model = CrossEncoderReranker(
                    fold_dir, device=str(device) if device else None,
                    max_length=int(rr.max_length), batch_size=int(rr.batch_size), use_fp16=bool(rr.use_fp16),
                )
                val_ids = splits.fold(fold)
                val_matrix = rerank_to_matrix(
                    fold_model, [qtext_by_qid[q] for q in val_ids], val_ids, article_ids,
                    [cand_by_qid[q] for q in val_ids], [pass_cal[row_of[q]] for q in val_ids], source="reranker_ft",
                )
                for i, q in enumerate(val_ids):
                    oof_scores[row_of[q]] = val_matrix.scores[i]
                test_matrix = rerank_to_matrix(
                    fold_model, qtext_test, fusion_test.query_ids, article_ids, cand_test, pass_test, source="reranker_ft"
                )
                test_accum += test_matrix.scores
                del fold_model
                _free_cuda()

            ft_cal = ScoreMatrix(fusion_cal.query_ids, article_ids, oof_scores, "reranker_ft")
            ft_test = ScoreMatrix(fusion_test.query_ids, article_ids, test_accum / splits.n_splits, "reranker_ft")

        res_ft = evaluate(ft_cal.rankings(depth), gt_dev, name="reranker_ft/dev_oof", k=k, recall_ks=recall_ks).compute_ci(**ci_kw)
        rows.append({"retriever": "reranker_ft", "split": "dev_oof", **res_ft.to_row()})
        logger.info(res_ft.summary())
        # Фолбэк (§6): FT берём, только если он бьёт zero-shot по OOF MAP@10.
        if res_ft.map_at_k > res_zs.map_at_k:
            chosen_cal, chosen_test, chosen = ft_cal, ft_test, "fine_tune"
        else:
            logger.info("FT (%.4f) не бьёт zero-shot (%.4f) — остаёмся на zero-shot (§6 фолбэк)",
                        res_ft.map_at_k, res_zs.map_at_k)

    # Канонический выход реранкера (его читает blend).
    sd = StageDir(
        cfg, "scores", stage_dir(cfg, "scores", subdir="reranker"), SCHEMA_RERANK,
        config_sections=("reranker", "fusion", "retrievers", "preprocess"),
        input_hashes=_data_hashes(cfg, "articles", "calibration", "test"),
    )
    chosen_cal.save(sd.path / "calibration.npz")
    chosen_test.save(sd.path / "test.npz")
    sd.write_manifest(chosen=chosen, candidate_top_k=top_k, chunks_per_article=m_chunks,
                      fine_tune_data=str(ft_cfg.get("data", "labeled")) if bool(ft_cfg.enabled) else None,
                      fine_tune_reused=str(ft_cfg.get("reuse_models")) if bool(ft_cfg.enabled) and ft_cfg.get("reuse_models") else None,
                      fine_tune_folds_resumed=n_resumed if bool(ft_cfg.enabled) and n_resumed else None,
                      zero_shot_map=round(res_zs.map_at_k, 4),
                      fine_tune_map=round(res_ft.map_at_k, 4) if res_ft else None)

    runs_dir = stage_dir(cfg, "runs")
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config_hash = cfg.hash_section("reranker", "fusion", "retrievers", "preprocess")
    write_json(runs_dir / "rerank_report.json",
               {"created_at": created_at, "config_hash": config_hash, "chosen": chosen, "rows": rows})
    _append_experiments(runs_dir / "experiments.csv", rows, config_hash, created_at)
    logger.info("rerank: выбран %s → scores/reranker/", chosen)
    return sd.path


def _require(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"нет {path} — сначала запустите `{produced_by}`")
    return path


# ─── стадия: финальный блэнд ─────────────────────────────────────────────
def _build_graph_featurizer(cfg: Config, article_ids: np.ndarray) -> GraphFeaturizer:
    """Собрать фолд-независимые входы граф-фич: HTML-ссылки + dense-векторы статей.

    Всё — из уже готовых артефактов (articles.f и кэш dense-индекса), порядок
    статей приводится к каноническому порядку колонок матриц скоров.
    """
    graph_cfg = cfg.section("ranking").graph_features
    articles = load_articles(cfg)
    body_of = {int(a): str(b) for a, b in zip(articles["article_id"], articles["body"])}
    adj = link_adjacency(article_ids, [body_of[int(a)] for a in article_ids])

    dense_index = _require(stage_dir(cfg, "scores") / "dense" / "index" / "meta.json", "retrieve").parent
    dense = DenseRetriever.load(dense_index, encoder=None)
    vec = article_vectors(dense.chunk_emb, dense.title_emb, dense.chunk_offsets)
    vec = _reorder_rows(vec, dense.article_ids, article_ids)
    logger.info("граф-фичи: %d статей, %d link-пар, dim=%d, top_m=%d",
                len(article_ids), int(adj.sum()) // 2, vec.shape[1], int(graph_cfg.top_m))
    return GraphFeaturizer(article_ids, link_adj=adj, article_vec=vec, top_m=int(graph_cfg.top_m),
                           interaction=bool(graph_cfg.get("interaction", True)),
                           co_weight=str(graph_cfg.get("co_weight", "cond")))


def stage_blend(cfg: Config) -> Path:
    """Финальное ранжирование: mini-LTR (LR/MLP-голова, опц. граф-фичи) или ручной blend.

    Все включённые варианты оцениваются честным OOF и попадают в отчёт;
    `ranking.method: auto` берёт лучший по OOF MAP@10 (фолбэк-паттерн §6).
    """
    scores_dir = stage_dir(cfg, "scores")
    fusion_cal = ScoreMatrix.load(_require(scores_dir / "fusion" / "calibration.npz", "fuse"))
    fusion_test = ScoreMatrix.load(scores_dir / "fusion" / "test.npz")
    reranker_cal = ScoreMatrix.load(_require(scores_dir / "reranker" / "calibration.npz", "rerank"))
    reranker_test = ScoreMatrix.load(scores_dir / "reranker" / "test.npz")
    names = _fusion_sources(cfg)  # источники для mini-LTR: bm25/char/dense
    src_cal = _load_matrices(cfg, names, "calibration")
    src_test = _load_matrices(cfg, names, "test")

    gt = parse_ground_truth(load_calibration(cfg))
    splits = _load_splits(cfg)
    gt_dev = {q: gt[q] for q in splits.dev}
    eval_cfg = cfg.section("evaluation")
    k = int(eval_cfg.k)
    recall_ks = [int(x) for x in eval_cfg.recall_ks]
    depth = max(recall_ks + [k])
    ranking_cfg = cfg.section("ranking")
    method = str(ranking_cfg.method)
    grid = int(ranking_cfg.weight_grid)
    l2 = float(ranking_cfg.ltr.l2)
    use_ranks = bool(ranking_cfg.ltr.use_ranks)
    graph_cfg = ranking_cfg.get("graph_features", None)
    mlp_cfg = ranking_cfg.get("mlp", None)
    ci_kw = dict(n_resamples=int(eval_cfg.bootstrap_resamples), ci=float(eval_cfg.ci), seed=cfg.seed)

    featurizer = None
    if graph_cfg is not None and bool(graph_cfg.enabled):
        featurizer = _build_graph_featurizer(cfg, fusion_cal.article_ids)

    def _make_mlp() -> MLPRankerLTR:
        return MLPRankerLTR(
            hidden=[int(h) for h in mlp_cfg.hidden], epochs=int(mlp_cfg.epochs),
            lr=float(mlp_cfg.lr), weight_decay=float(mlp_cfg.weight_decay), seed=cfg.seed,
        )

    # OOF всех включённых вариантов — в отчёт; финал берёт настроенный method
    # (auto = лучший по OOF MAP@10, фолбэк-паттерн §6).
    rows: list[dict] = []
    results: dict[str, object] = {}

    def _add_variant(variant: str, oof_rankings: dict[int, list[int]]):
        res = evaluate(oof_rankings, gt_dev, name=f"blend_{variant}/dev_oof" if variant != "blend" else "blend/dev_oof",
                       k=k, recall_ks=recall_ks).compute_ci(**ci_kw)
        rows.append({"retriever": "blend" if variant == "blend" else f"blend_{variant}", "split": "dev_oof", **res.to_row()})
        results[variant] = res
        logger.info(res.summary())
        return res

    oof_manual, fold_weights = oof_blend(fusion_cal, reranker_cal, gt, splits, k=k, depth=depth, grid=grid)
    _add_variant("blend", oof_manual)

    oof_lr, _ = oof_blend_ltr(src_cal, reranker_cal, gt, splits, names=names, l2=l2, use_ranks=use_ranks, depth=depth)
    _add_variant("lr", oof_lr)

    if featurizer is not None:
        oof_lr_graph, _ = oof_blend_ltr(src_cal, reranker_cal, gt, splits, names=names, l2=l2,
                                        use_ranks=use_ranks, depth=depth, featurizer=featurizer)
        _add_variant("lr_graph", oof_lr_graph)

    if mlp_cfg is not None and bool(mlp_cfg.enabled):
        oof_mlp, _ = oof_blend_ltr(src_cal, reranker_cal, gt, splits, names=names, l2=l2, use_ranks=use_ranks,
                                   depth=depth, featurizer=featurizer, head_factory=_make_mlp)
        _add_variant("mlp", oof_mlp)

    if method == "auto":
        chosen = max(results, key=lambda v: results[v].map_at_k)
        logger.info("ranking.method=auto: выбран %s (OOF MAP@10 %.4f)", chosen, results[chosen].map_at_k)
    else:
        # Явный method: lr при включённых граф-фичах означает lr_graph.
        chosen = {"lr": "lr_graph" if featurizer is not None else "lr"}.get(method, method)
    if chosen not in results:
        raise ValueError(f"ranking.method={method!r} требует включённого варианта {chosen!r} "
                         f"(есть: {sorted(results)}; проверьте ranking.mlp.enabled/graph_features.enabled)")
    res_primary = results[chosen]

    # Значимость: финал против OOF-гибрида (per-query AP из этапа 5) + попарно
    # между вариантами головы (вклад граф-фич и нелинейности по отдельности).
    significance = {}
    fusion_ap_path = stage_dir(cfg, "runs") / "fusion_oof_ap.json"
    if fusion_ap_path.exists():
        fusion_ap = {int(q): float(v) for q, v in read_json(fusion_ap_path).items()}
        common = sorted(set(res_primary.per_query_ap) & set(fusion_ap))
        p = paired_permutation_test(
            np.array([res_primary.per_query_ap[q] for q in common]),
            np.array([fusion_ap[q] for q in common]), seed=cfg.seed,
        )
        significance[f"{chosen}_vs_fusion"] = round(p, 4)
    for a, b in (("lr_graph", "lr"), ("mlp", "lr_graph"), ("mlp", "lr")):
        if a in results and b in results and not (a == "mlp" and b == "lr" and "lr_graph" in results):
            qs = sorted(set(results[a].per_query_ap) & set(results[b].per_query_ap))
            significance[f"{a}_vs_{b}"] = round(paired_permutation_test(
                np.array([results[a].per_query_ap[q] for q in qs]),
                np.array([results[b].per_query_ap[q] for q in qs]), seed=cfg.seed,
            ), 4)
    logger.info("значимость (permutation): %s", significance)

    # Финал: обучаем/подбираем выбранный вариант на всех dev; для теста граф
    # строится по всей dev-разметке (test-запросы своих меток не имеют).
    final_weight = None
    coefficients: dict = {}
    if chosen == "blend":
        final_weight = search_blend_weight(fusion_cal, reranker_cal, gt, splits.dev, k=k, grid=grid)
        final_cal = blend_matrix(fusion_cal, reranker_cal, weight=final_weight)
        final_test = blend_matrix(fusion_test, reranker_test, weight=final_weight)
        logger.info("итоговый вес блэнда (dev): fusion=%.2f reranker=%.2f", final_weight, 1 - final_weight)
    else:
        f_used = featurizer if chosen in ("lr_graph", "mlp") else None
        head = _make_mlp() if chosen == "mlp" else None
        model, feature_names = fit_blend_ltr(src_cal, reranker_cal, gt, splits.dev, names=names,
                                             l2=l2, use_ranks=use_ranks, featurizer=f_used, head=head)
        predict_kw = dict(names=names, use_ranks=use_ranks, featurizer=f_used, gt_train=gt_dev)
        final_cal = predict_blend_ltr(model, src_cal, reranker_cal, **predict_kw)
        final_test = predict_blend_ltr(model, src_test, reranker_test, **predict_kw)
        if chosen != "mlp":
            coefficients = model.coefficients(feature_names)
            logger.info("итоговые коэффициенты mini-LTR (dev): %s", coefficients)

    sd = StageDir(
        cfg, "scores", stage_dir(cfg, "scores", subdir="blend"), SCHEMA_BLEND,
        config_sections=("ranking", "reranker", "fusion", "retrievers", "preprocess"),
        input_hashes=_data_hashes(cfg, "articles", "calibration", "test"),
    )
    check_score_matrix(final_cal, fusion_cal.article_ids)
    check_score_matrix(final_test, fusion_test.article_ids)
    final_cal.save(sd.path / "calibration.npz")
    final_test.save(sd.path / "test.npz")
    sd.write_manifest(method=method, chosen=chosen, weight_fusion=final_weight, coefficients=coefficients,
                      graph_features=featurizer is not None,
                      oof_map=round(res_primary.map_at_k, 4))

    runs_dir = stage_dir(cfg, "runs")
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config_hash = cfg.hash_section("ranking", "reranker", "fusion", "retrievers", "preprocess")
    write_json(runs_dir / "blend_report.json",
               {"created_at": created_at, "config_hash": config_hash, "method": method, "chosen": chosen,
                "graph_features": featurizer is not None,
                "weight_fusion": final_weight, "coefficients": coefficients,
                "significance": significance, "rows": rows})
    _append_experiments(runs_dir / "experiments.csv", rows, config_hash, created_at)
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
