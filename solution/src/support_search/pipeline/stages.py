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
from ..logging_utils import get_logger
from ..preprocess import Tokenizer
from ..retrievers import RETRIEVER_REGISTRY, BM25Retriever, Corpus, build_corpus
from .artifacts import StageDir, hash_file, stage_dir

logger = get_logger("pipeline.stages")

# Версии схем артефактов — рост версии инвалидирует старый кэш (§5.1.6).
SCHEMA_CORPUS = 1
SCHEMA_SPLITS = 1
SCHEMA_SCORES = 1


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
    sd.write_manifest(
        n_articles=len(corpus),
        empty_text_frac=round(empty_frac, 4),
        text_len_median=int(np.median(lengths)),
        text_len_max=int(lengths.max()),
    )
    logger.info(
        "preprocess: %d статей, пустой текст %.1f%%, медиана %d симв., макс %d симв.",
        len(corpus), 100 * empty_frac, int(np.median(lengths)), int(lengths.max()),
    )
    return sd.path


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
