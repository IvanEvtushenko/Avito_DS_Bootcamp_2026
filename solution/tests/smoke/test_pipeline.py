"""Smoke-тест всего пайплайна на игрушечных данных (план §5.1.9).

Прогоняет preprocess → make-folds → retrieve → evaluate → make-answer →
validate-answer на mini-корпусе. Без GPU и без скачивания моделей: лексические
ретриверы работают на CPU. Проверяет, что стадии стыкуются, answer.csv проходит
контракт и повторный прогон детерминирован.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

_SOL = pathlib.Path(__file__).resolve().parents[2]
for _p in (_SOL / "src", _SOL / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mini_data import write_mini_dataset  # noqa: E402
from support_search.config import load_config  # noqa: E402
from support_search.data.io import read_json  # noqa: E402
from support_search.pipeline import (  # noqa: E402
    stage_evaluate,
    stage_fuse,
    stage_make_answer,
    stage_make_folds,
    stage_preprocess,
    stage_retrieve,
    stage_validate_answer,
)


class TestPipelineSmoke(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(self._tmp.name)
        self.data_dir = root / "data"
        self.artifacts_dir = root / "artifacts"
        write_mini_dataset(self.data_dir)
        self.cfg = load_config(
            _SOL / "configs" / "default.yaml",
            overrides=[
                f"data.dir={self.data_dir}",
                f"artifacts_dir={self.artifacts_dir}",
                "retrievers.char_tfidf.min_df=1",   # min_df=2 отсечёт почти всё на 12 статьях
                "retrievers.dense.enabled=false",   # smoke без модели/GPU: fusion над лексикой
                "reranker.enabled=false",           # реранкеру нужен dense — выключен
                "export.primary_source=fusion",
                "fusion.random_search.n_samples=50",
                "evaluation.bootstrap_resamples=200",
                "folds.holdout_frac=0.2",
            ],
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_end_to_end(self):
        stage_preprocess(self.cfg)
        self.assertTrue((self.artifacts_dir / "corpus" / "corpus.parquet").exists())

        stage_make_folds(self.cfg)
        folds = read_json(self.artifacts_dir / "splits" / "folds.json")
        self.assertEqual(len(folds["holdout"]) + len(folds["dev"]), 10)

        stage_retrieve(self.cfg)
        self.assertTrue((self.artifacts_dir / "scores" / "bm25" / "test.npz").exists())
        self.assertTrue((self.artifacts_dir / "scores" / "char_tfidf" / "calibration.npz").exists())

        stage_evaluate(self.cfg)
        report = read_json(self.artifacts_dir / "runs" / "eval_report.json")
        self.assertTrue(len(report["rows"]) > 0)
        # На игрушечном корпусе BM25 должен быть заметно лучше случайного.
        dev_bm25 = [r for r in report["rows"] if r["retriever"] == "bm25" and r["split"] == "dev"]
        self.assertTrue(dev_bm25 and dev_bm25[0]["map@10"] > 0.5)

        # Слияние bm25 + char_tfidf (dense выключен) → матрица fusion + кандидаты.
        stage_fuse(self.cfg)
        self.assertTrue((self.artifacts_dir / "scores" / "fusion" / "test.npz").exists())
        self.assertTrue((self.artifacts_dir / "candidates" / "test.parquet").exists())
        fusion_report = read_json(self.artifacts_dir / "runs" / "fusion_report.json")
        self.assertTrue(any(r["retriever"] == "fusion_weighted" for r in fusion_report["rows"]))

        answer_path = stage_make_answer(self.cfg)  # primary_source=fusion
        content_first = answer_path.read_text(encoding="utf-8")

        # validate-answer не должен бросать исключение.
        stage_validate_answer(self.cfg)

        # Детерминизм: повторный make-answer даёт идентичный файл.
        stage_make_answer(self.cfg)
        self.assertEqual(answer_path.read_text(encoding="utf-8"), content_first)

    def test_retrieve_cache_is_reused(self):
        stage_preprocess(self.cfg)
        stage_make_folds(self.cfg)
        stage_retrieve(self.cfg)
        manifest = read_json(self.artifacts_dir / "scores" / "bm25" / "manifest.json")
        # Повторный вызов не должен падать и должен видеть свежий кэш.
        stage_retrieve(self.cfg)
        manifest_again = read_json(self.artifacts_dir / "scores" / "bm25" / "manifest.json")
        self.assertEqual(manifest["config_hash"], manifest_again["config_hash"])


if __name__ == "__main__":
    unittest.main()
