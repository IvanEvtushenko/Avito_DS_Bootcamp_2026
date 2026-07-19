#!/usr/bin/env python3
"""Ночной свип дообучения реранкера + рефит на всех 500 (план §4.5).

Три фазы, каждая пишет в отдельный artifacts_dir; скрипт возобновляемый —
готовая фаза (есть runs/blend_report.json) пропускается:

  1. FT top_k=30 и top_k=50 на dev-400 (5 фолдов, честный OOF) → выбор top_k;
  2. для победившего top_k — epochs=1 и epochs=3 (epochs=2 уже есть) → выбор epochs;
  3. рефит при лучшем (top_k, epochs) на ВСЕХ 500 (folds.holdout_frac=0) → answer.csv.

Отбор (top_k, epochs) — ТОЛЬКО по dev-400 blend OOF; holdout при этом не трогается.
Рефит на 500 задействует разметку holdout в ОБУЧЕНИИ (более полные test-модели,
план §4.5) — это НЕ трата бюджета holdout. Подтверждение победителя на holdout
(одно обращение, бюджет ≤3) — отдельный шаг holdout_check.py ПОСЛЕ (см. README).

Каждая фаза — это `cli all` в отдельном процессе (изолированный CUDA-контекст,
без накопления фрагментации памяти за ночь). Прогресс — в stdout и в
artifacts_rr_ft_final/sweep_summary.json.

  cd solution && PYTHONPATH=src HF_HUB_OFFLINE=1 python3 refit_sweep.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SOL = Path(__file__).resolve().parent
ENV = {
    **os.environ,
    "PYTHONPATH": "src",
    "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
    "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
}
FINAL_DIR = "artifacts_rr_ft_final"


def _log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}", flush=True)


def _dev_oof(art_dir: str) -> float:
    """dev-OOF MAP@10 выбранной головы blend из runs/blend_report.json."""
    report = json.loads((SOL / art_dir / "runs" / "blend_report.json").read_text())
    chosen = report.get("chosen", "lr")
    retr = "blend" if chosen == "blend" else f"blend_{chosen}"
    for row in report["rows"]:
        if row.get("retriever") == retr and row.get("split") == "dev_oof":
            return float(row["map@10"])
    raise KeyError(f"{art_dir}: не нашёл строку {retr}/dev_oof в blend_report")


def _prune_models(art_dir: str) -> None:
    """Удалить 5 фолд-моделей прогона (~11 ГБ). Отчёты/скоры/answer.csv остаются —
    OOF уже прочитан из blend_report, test-предсказания уже в scores/blend. Модели
    нужны только dev-400-победителю (для holdout-подтверждения) — его не трогаем."""
    p = SOL / art_dir / "models"
    if p.exists():
        shutil.rmtree(p)
        _log(f"[clean] {art_dir}/models удалены")


def run_phase(experiment: str, art_dir: str, overrides: list[str] | None = None) -> float:
    """`cli all` для конфига в свой artifacts_dir; возвращает dev-OOF MAP@10.

    Возобновляемость: если blend_report.json уже есть — фаза пропускается.
    """
    report = SOL / art_dir / "runs" / "blend_report.json"
    if report.exists():
        oof = _dev_oof(art_dir)
        _log(f"[skip] {art_dir}: уже посчитан (dev OOF={oof:.4f})")
        return oof
    cmd = [sys.executable, "-m", "support_search.cli", "all", "--experiment", experiment]
    if overrides:
        cmd += ["--set", *overrides]
    _log(f"[run ] {art_dir}: {' '.join(cmd[4:])}")
    subprocess.run(cmd, check=True, cwd=SOL, env=ENV)
    oof = _dev_oof(art_dir)
    _log(f"[done] {art_dir}: dev OOF={oof:.4f}")
    return oof


def main() -> int:
    summary: dict = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "phases": {}}

    # Держим модели только текущего лучшего dev-400-прогона (для holdout-подтверждения);
    # у остальных — удаляем сразу после чтения OOF. Пик по диску ≈ 2 прогона (~22 ГБ).
    best = {"dir": None, "oof": -1.0}

    def consider(art_dir: str, oof: float) -> None:
        if oof > best["oof"]:
            if best["dir"] is not None:
                _prune_models(best["dir"])
            best["dir"], best["oof"] = art_dir, oof
        else:
            _prune_models(art_dir)

    # ── Фаза 1: top_k=30 vs 50 (dev-400, epochs=2) ──────────────────────────
    oof = {}
    for tk, exp in ((30, "rerank_ft_k30.yaml"), (50, "rerank_ft_k50.yaml")):
        d = f"artifacts_rr_ft_k{tk}"
        oof[tk] = run_phase(exp, d)
        consider(d, oof[tk])
    top_k = max(oof, key=lambda k: (oof[k], -k))   # тай-брейк: меньший top_k (дешевле)
    summary["phases"]["top_k"] = {"oof": oof, "winner": top_k}
    _log(f"=== top_k: 30→{oof[30]:.4f}  50→{oof[50]:.4f}  → выбран {top_k} ===")

    # ── Фаза 2: epochs 1 / 2 / 3 при победившем top_k (dev-400) ──────────────
    base_exp = f"rerank_ft_k{top_k}.yaml"
    ep_oof = {2: oof[top_k]}                        # epochs=2 = базовый прогон фазы 1
    for ep in (1, 3):
        d = f"artifacts_rr_ft_k{top_k}_ep{ep}"
        ep_oof[ep] = run_phase(base_exp, d, [f"artifacts_dir={d}", f"reranker.fine_tune.epochs={ep}"])
        consider(d, ep_oof[ep])
    # best["dir"] = сохранённый победитель; epochs выводим из его имени (согласовано с consider,
    # где на равенстве OOF остаётся ранее встреченный = epochs=2, валидированный дефолт).
    win_dir = best["dir"]
    best_ep = 2 if win_dir == f"artifacts_rr_ft_k{top_k}" else int(win_dir.rsplit("_ep", 1)[1])
    summary["phases"]["epochs"] = {"oof": ep_oof, "winner": best_ep, "winner_dir": win_dir}
    _log(f"=== epochs (top_k={top_k}): " + " ".join(f"{e}→{ep_oof[e]:.4f}" for e in sorted(ep_oof))
         + f"  → выбран {best_ep} (модели в {win_dir}) ===")

    # ── Фаза 3: рефит при (top_k, best_ep) на всех 500 → сабмит ──────────────
    # resume_folds: полные фолд-модели от убитого прогона переиспользуются,
    # обучаются только недостающие (детерминизм сохраняется: те же seed/фолды).
    refit_oof = run_phase(base_exp, FINAL_DIR, [
        f"artifacts_dir={FINAL_DIR}", "folds.holdout_frac=0.0",
        f"reranker.fine_tune.epochs={best_ep}", "reranker.fine_tune.resume_folds=true",
    ])
    _prune_models(FINAL_DIR)   # рефит-модели видели holdout → для его оценки бесполезны; answer.csv остаётся
    answer = SOL / FINAL_DIR / "answers" / "answer.csv"
    summary["phases"]["refit_500"] = {
        "artifacts_dir": FINAL_DIR, "top_k": top_k, "epochs": best_ep,
        "oof_500": refit_oof, "answer": str(answer),
    }
    summary["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (SOL / FINAL_DIR / "sweep_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    _log("================= ИТОГ =================")
    _log(f"top_k={top_k}, epochs={best_ep}; dev-400 OOF={ep_oof[best_ep]:.4f}; OOF-500(рефит)={refit_oof:.4f}")
    _log(f"Сабмит: {answer}")
    _log(f"dev-400 модели победителя сохранены в {win_dir} (для holdout-подтверждения)")
    _log("Подтверждение на holdout (plain blend, отдельное обращение к бюджету):")
    _log(f"  PYTHONPATH=src HF_HUB_OFFLINE=1 python3 holdout_check.py ft_dir {win_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
