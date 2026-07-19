# Avito Help-Article Retrieval (DS Bootcamp 2026)

Retrieval-этап RAG: по вопросу пользователя вернуть ранжированный топ-10 статей
справки Авито. Метрика — **MAP@10**. Весь код — в [`solution/`](solution/).

## Как читать историю

Линейная история на `master` (Conventional Commits). Вехи помечены тегами:

- **`v1-baseline`** — полный рабочий пайплайн: препроцессинг (HTML→текст, чанкинг)
  → лексика (BM25 + char-TF-IDF) → dense-ретривер (multilingual-e5) → hybrid
  fusion → cross-encoder реранкер `BAAI/bge-reranker-v2-m3` → blend.
- **`v2-comparison`** — learning-to-rank фьюжн; сравнение эмбеддеров
  (e5 / FRIDA / bge-m3 / user-bge-m3) и сменных реранкеров
  (jina-v3, qwen3-0.6b/4b/8b, bge) через `compare_rerankers.py`.
- **`v3-finetune`** — дообучение bge-реранкера на labeled + synthetic парах,
  мульти-чанковый скоринг, holdout-диагностика. Итог — **MAP@10 ≈ 0.69**.

`git checkout v1-baseline` (и т.д.) — посмотреть проект на конкретной вехе.

## Запуск

Данные, окружение и `make`-цели — см. [`solution/README.md`](solution/README.md).
Тяжёлые артефакты (индексы, эмбеддинги, веса) в git не хранятся — они
детерминированно пересобираются из данных и конфигов.
