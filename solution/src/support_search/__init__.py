"""support_search — retrieval-этап RAG-пайплайна поддержки Авито.

По тексту вопроса пользователя возвращает ранжированный топ-10 статей справки.
Метрика — MAP@10. Пакет собран по принципу «одна папка = одна ступень
пайплайна» с однонаправленными зависимостями:

    data -> preprocess -> retrievers -> fusion -> rerank -> ranking -> export

Сквозные утилиты (`contracts`, `eval`, `config`) не зависят от ступеней и от
них можно зависеть всем.
"""
from __future__ import annotations

__version__ = "0.1.0"
