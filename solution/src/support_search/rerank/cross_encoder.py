"""Cross-encoder reranker (`BAAI/bge-reranker-v2-m3`) — точная переоценка топ-50.

В отличие от bi-encoder, cross-encoder видит запрос и статью одновременно
(attention между их токенами), поэтому точнее, но медленнее и не предвычисляет
векторы. Его роль (план §6): не искать по всему корпусу, а переставить уже
найденные ~50 кандидатов, поднимая релевантные в топ-10.

Модель выдаёт один логит на пару (запрос, пассаж) — чем выше, тем релевантнее.
Загружается через `transformers` (пакета FlagEmbedding в окружении нет). Как и
энкодер, реранкер внедряется по контракту `Reranker`: в тестах — лексическая
заглушка без GPU/сети.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from ..logging_utils import get_logger

logger = get_logger("rerank.cross_encoder")


@runtime_checkable
class Reranker(Protocol):
    """Контракт реранкера: параллельные списки запросов и пассажей → скоры [N]."""

    def score_pairs(self, queries: Sequence[str], passages: Sequence[str]) -> np.ndarray: ...

    def info(self) -> dict[str, object]: ...


class CrossEncoderReranker:
    """bge-reranker-v2-m3 через AutoModelForSequenceClassification (1 логит на пару)."""

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-reranker-v2-m3",
        *,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 64,
        use_fp16: bool = True,
        revision: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.model_name = str(model_name_or_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and self.device.startswith("cuda")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=revision)
        dtype = torch.float16 if self.use_fp16 else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, revision=revision, torch_dtype=dtype
        ).to(self.device).eval()
        logger.info(
            "CrossEncoderReranker: model=%s device=%s fp16=%s max_len=%d",
            self.model_name, self.device, self.use_fp16, max_length,
        )

    def score_pairs(self, queries: Sequence[str], passages: Sequence[str]) -> np.ndarray:
        """Логиты релевантности для параллельных списков (queries[i], passages[i])."""
        torch = self._torch
        if len(queries) != len(passages):
            raise ValueError("queries и passages должны быть одной длины")
        out: list[np.ndarray] = []
        for start in range(0, len(queries), self.batch_size):
            q = list(queries[start : start + self.batch_size])
            p = list(passages[start : start + self.batch_size])
            enc = self.tokenizer(
                q, p, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits.view(-1).float()
            out.append(logits.cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros(0, dtype=np.float32)

    def info(self) -> dict[str, object]:
        return {"type": "cross_encoder", "model": self.model_name, "device": self.device,
                "fp16": self.use_fp16, "max_length": self.max_length}


class LexicalStubReranker:
    """Заглушка для тестов: скор пары = доля слов запроса, встреченных в пассаже.

    Не модель — детерминированная лексическая близость на CPU. Достаточно, чтобы
    проверить оркестрацию реранка/бленда без скачивания cross-encoder.
    """

    import re as _re
    _WORD_RE = _re.compile(r"[0-9a-zа-яё]+", _re.IGNORECASE)

    def score_pairs(self, queries: Sequence[str], passages: Sequence[str]) -> np.ndarray:
        scores = np.zeros(len(queries), dtype=np.float32)
        for i, (q, p) in enumerate(zip(queries, passages)):
            qw = set(self._WORD_RE.findall(q.lower()))
            pw = set(self._WORD_RE.findall(p.lower()))
            scores[i] = len(qw & pw) / len(qw) if qw else 0.0
        return scores

    def info(self) -> dict[str, object]:
        return {"type": "stub"}
