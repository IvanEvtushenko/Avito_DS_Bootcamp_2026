"""Текстовые энкодеры для dense-ретривера (внедряются как зависимость).

`DenseRetriever` знает только контракт `Encoder` (два метода: кодировать запросы и
кодировать пассажи в L2-нормированные векторы), но не реализацию. Это позволяет:

- в проде подставить `E5Encoder` (`intfloat/multilingual-e5-large` через
  `transformers`) — проверенную на русском retrieval-модель с префиксами
  `query:` / `passage:`;
- в тестах подставить `HashingStubEncoder` — детерминированный CPU-энкодер без
  скачивания модели (план §5.1.9: smoke-тест не тянет веса и не требует GPU).

E5 кодирует по инструкции модели: усреднение последнего слоя по маске внимания и
L2-нормализация; запрос и пассаж различаются только префиксом.
"""
from __future__ import annotations

import re
import zlib
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from ..logging_utils import get_logger

logger = get_logger("retrievers.encoders")


@runtime_checkable
class Encoder(Protocol):
    """Контракт энкодера: тексты → матрица L2-нормированных векторов [N, dim]."""

    dim: int

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray: ...

    def info(self) -> dict[str, object]: ...


class E5Encoder:
    """HF bi-encoder через transformers (пулинг + L2), дефолт — e5-large.

    Один класс покрывает разные retrieval-модели: пулинг (`mean` для e5, `cls`
    для BGE-семейства), архитектуру (`auto` для XLM-R, `t5_encoder` для FRIDA) и
    префиксы (`query:`/`passage:` у e5, `search_query:`/`search_document:` у
    FRIDA, пустые у BGE). Это делает сравнение эмбеддеров (§7.3) честным — у
    каждой модели её собственный протокол кодирования.

    Тяжёлые импорты (`torch`, `transformers`) и загрузка модели — лениво в
    конструкторе. По умолчанию fp16 на CUDA.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        *,
        device: str | None = None,
        max_seq_len: int = 512,
        batch_size: int = 64,
        use_fp16: bool = True,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        pooling: str = "mean",
        model_class: str = "auto",
    ) -> None:
        import torch
        from transformers import AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and self.device.startswith("cuda")
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.pooling = pooling
        self.model_class = model_class

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.use_fp16 else torch.float32
        if model_class == "t5_encoder":
            from transformers import T5EncoderModel

            self.model = T5EncoderModel.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()
        else:
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()
        cfg = self.model.config
        self.dim = int(getattr(cfg, "hidden_size", 0) or getattr(cfg, "d_model", 0))
        logger.info(
            "E5Encoder: model=%s pooling=%s arch=%s device=%s fp16=%s dim=%d max_len=%d",
            model_name, pooling, model_class, self.device, self.use_fp16, self.dim, max_seq_len,
        )

    def _pool(self, last_hidden, attention_mask):
        if self.pooling == "cls":
            return last_hidden[:, 0]  # BGE-семейство: первый токен
        # mean: усреднение по токенам с учётом маски (паддинг не влияет)
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        torch = self._torch
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [prefix + (t or "") for t in texts[start : start + self.batch_size]]
            enc = self.tokenizer(
                batch, max_length=self.max_seq_len, truncation=True, padding=True, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                # Только input_ids/attention_mask — совместимо и с XLM-R, и с T5.
                hidden = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
                emb = self._pool(hidden, enc["attention_mask"])
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out.append(emb.float().cpu().numpy())
        return np.vstack(out).astype(np.float32) if out else np.zeros((0, self.dim), dtype=np.float32)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(list(texts), self.query_prefix)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(list(texts), self.passage_prefix)

    def info(self) -> dict[str, object]:
        return {
            "type": "hf", "model": self.model_name, "pooling": self.pooling, "arch": self.model_class,
            "device": self.device, "fp16": self.use_fp16, "dim": self.dim, "max_seq_len": self.max_seq_len,
        }


class HashingStubEncoder:
    """Детерминированный энкодер для тестов: слова → хэшированный вектор + L2.

    Не модель — воспроизводит лексическую похожесть без GPU/сети: тексты с общими
    словами получают высокий косинус, поэтому релевантная статья всё равно
    всплывает наверх. Достаточно, чтобы проверить логику dense-ретривера (max по
    чанкам, вклад заголовка) и стыковку стадий.
    """

    _WORD_RE = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)

    def __init__(self, dim: int = 64, seed: int = 0) -> None:
        self.dim = dim
        self.seed = seed

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in self._WORD_RE.findall((text or "").lower()):
                # crc32 — детерминированный хэш (в отличие от встроенного hash()).
                j = zlib.crc32(word.encode("utf-8"), self.seed) % self.dim
                vecs[i, j] += 1.0
            norm = np.linalg.norm(vecs[i])
            if norm > 0:
                vecs[i] /= norm
        return vecs

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(list(texts))

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(list(texts))

    def info(self) -> dict[str, object]:
        return {"type": "stub", "dim": self.dim, "seed": self.seed}
