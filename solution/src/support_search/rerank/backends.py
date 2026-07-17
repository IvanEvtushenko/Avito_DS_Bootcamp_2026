"""Backend-и реранкера и фабрика выбора по `reranker.backend` (§6, сравнение моделей).

Все реранкеры внедряются по одному контракту `Reranker` (`cross_encoder`):
параллельные списки (query, passage) → скоры. Так же, как энкодер выбирает
архитектуру по `retrievers.dense.model_class`, реранкер выбирает семейство модели
по `reranker.backend`. Дефолтный backend (`sequence_classification`) — текущий
`CrossEncoderReranker` (bge), поэтому основной прогон не меняется, а ablation-конфиги
Qwen3/Jina/GGUF подключают свой backend одной строкой.

Листовые модели (jina) дополнительно реализуют `score_listwise`: один запрос и
весь его список кандидатов оцениваются совместно. `apply.rerank_to_matrix` сам
выбирает листовой путь, если у реранкера есть этот метод.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ..logging_utils import get_logger
from .cross_encoder import CrossEncoderReranker, Reranker

logger = get_logger("rerank.backends")

# Инструкция для instruction-aware реранкеров (Qwen3) по умолчанию — из карточки модели.
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

# Системный prefix/suffix chat-шаблона Qwen3-Reranker (yes/no протокол из карточки модели).
_QWEN3_PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    '<|im_start|>user\n'
)
_QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _resolve_gguf(repo_id: str, filename: str, revision: str | None) -> str:
    """Локальный путь к GGUF-файлу, устойчивый к offline-режиму.

    Сначала пробуем `hf_hub_download` (уважает HF_HUB_OFFLINE). Если не вышло —
    ищем файл прямо в кэше `models--<repo>/snapshots/<revision>/`, т.к. закреплённый
    revision — это готовый снапшот и HEAD-запрос к hub не нужен.
    """
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, local_files_only=True)
    except Exception:
        import glob
        from huggingface_hub.constants import HF_HUB_CACHE

        stem = "models--" + repo_id.replace("/", "--")
        rev = revision or "*"
        hits = glob.glob(str(Path(HF_HUB_CACHE) / stem / "snapshots" / rev / filename))
        if not hits:
            raise FileNotFoundError(f"GGUF не найден в кэше: {repo_id}:{filename} (revision={revision})")
        return hits[0]


class Qwen3CausalLMReranker:
    """Qwen3-Reranker (0.6B/4B): pairwise-скор через логиты токенов «yes»/«no».

    Модель — causal LM, а не sequence-classification head. Пара (query, passage)
    оборачивается в её chat-шаблон (<Instruct>/<Query>/<Document>), логит последней
    позиции сравнивается для токенов «yes» и «no»; скор = P(yes) после log_softmax
    по этим двум токенам. Рецепт и prefix/suffix взяты из карточки модели;
    инструкция настраивается через `reranker.instruction`.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-Reranker-0.6B",
        *,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 16,
        use_fp16: bool = True,
        instruction: str | None = None,
        revision: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_name = str(model_name_or_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and self.device.startswith("cuda")
        self.instruction = instruction or DEFAULT_INSTRUCTION

        # padding_side=left: у causal LM «последняя позиция» должна быть реальным
        # последним токеном, поэтому паддинг идёт слева.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=revision, padding_side="left")
        dtype = torch.float16 if self.use_fp16 else torch.float32
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name_or_path, revision=revision, dtype=dtype)
            .to(self.device)
            .eval()
        )
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix_tokens = self.tokenizer.encode(_QWEN3_PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(_QWEN3_SUFFIX, add_special_tokens=False)
        logger.info(
            "Qwen3CausalLMReranker: model=%s device=%s fp16=%s max_len=%d",
            self.model_name, self.device, self.use_fp16, max_length,
        )

    def _format(self, query: str, passage: str) -> str:
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {passage}"

    def score_pairs(self, queries: Sequence[str], passages: Sequence[str]) -> np.ndarray:
        """P(yes) релевантности для параллельных списков (queries[i], passages[i])."""
        torch = self._torch
        if len(queries) != len(passages):
            raise ValueError("queries и passages должны быть одной длины")
        budget = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        out: list[np.ndarray] = []
        for start in range(0, len(queries), self.batch_size):
            texts = [
                self._format(q, p)
                for q, p in zip(queries[start : start + self.batch_size], passages[start : start + self.batch_size])
            ]
            enc = self.tokenizer(
                texts, padding=False, truncation="longest_first",
                return_attention_mask=False, max_length=budget,
            )
            for i, ids in enumerate(enc["input_ids"]):
                enc["input_ids"][i] = self.prefix_tokens + ids + self.suffix_tokens
            enc = self.tokenizer.pad(enc, padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :]  # логиты последней позиции
                stacked = torch.stack([logits[:, self.token_false_id], logits[:, self.token_true_id]], dim=1)
                probs = torch.nn.functional.log_softmax(stacked, dim=1)[:, 1].exp()
            out.append(probs.float().cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros(0, dtype=np.float32)

    def info(self) -> dict[str, object]:
        return {"type": "qwen3_causal_lm", "model": self.model_name, "device": self.device,
                "fp16": self.use_fp16, "max_length": self.max_length, "instruction": self.instruction}


class JinaListwiseReranker:
    """jina-reranker-v3: листовой реранкер (один запрос + все кандидаты вместе).

    Использует собственный метод `model.rerank(query, documents)` (trust_remote_code),
    возвращающий relevance_score на документ. Через `score_listwise` реранкер видит
    весь список кандидатов запроса сразу (в отличие от pairwise-моделей), что и есть
    смысл listwise. `apply.rerank_to_matrix` вызывает `score_listwise` по-запросно.
    """

    def __init__(
        self,
        model_name_or_path: str = "jinaai/jina-reranker-v3",
        *,
        device: str | None = None,
        use_fp16: bool = True,
        max_doc_length: int = 2048,
        max_query_length: int = 512,
        revision: str | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModel

        self._torch = torch
        self.model_name = str(model_name_or_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device.startswith("cuda")
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length
        dtype = torch.float16 if self.use_fp16 else torch.float32
        # jina сам перезагружает свой токенизатор по model.name_or_path без revision;
        # в offline (HF_HUB_OFFLINE=1) «main» не резолвится, поэтому грузим из локального
        # каталога снапшота — тогда внутренняя дозагрузка тоже находит файлы офлайн.
        source, src_revision = str(model_name_or_path), revision
        try:
            from huggingface_hub import snapshot_download

            source = snapshot_download(str(model_name_or_path), revision=revision, local_files_only=True)
            src_revision = None  # revision уже вшит в путь снапшота
        except Exception:  # нет в кэше / онлайн-режим — оставляем repo id и revision
            pass
        self.model = (
            AutoModel.from_pretrained(source, revision=src_revision, trust_remote_code=trust_remote_code, dtype=dtype)
            .to(self.device)
            .eval()
        )
        logger.info("JinaListwiseReranker: model=%s device=%s fp16=%s", self.model_name, self.device, self.use_fp16)

    def score_listwise(self, query: str, passages: Sequence[str]) -> np.ndarray:
        """Скоры релевантности для всех кандидатов одного запроса (листовой контекст)."""
        passages = list(passages)
        if not passages:
            return np.zeros(0, dtype=np.float32)
        with self._torch.no_grad():
            results = self.model.rerank(
                query, passages, top_n=None,
                max_doc_length=self.max_doc_length, max_query_length=self.max_query_length,
            )
        scores = np.zeros(len(passages), dtype=np.float32)
        for r in results:  # порядок результатов — по убыванию скора; раскладываем по index
            scores[int(r["index"])] = float(r["relevance_score"])
        return scores

    def score_pairs(self, queries: Sequence[str], passages: Sequence[str]) -> np.ndarray:
        """Фолбэк на pairwise (каждая пара — список из одного документа). Основной путь —
        `score_listwise` (см. `apply.rerank_to_matrix`)."""
        return np.array(
            [self.score_listwise(q, [p])[0] for q, p in zip(queries, passages)], dtype=np.float32
        )

    def info(self) -> dict[str, object]:
        return {"type": "jina_listwise", "model": self.model_name, "device": self.device, "fp16": self.use_fp16}


class LlamaCppQwen3Reranker:
    """Qwen3-Reranker-8B в квантованном GGUF через llama.cpp (yes/no логиты).

    Тот же yes/no протокол, что и `Qwen3CausalLMReranker`, но инференс через
    `llama-cpp-python` по GGUF-файлу — так 8B-модель влезает в 12 ГБ VRAM в Q4_K_M.
    Пакет `llama-cpp-python` в окружении может отсутствовать: тогда конструктор
    падает с понятной инструкцией по установке (остальные backend-ы не затронуты).
    """

    def __init__(
        self,
        model_name_or_path: str = "QuantFactory/Qwen3-Reranker-8B-GGUF",
        *,
        model_file: str = "Qwen3-Reranker-8B.Q4_K_M.gguf",
        max_length: int = 512,
        instruction: str | None = None,
        revision: str | None = None,
        n_gpu_layers: int = -1,
        **_ignored: object,  # напр. base_model_name — llama.cpp токенизирует сам, HF-токенизатор не нужен
    ) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # пакет намеренно не тянется автоматически (сборка под CUDA)
            raise ImportError(
                "backend=llama_cpp_qwen3 требует llama-cpp-python. GGUF-веса скачаны, но пакет не установлен. "
                "Установите: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
            ) from exc

        self.model_name = str(model_name_or_path)
        self.model_file = str(model_file)
        self.max_length = max_length
        self.instruction = instruction or DEFAULT_INSTRUCTION
        gguf_path = _resolve_gguf(self.model_name, self.model_file, revision)
        # logits_all=True — чтобы читать логиты последней позиции промпта (не только генерации).
        self.llm = Llama(model_path=gguf_path, n_ctx=max_length, n_gpu_layers=n_gpu_layers, logits_all=True, verbose=False)
        # id токенов «yes»/«no» в словаре GGUF (по одному токену без BOS).
        self.token_true_id = self.llm.tokenize(b"yes", add_bos=False)[-1]
        self.token_false_id = self.llm.tokenize(b"no", add_bos=False)[-1]
        logger.info("LlamaCppQwen3Reranker: model=%s file=%s n_ctx=%d", self.model_name, self.model_file, max_length)

    def _prompt(self, query: str, passage: str) -> str:
        body = f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {passage}"
        return _QWEN3_PREFIX + body + _QWEN3_SUFFIX

    def score_pairs(self, queries: Sequence[str], passages: Sequence[str]) -> np.ndarray:
        import math

        if len(queries) != len(passages):
            raise ValueError("queries и passages должны быть одной длины")
        scores = np.zeros(len(queries), dtype=np.float32)
        for i, (q, p) in enumerate(zip(queries, passages)):
            tokens = self.llm.tokenize(self._prompt(q, p).encode("utf-8"))[: self.max_length]
            self.llm.reset()
            self.llm.eval(tokens)
            logits = self.llm.scores[len(tokens) - 1]  # логиты последней позиции
            lt, lf = float(logits[self.token_true_id]), float(logits[self.token_false_id])
            m = max(lt, lf)
            scores[i] = math.exp(lt - m) / (math.exp(lt - m) + math.exp(lf - m))  # P(yes) по двум токенам
        return scores

    def info(self) -> dict[str, object]:
        return {"type": "llama_cpp_qwen3", "model": self.model_name, "model_file": self.model_file,
                "max_length": self.max_length, "instruction": self.instruction}


_BACKENDS = ("sequence_classification", "qwen3_causal_lm", "jina_listwise", "llama_cpp_qwen3")


def build_reranker(section) -> Reranker:
    """Собрать реранкер по секции `reranker` конфига, выбирая семейство по `backend`.

    Единая точка выбора модели: основной прогон (bge, дефолтный backend) не меняется,
    а Qwen3/Jina/GGUF подключаются переопределением `reranker.backend` в ablation-конфиге.
    """
    backend = str(section.get("backend", "sequence_classification"))
    model_name = str(section.model_name)
    device = section.get("device", None)
    device = str(device) if device else None
    revision = section.get("revision", None)
    revision = str(revision) if revision else None
    instruction = section.get("instruction", None)
    instruction = str(instruction) if instruction else None
    max_length = int(section.get("max_length", 512))
    batch_size = int(section.get("batch_size", 64))
    use_fp16 = bool(section.get("use_fp16", True))

    if backend == "sequence_classification":
        return CrossEncoderReranker(
            model_name, device=device, max_length=max_length, batch_size=batch_size,
            use_fp16=use_fp16, revision=revision,
        )
    if backend == "qwen3_causal_lm":
        return Qwen3CausalLMReranker(
            model_name, device=device, max_length=max_length, batch_size=batch_size,
            use_fp16=use_fp16, instruction=instruction, revision=revision,
        )
    if backend == "jina_listwise":
        return JinaListwiseReranker(
            model_name, device=device, use_fp16=use_fp16, revision=revision,
            trust_remote_code=bool(section.get("trust_remote_code", True)),
            max_doc_length=int(section.get("max_doc_length", 2048)),
            max_query_length=int(section.get("max_query_length", 512)),
        )
    if backend == "llama_cpp_qwen3":
        return LlamaCppQwen3Reranker(
            model_name, model_file=str(section.get("model_file", "")),
            base_model_name=str(section.get("base_model_name", "Qwen/Qwen3-Reranker-8B")),
            max_length=max_length, instruction=instruction, revision=revision,
        )
    raise ValueError(f"неизвестный reranker.backend={backend!r}; допустимо: {list(_BACKENDS)}")
