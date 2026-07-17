"""Cross-encoder reranker (этап 6): zero-shot инференс + fine-tune по фолдам."""
from __future__ import annotations

from .apply import (
    NON_CANDIDATE,
    build_passages,
    build_passages_from_qemb,
    candidate_lists,
    rerank_to_matrix,
)
from .backends import (
    JinaListwiseReranker,
    LlamaCppQwen3Reranker,
    Qwen3CausalLMReranker,
    build_reranker,
)
from .cross_encoder import CrossEncoderReranker, LexicalStubReranker, Reranker
from .negatives import TrainGroup, build_training_groups
from .train import fine_tune_reranker

__all__ = [
    "NON_CANDIDATE",
    "build_passages",
    "build_passages_from_qemb",
    "candidate_lists",
    "rerank_to_matrix",
    "build_reranker",
    "CrossEncoderReranker",
    "Qwen3CausalLMReranker",
    "JinaListwiseReranker",
    "LlamaCppQwen3Reranker",
    "LexicalStubReranker",
    "Reranker",
    "TrainGroup",
    "build_training_groups",
    "fine_tune_reranker",
]
