"""Fine-tune cross-encoder реранкера по фолдам (план §6, этап 6).

Обучение честное по фолдам (§4.2): модель фолда f учится на train(f) и предсказывает
val(f) → OOF-скор от модели, не видевшей разметку запроса. Контрастная функция
потерь на группу: логиты по [позитив, негатив₁, …, негативₙ] проходят softmax,
цель — позитив (индекс 0). Это стандартный listwise-лосс для реранкеров и
естественно устойчив к дисбалансу «1 позитив : N негативов».

Гиперпараметры (эпохи, LR, размер батча) — разумные дефолты, **не тюнятся** (§4.3).
Чекпоинт фолда выбирается не по val(f) (это привело бы к утечке в OOF), а просто
сохраняется финальная модель после фиксированного числа эпох.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

from ..logging_utils import get_logger
from .negatives import TrainGroup

logger = get_logger("rerank.train")

# Для детерминированного cuBLAS-matmul под use_deterministic_algorithms. Ставим до
# первой CUDA-операции (импорт train.py идёт раньше инициализации CUDA в пайплайне);
# иначе torch лишь предупредит и откатится на недетерминированный путь.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _set_seed(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Воспроизводимость обучения: детерминированные ядра там, где они есть.
    # warn_only=True — если у операции нет детерминированной реализации, torch
    # предупредит и продолжит, а не упадёт (безопасно для будущих правок пайплайна).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # автотюнинг conv-алгоритмов недетерминирован
    torch.use_deterministic_algorithms(True, warn_only=True)


def _encode_groups(tokenizer, groups: Sequence[TrainGroup], max_length: int, device):
    """Группы → тензоры пар (queries×passages), позитив каждой группы — первым."""
    flat_queries: list[str] = []
    flat_passages: list[str] = []
    for group in groups:
        flat_queries.append(group.query)
        flat_passages.append(group.positive)
        for negative in group.negatives:
            flat_queries.append(group.query)
            flat_passages.append(negative)
    enc = tokenizer(
        flat_queries, flat_passages, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    return {k: v.to(device) for k, v in enc.items()}


def _train_epoch(model, groups, optimizer, scaler, tokenizer, *, batch_groups, group_size, max_length, device, fp16, rng, scheduler=None):
    import torch

    model.train()
    order = rng.permutation(len(groups))
    total_loss, total_n = 0.0, 0
    for start in range(0, len(order), batch_groups):
        batch = [groups[i] for i in order[start : start + batch_groups]]
        enc = _encode_groups(tokenizer, batch, max_length, device)
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            logits = model(**enc).logits.view(len(batch), group_size)
            loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(batch), dtype=torch.long, device=device))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        total_loss += float(loss.item()) * len(batch)
        total_n += len(batch)
    return total_loss / max(total_n, 1)


def _build_scheduler(optimizer, *, schedule: str, warmup_ratio: float, total_steps: int):
    """LR-шедулер: linear warmup → косинусное затухание (стандарт FT реранкеров).

    `schedule="constant"` → None (текущее поведение без изменений). warmup_ratio —
    доля шагов на разогрев (обычно 0.1): маленький LR в начале защищает
    предобученные веса от разноса на первых шумных батчах.
    """
    import math

    import torch

    if schedule == "constant":
        return None
    if schedule != "cosine":
        raise ValueError(f"неизвестный schedule={schedule!r}; допустимо: constant | cosine")
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def fine_tune_reranker(
    base_model_name: str,
    train_groups: Sequence[TrainGroup],
    output_dir: str | Path,
    *,
    epochs: int = 2,
    lr: float = 2.0e-5,
    batch_groups: int = 4,
    max_length: int = 384,
    use_fp16: bool = True,
    freeze_embeddings: bool = True,
    schedule: str = "constant",
    warmup_ratio: float = 0.0,
    device: str | None = None,
    seed: int = 42,
) -> Path:
    """Обучить реранкер на группах и сохранить финальную модель в output_dir."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not train_groups:
        raise ValueError("нет обучающих групп")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = use_fp16 and device.startswith("cuda")
    group_size = 1 + len(train_groups[0].negatives)
    _set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name).to(device)
    if freeze_embeddings:
        # Матрица эмбеддингов (~256M из 560M параметров) — крупнейший источник памяти
        # оптимизатора: её fp32-состояния Adam и временный буфер .sqrt() (~1 ГБ) и дают
        # OOM на 12 ГБ. Для cross-encoder реранкера входные эмбеддинги можно не трогать.
        for p in model.get_input_embeddings().parameters():
            p.requires_grad_(False)
        model.enable_input_require_grads()  # иначе grad-checkpointing без обучаемого входа даёт ошибку
    model.gradient_checkpointing_enable()  # экономия памяти на 12 ГБ
    # foreach=False обязателен для 12 ГБ: multi-tensor AdamW на .step() аллоцирует
    # временные буферы размером со все состояния поверх постоянных весов/градиентов → OOM.
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, foreach=False)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    rng = np.random.default_rng(seed)
    steps_per_epoch = (len(train_groups) + batch_groups - 1) // batch_groups
    scheduler = _build_scheduler(
        optimizer, schedule=schedule, warmup_ratio=warmup_ratio, total_steps=max(1, epochs * steps_per_epoch)
    )

    for epoch in range(epochs):
        loss = _train_epoch(
            model, train_groups, optimizer, scaler, tokenizer,
            batch_groups=batch_groups, group_size=group_size, max_length=max_length,
            device=device, fp16=fp16, rng=rng, scheduler=scheduler,
        )
        cur_lr = optimizer.param_groups[0]["lr"]
        logger.info("fine-tune epoch %d/%d: train_loss=%.4f lr=%.2e (groups=%d, schedule=%s)",
                    epoch + 1, epochs, loss, cur_lr, len(train_groups), schedule)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
