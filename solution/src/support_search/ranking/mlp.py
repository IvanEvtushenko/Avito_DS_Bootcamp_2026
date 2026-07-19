"""Listwise MLP-голова финального ранжирования — нелинейная замена mini-LTR.

Мотивация (диагностика + граф-фичи): полезность связи «кандидат — пара к
топ-1» зависит от уверенности реранкера в самом топ-1 — это взаимодействие
признаков, которое линейная LR не выражает. Двух-трёхслойный MLP на тех же
признаках кандидата такие взаимодействия выучивает, оставаясь крошечным
(~10³ параметров на ~16К обучающих строк фолда).

Отличия от pointwise LR (`support_search.ltr`):

- **listwise softmax-CE по запросу** (как в fine-tune реранкера): softmax по
  кандидатам запроса, лосс = −mean log P(позитив). Оптимизируется порядок
  внутри запроса — то, что и меряет MAP@10, — а дисбаланс классов исчезает
  по построению (не нужен class_weight).
- Полный батч (все запросы сразу): фиксированное число эпох, без выбора
  чекпоинта по валидации (§4.2 — как у fine-tune: иначе утечка в OOF).

Гиперпараметры — разумные дефолты (§4.3, «дорогие» не тюним): архитектура и
weight decay фиксированы конфигом, сид — из общего seed.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..logging_utils import get_logger

logger = get_logger("ranking.mlp")


class MLPRankerLTR:
    """Маленький MLP поверх признаков кандидата; интерфейс — как у LR-LTR.

    Параметры
    ---------
    hidden       : размеры скрытых слоёв (ReLU между ними).
    epochs       : число полнобатчевых шагов AdamW.
    lr           : learning rate.
    weight_decay : L2-штраф AdamW (аналог `l2` у LR).
    seed         : фиксирует инициализацию и порядок операций.
    """

    def __init__(
        self,
        *,
        hidden: Sequence[int] = (32, 16),
        epochs: int = 300,
        lr: float = 1e-2,
        weight_decay: float = 1e-3,
        seed: int = 42,
    ) -> None:
        self.hidden = [int(h) for h in hidden]
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self._model = None

    def _standardize(self, X: np.ndarray, *, fit: bool) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if fit:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0)
            self.std_[self.std_ == 0] = 1.0
        return (X - self.mean_) / self.std_

    def _build(self, n_features: int):
        import torch

        torch.manual_seed(self.seed)
        layers: list = []
        width = n_features
        for h in self.hidden:
            layers += [torch.nn.Linear(width, h), torch.nn.ReLU()]
            width = h
        layers.append(torch.nn.Linear(width, 1))
        return torch.nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "MLPRankerLTR":
        """Обучение на строках-кандидатах; `groups[i]` — индекс запроса строки i.

        Запросы без позитива среди кандидатов (GT вне топ-K fusion, ~2%)
        в лоссе не участвуют: listwise-цель для них не определена.
        """
        import torch

        Xs = self._standardize(X, fit=True)
        y = np.asarray(y, dtype=np.float32)
        groups = np.asarray(groups)

        # Строки → паддинг [n_groups, max_len] для softmax по запросу.
        _, group_idx = np.unique(groups, return_inverse=True)
        n_groups = int(group_idx.max()) + 1
        max_len = int(np.bincount(group_idx).max())
        row_pos = np.zeros(len(groups), dtype=np.int64)
        counts: dict[int, int] = {}
        for i, g in enumerate(group_idx):
            row_pos[i] = counts.get(int(g), 0)
            counts[int(g)] = row_pos[i] + 1

        feat = torch.zeros(n_groups, max_len, Xs.shape[1])
        label = torch.zeros(n_groups, max_len)
        pad = torch.ones(n_groups, max_len, dtype=torch.bool)
        gi = torch.as_tensor(group_idx, dtype=torch.long)
        rp = torch.as_tensor(row_pos, dtype=torch.long)
        feat[gi, rp] = torch.as_tensor(Xs)
        label[gi, rp] = torch.as_tensor(y)
        pad[gi, rp] = False
        has_positive = label.sum(dim=1) > 0

        torch.manual_seed(self.seed)
        model = self._build(Xs.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            logits = model(feat).squeeze(-1).masked_fill(pad, float("-inf"))
            log_p = torch.log_softmax(logits, dim=1)
            # Мульти-позитив: среднее −log P по позитивам запроса.
            per_query = -(log_p.masked_fill(label == 0, 0.0).sum(dim=1) / label.sum(dim=1).clamp(min=1.0))
            loss = per_query[has_positive].mean()
            loss.backward()
            optimizer.step()
            if epoch == 0 or (epoch + 1) % 100 == 0:
                logger.info("mlp epoch %d/%d: loss=%.4f", epoch + 1, self.epochs, loss.item())

        model.eval()
        self._model = model
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """σ(logit) кандидата — монотонна логиту, диапазон (0,1) как у LR."""
        import torch

        if self._model is None:
            raise RuntimeError("MLPRankerLTR.predict_proba вызван до fit()")
        Xs = self._standardize(X, fit=False)
        with torch.no_grad():
            logits = self._model(torch.as_tensor(Xs)).squeeze(-1)
        return torch.sigmoid(logits).numpy().astype(np.float64)
