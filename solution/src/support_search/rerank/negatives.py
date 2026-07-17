"""Майнинг обучающих групп для fine-tune реранкера (план §6, этап 6).

Обучающий пример — группа «(запрос, позитив) + N хард-негативов». Позитив —
пассаж GT-статьи; хард-негативы — верхние кандидаты ретривера, которых нет в GT
(именно верхние: они «похожи, но неверны», на них модель учится сильнее всего).
Неразмеченные ≠ нерелевантные, поэтому негативы берём только из уже найденных
кандидатов, а не из всего корпуса.

TODO §7.1: после графа ссылок дополнительно исключать из негативов статьи,
связанные ссылкой с позитивом (защита от false negatives).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass
class TrainGroup:
    """Одна обучающая группа: запрос, один позитив и фиксированный набор негативов."""

    query: str
    positive: str
    negatives: list[str]


def build_training_groups(
    query_ids: Sequence[int],
    query_text_of: Mapping[int, str],
    ground_truth: Mapping[int, set[int]],
    candidates_of: Mapping[int, Sequence[int]],
    passage_of: Mapping[tuple[int, int], str],
    *,
    n_negatives: int = 7,
) -> list[TrainGroup]:
    """Собрать группы фиксированного размера (1 позитив + n_negatives негативов).

    Всё адресуется по `query_id`, поэтому для фолда достаточно передать его
    train-идентификаторы. Группы с недостатком негативов или без пассажа позитива
    пропускаются — так все группы одного размера и батч батчируется тривиально.
    """
    groups: list[TrainGroup] = []
    for qid in query_ids:
        qid = int(qid)
        relevant = ground_truth.get(qid, set())
        if not relevant:
            continue
        hard_negatives = [a for a in candidates_of[qid] if a not in relevant]
        neg_passages = [
            passage_of[(qid, a)] for a in hard_negatives[:n_negatives] if (qid, a) in passage_of
        ]
        if len(neg_passages) < n_negatives:
            continue
        for positive in relevant:
            key = (qid, int(positive))
            if key in passage_of:
                groups.append(TrainGroup(query_text_of[qid], passage_of[key], list(neg_passages)))
    return groups
