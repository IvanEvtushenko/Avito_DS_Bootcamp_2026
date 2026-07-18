"""LR-шедулер дообучения реранкера: warmup растёт, cosine затухает, constant = None."""
from __future__ import annotations

import pathlib
import sys
import unittest

_SOL = pathlib.Path(__file__).resolve().parents[2]
for _p in (_SOL / "src", _SOL / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

from support_search.rerank.train import _build_scheduler  # noqa: E402

LR = 1.0e-5


def _optimizer():
    return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=LR)


def _lr_trace(schedule: str, warmup_ratio: float, total_steps: int) -> list[float]:
    opt = _optimizer()
    sch = _build_scheduler(opt, schedule=schedule, warmup_ratio=warmup_ratio, total_steps=total_steps)
    lrs = []
    for _ in range(total_steps):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    return lrs


class TestScheduler(unittest.TestCase):
    def test_constant_is_none(self):
        self.assertIsNone(_build_scheduler(_optimizer(), schedule="constant", warmup_ratio=0.0, total_steps=10))

    def test_unknown_schedule_raises(self):
        with self.assertRaises(ValueError):
            _build_scheduler(_optimizer(), schedule="linear", warmup_ratio=0.0, total_steps=10)

    def test_warmup_rises_to_base_lr(self):
        lrs = _lr_trace("cosine", warmup_ratio=0.1, total_steps=100)
        self.assertLess(lrs[0], lrs[5])                    # разогрев растёт
        self.assertAlmostEqual(lrs[9], LR, delta=LR * 1e-6)  # конец warmup = базовый LR

    def test_cosine_decays_to_near_zero(self):
        lrs = _lr_trace("cosine", warmup_ratio=0.1, total_steps=100)
        self.assertGreater(lrs[10], lrs[50])
        self.assertGreater(lrs[50], lrs[99])
        self.assertLess(lrs[99], LR * 0.01)                # к концу почти ноль

    def test_zero_warmup_starts_at_base(self):
        lrs = _lr_trace("cosine", warmup_ratio=0.0, total_steps=50)
        self.assertAlmostEqual(lrs[0], LR, delta=LR * 1e-6)

    def test_monotone_decay_after_warmup(self):
        lrs = _lr_trace("cosine", warmup_ratio=0.2, total_steps=50)
        tail = lrs[10:]
        self.assertTrue(all(a >= b for a, b in zip(tail, tail[1:])))


if __name__ == "__main__":
    unittest.main()
