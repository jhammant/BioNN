"""Continual learning benchmark — catastrophic forgetting test.

Train on Task A, then Task B, then re-evaluate Task A.
Metric: forgetting rate = (A_before - A_after) / A_before.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bionn.benchmarks.base import BaseBenchmark, add_noise
from bionn.models.base import BaseModel


class ContinualBenchmark(BaseBenchmark):
    name = "continual"

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        cc = cfg["benchmarks"]["continual"]
        self.task_a_epochs = cc["task_a_epochs"]
        self.task_b_epochs = cc["task_b_epochs"]
        self.trials_per_epoch = cc["trials_per_epoch"]
        self.train_noise = cc["noise_during_training"]

    def _eval(self, model: BaseModel, patterns: np.ndarray, rng: np.random.RandomState, **kwargs) -> float:
        correct = 0
        total = self.trials_per_epoch
        order = list(range(self.n_patterns)) * (total // self.n_patterns)
        for tc in order:
            correct += int(model.predict(add_noise(patterns[tc], 0.05, rng), **kwargs) == tc)
        return correct / total

    def run(self, model: BaseModel, seed: int, **kwargs) -> dict[str, Any]:
        rng = np.random.RandomState(seed)
        model.reset(seed)

        # Task A and Task B use different pattern sets
        patterns_a = self._make_patterns(rng)
        patterns_b = self._make_patterns(rng)

        # Phase 1: Train on Task A
        for _ in range(self.task_a_epochs):
            for tc in self._trial_order(self.trials_per_epoch, rng):
                model.train_step(add_noise(patterns_a[tc], self.train_noise, rng), tc, **kwargs)

        acc_a_before = self._eval(model, patterns_a, rng, **kwargs)

        # Phase 2: Train on Task B
        for _ in range(self.task_b_epochs):
            for tc in self._trial_order(self.trials_per_epoch, rng):
                model.train_step(add_noise(patterns_b[tc], self.train_noise, rng), tc, **kwargs)

        acc_b = self._eval(model, patterns_b, rng, **kwargs)
        acc_a_after = self._eval(model, patterns_a, rng, **kwargs)

        forgetting = 0.0
        if acc_a_before > 0:
            forgetting = (acc_a_before - acc_a_after) / acc_a_before

        return {
            "task_a_accuracy_before": acc_a_before,
            "task_a_accuracy_after": acc_a_after,
            "task_b_accuracy": acc_b,
            "forgetting_rate": forgetting,
        }
