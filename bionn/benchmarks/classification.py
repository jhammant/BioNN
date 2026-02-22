"""Pattern classification benchmark — 4-class discrimination."""

from __future__ import annotations

from typing import Any

import numpy as np

from bionn.benchmarks.base import BaseBenchmark, add_noise
from bionn.models.base import BaseModel


class ClassificationBenchmark(BaseBenchmark):
    name = "classification"

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        bc = cfg["benchmarks"]["classification"]
        self.max_epochs = bc["max_epochs"]
        self.trials_per_epoch = bc["trials_per_epoch"]
        self.target_acc = bc["target_accuracy"]
        self.train_noise = bc["noise_during_training"]

    def run(self, model: BaseModel, seed: int, **kwargs) -> dict[str, Any]:
        rng = np.random.RandomState(seed)
        patterns = self._make_patterns(rng)
        model.reset(seed)

        learning_curve: list[float] = []
        epoch_to_target: int | None = None

        for epoch in range(self.max_epochs):
            correct = 0
            order = self._trial_order(self.trials_per_epoch, rng)
            for tc in order:
                p = add_noise(patterns[tc], self.train_noise, rng)
                correct += model.train_step(p, tc, **kwargs)
            acc = correct / self.trials_per_epoch
            learning_curve.append(acc)
            if acc >= self.target_acc and epoch_to_target is None:
                epoch_to_target = epoch + 1

        return {
            "learning_curve": learning_curve,
            "final_accuracy": learning_curve[-1],
            "epochs_to_target": epoch_to_target,
        }
