"""Noise robustness benchmark — train clean, evaluate at increasing noise."""

from __future__ import annotations

from typing import Any

import numpy as np

from bionn.benchmarks.base import BaseBenchmark, add_noise
from bionn.models.base import BaseModel


class NoiseBenchmark(BaseBenchmark):
    name = "noise"

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        nc = cfg["benchmarks"]["noise"]
        self.train_epochs = nc["train_epochs"]
        self.eval_trials = nc["eval_trials_per_level"]
        self.noise_levels = nc["noise_levels"]
        self.train_noise = nc["noise_during_training"]

    def run(self, model: BaseModel, seed: int, **kwargs) -> dict[str, Any]:
        rng = np.random.RandomState(seed)
        patterns = self._make_patterns(rng)
        model.reset(seed)
        tpe = self.eval_trials  # reuse as trials_per_epoch during training

        # Train on clean(ish) data
        for _ in range(self.train_epochs):
            order = list(range(self.n_patterns)) * (tpe // self.n_patterns)
            rng.shuffle(order)
            for tc in order:
                model.train_step(add_noise(patterns[tc], self.train_noise, rng), tc, **kwargs)

        # Evaluate at each noise level
        accuracies: list[float] = []
        for sigma in self.noise_levels:
            correct = 0
            order = list(range(self.n_patterns)) * (self.eval_trials // self.n_patterns)
            for tc in order:
                p = add_noise(patterns[tc], sigma, rng)
                correct += int(model.predict(p, **kwargs) == tc)
            accuracies.append(correct / self.eval_trials)

        # Noise tolerance threshold — highest noise where acc >= 50%
        threshold = 0.0
        for sigma, acc in zip(self.noise_levels, accuracies):
            if acc >= 0.50:
                threshold = sigma

        return {
            "noise_levels": self.noise_levels,
            "accuracies": accuracies,
            "noise_tolerance_threshold": threshold,
        }
