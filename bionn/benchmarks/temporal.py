"""Temporal sequence learning benchmark.

Trains the model to predict the next pattern in a fixed cyclic sequence
(e.g. A->B->C->D->A->...). Metric: sequence prediction accuracy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bionn.benchmarks.base import BaseBenchmark, add_noise
from bionn.models.base import BaseModel


class TemporalBenchmark(BaseBenchmark):
    name = "temporal"

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        tc = cfg["benchmarks"]["temporal"]
        self.seq_len = tc["sequence_length"]
        self.max_epochs = tc["max_epochs"]
        self.trials_per_epoch = tc["trials_per_epoch"]
        self.train_noise = tc["noise_during_training"]

    def run(self, model: BaseModel, seed: int, **kwargs) -> dict[str, Any]:
        rng = np.random.RandomState(seed)
        patterns = self._make_patterns(rng)
        model.reset(seed)

        # Sequence: 0 -> 1 -> 2 -> 3 -> 0 -> ...
        # Input = current pattern, target = next pattern index
        sequence = list(range(min(self.seq_len, self.n_patterns)))

        learning_curve: list[float] = []
        for epoch in range(self.max_epochs):
            correct = 0
            for _ in range(self.trials_per_epoch):
                pos = rng.randint(0, len(sequence))
                current_idx = sequence[pos]
                next_idx = sequence[(pos + 1) % len(sequence)]
                p = add_noise(patterns[current_idx], self.train_noise, rng)
                correct += model.train_step(p, next_idx, **kwargs)
            learning_curve.append(correct / self.trials_per_epoch)

        # Final evaluation — one full pass through the sequence
        eval_correct = 0
        eval_total = 0
        for pos in range(len(sequence)):
            current_idx = sequence[pos]
            next_idx = sequence[(pos + 1) % len(sequence)]
            pred = model.predict(patterns[current_idx], **kwargs)
            eval_correct += int(pred == next_idx)
            eval_total += 1

        return {
            "learning_curve": learning_curve,
            "final_accuracy": learning_curve[-1],
            "sequence_accuracy": eval_correct / eval_total,
        }
