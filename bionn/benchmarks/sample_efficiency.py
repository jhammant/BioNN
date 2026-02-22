"""Sample efficiency benchmark — vary training budget, measure competence."""

from __future__ import annotations

from typing import Any

import numpy as np

from bionn.benchmarks.base import BaseBenchmark, add_noise
from bionn.models.base import BaseModel


class SampleEfficiencyBenchmark(BaseBenchmark):
    name = "sample_efficiency"

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        sc = cfg["benchmarks"]["sample_efficiency"]
        self.budgets = sc["budgets_per_class"]
        self.eval_trials = sc["eval_trials"]
        self.competence = sc["competence_threshold"]

    def run(self, model: BaseModel, seed: int, **kwargs) -> dict[str, Any]:
        rng = np.random.RandomState(seed)
        patterns = self._make_patterns(rng)

        accuracies: list[float] = []
        samples_to_competence: int | None = None

        for budget in self.budgets:
            model.reset(seed)
            # Train: `budget` samples per class
            for _ in range(budget):
                for tc in range(self.n_patterns):
                    p = add_noise(patterns[tc], 0.05, rng)
                    model.train_step(p, tc, **kwargs)

            # Evaluate
            correct = 0
            order = list(range(self.n_patterns)) * (self.eval_trials // self.n_patterns)
            for tc in order:
                correct += int(model.predict(add_noise(patterns[tc], 0.05, rng), **kwargs) == tc)
            acc = correct / self.eval_trials
            accuracies.append(acc)

            if acc >= self.competence and samples_to_competence is None:
                samples_to_competence = budget * self.n_patterns

        return {
            "budgets_per_class": self.budgets,
            "accuracies": accuracies,
            "samples_to_competence": samples_to_competence,
        }
