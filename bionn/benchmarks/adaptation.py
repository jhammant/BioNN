"""Adaptation benchmark — train, shuffle labels, measure re-learning speed."""

from __future__ import annotations

from typing import Any

import numpy as np

from bionn.benchmarks.base import BaseBenchmark, add_noise
from bionn.models.base import BaseModel


class AdaptationBenchmark(BaseBenchmark):
    name = "adaptation"

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        ac = cfg["benchmarks"]["adaptation"]
        self.pre_train_epochs = ac["pre_train_epochs"]
        self.re_adapt_epochs = ac["re_adapt_epochs"]
        self.trials_per_epoch = ac["trials_per_epoch"]
        self.target_acc = ac["target_accuracy"]
        self.train_noise = ac["noise_during_training"]

    def run(self, model: BaseModel, seed: int, **kwargs) -> dict[str, Any]:
        rng = np.random.RandomState(seed)
        patterns = self._make_patterns(rng)
        model.reset(seed)

        # Phase 1: pre-train on original mapping
        pre_curve: list[float] = []
        pre_epoch_to_target: int | None = None
        for epoch in range(self.pre_train_epochs):
            correct = 0
            for tc in self._trial_order(self.trials_per_epoch, rng):
                p = add_noise(patterns[tc], self.train_noise, rng)
                correct += model.train_step(p, tc, **kwargs)
            acc = correct / self.trials_per_epoch
            pre_curve.append(acc)
            if acc >= self.target_acc and pre_epoch_to_target is None:
                pre_epoch_to_target = epoch + 1

        # Phase 2: shuffle label mapping, re-train
        perm = rng.permutation(self.n_patterns)
        shuffled = patterns[perm]

        re_curve: list[float] = []
        re_epoch_to_target: int | None = None
        for epoch in range(self.re_adapt_epochs):
            correct = 0
            for tc in self._trial_order(self.trials_per_epoch, rng):
                p = add_noise(shuffled[tc], self.train_noise, rng)
                correct += model.train_step(p, tc, **kwargs)
            acc = correct / self.trials_per_epoch
            re_curve.append(acc)
            if acc >= self.target_acc and re_epoch_to_target is None:
                re_epoch_to_target = epoch + 1

        ratio = None
        if pre_epoch_to_target and re_epoch_to_target:
            ratio = re_epoch_to_target / pre_epoch_to_target

        return {
            "pre_learning_curve": pre_curve,
            "re_learning_curve": re_curve,
            "pre_epochs_to_target": pre_epoch_to_target,
            "re_epochs_to_target": re_epoch_to_target,
            "adaptation_ratio": ratio,
        }
