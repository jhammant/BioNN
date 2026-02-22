"""Base benchmark ABC and shared pattern utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from bionn.models.base import BaseModel


def make_patterns(n_patterns: int, n_channels: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate random patterns with 2 'hot' channels per class."""
    pats = []
    for _ in range(n_patterns):
        p = rng.uniform(0.05, 0.2, n_channels)
        hot = rng.choice(n_channels, size=2, replace=False)
        p[hot] = rng.uniform(0.7, 1.0, 2)
        pats.append(p)
    return np.array(pats)


def add_noise(pattern: np.ndarray, sigma: float, rng: np.random.RandomState) -> np.ndarray:
    return np.clip(pattern + rng.normal(0, sigma, pattern.shape), 0, 1)


class BaseBenchmark(ABC):
    """Every benchmark returns a dict of metrics keyed by model name."""

    name: str = "base"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.gcfg = cfg["general"]
        self.n_patterns = self.gcfg["num_patterns"]
        self.n_channels = self.gcfg["num_channels"]

    @abstractmethod
    def run(
        self,
        model: BaseModel,
        seed: int,
        **kwargs,
    ) -> dict[str, Any]:
        """Execute benchmark for one model + one seed. Return metrics dict."""

    def _make_patterns(self, rng: np.random.RandomState) -> np.ndarray:
        return make_patterns(self.n_patterns, self.n_channels, rng)

    def _trial_order(self, trials_per_epoch: int, rng: np.random.RandomState) -> list[int]:
        order = list(range(self.n_patterns)) * (trials_per_epoch // self.n_patterns)
        rng.shuffle(order)
        return order
