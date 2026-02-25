"""Online SGD baseline — single-layer softmax, one sample at a time."""

from __future__ import annotations

import numpy as np

from bionn.models.base import BaseModel


class OnlineModel(BaseModel):
    name = "online"

    def __init__(self, cfg: dict) -> None:
        self.n_in = cfg["general"]["num_channels"]
        self.n_out = cfg["general"]["num_patterns"]
        self.lr = 0.15
        self.reset(cfg["general"]["seeds"][0])

    def reset(self, seed: int) -> None:
        rng = np.random.RandomState(seed)
        self.W = rng.randn(self.n_in, self.n_out) * 0.3
        self.b = np.zeros(self.n_out)
        self._last_activity = np.zeros(self.n_out)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def train_step(self, pattern: np.ndarray, target: int, **kwargs) -> int:
        z = pattern @ self.W + self.b
        self._last_activity = z
        probs = self._softmax(z)
        pred = int(np.argmax(probs))
        oh = np.zeros(self.n_out)
        oh[target] = 1
        err = probs - oh
        self.W -= self.lr * np.outer(pattern, err)
        self.b -= self.lr * err
        return int(pred == target)

    def predict(self, pattern: np.ndarray, **kwargs) -> int:
        z = pattern @ self.W + self.b
        self._last_activity = z
        probs = self._softmax(z)
        return int(np.argmax(probs))

    def get_internal_activity(self) -> np.ndarray:
        return self._last_activity
