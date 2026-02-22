"""Two-layer NumPy MLP baseline."""

from __future__ import annotations

import numpy as np

from bionn.models.base import BaseModel


class MLPModel(BaseModel):
    name = "mlp"

    def __init__(self, cfg: dict) -> None:
        self.n_in = cfg["general"]["num_channels"]
        self.n_out = cfg["general"]["num_patterns"]
        self.n_hid = 16
        self.lr = 0.1
        self.reset(cfg["general"]["seeds"][0])

    def reset(self, seed: int) -> None:
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(self.n_in, self.n_hid) * 0.3
        self.b1 = np.zeros(self.n_hid)
        self.W2 = rng.randn(self.n_hid, self.n_out) * 0.3
        self.b2 = np.zeros(self.n_out)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        self._z1 = x @ self.W1 + self.b1
        self._a1 = np.maximum(0, self._z1)
        z2 = self._a1 @ self.W2 + self.b2
        e = np.exp(z2 - z2.max())
        self._probs = e / e.sum()
        return self._probs

    def train_step(self, pattern: np.ndarray, target: int, **kwargs) -> int:
        self._forward(pattern)
        oh = np.zeros(self.n_out)
        oh[target] = 1
        dz2 = self._probs - oh
        self.W2 -= self.lr * np.outer(self._a1, dz2)
        self.b2 -= self.lr * dz2
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self._z1 > 0)
        self.W1 -= self.lr * np.outer(pattern, dz1)
        self.b1 -= self.lr * dz1
        return int(np.argmax(self._probs) == target)

    def predict(self, pattern: np.ndarray, **kwargs) -> int:
        self._forward(pattern)
        return int(np.argmax(self._probs))
