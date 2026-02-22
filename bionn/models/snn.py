"""Leaky integrate-and-fire spiking neural network baseline (pure NumPy)."""

from __future__ import annotations

import numpy as np

from bionn.models.base import BaseModel


class SNNModel(BaseModel):
    name = "snn"

    def __init__(self, cfg: dict) -> None:
        self.n_in = cfg["general"]["num_channels"]
        self.n_out = cfg["general"]["num_patterns"]
        self.n_hid = 32
        self.dt = 1.0       # ms
        self.tau = 10.0      # membrane time constant
        self.thresh = 1.0
        self.reset_v = 0.0
        self.n_steps = 30
        self.lr = 0.05
        self.reset(cfg["general"]["seeds"][0])

    def reset(self, seed: int) -> None:
        rng = np.random.RandomState(seed)
        self.W_in = rng.randn(self.n_in, self.n_hid) * 0.3
        self.W_out = rng.randn(self.n_hid, self.n_out) * 0.3

    def _run(self, pattern: np.ndarray) -> np.ndarray:
        v_hid = np.zeros(self.n_hid)
        spike_count = np.zeros(self.n_hid)
        current_in = pattern @ self.W_in
        decay = np.exp(-self.dt / self.tau)
        for _ in range(self.n_steps):
            v_hid = v_hid * decay + current_in
            spikes = v_hid >= self.thresh
            spike_count += spikes
            v_hid[spikes] = self.reset_v
        return spike_count

    def _logits(self, spike_count: np.ndarray) -> np.ndarray:
        z = spike_count @ self.W_out
        e = np.exp(z - z.max())
        return e / e.sum()

    def train_step(self, pattern: np.ndarray, target: int, **kwargs) -> int:
        sc = self._run(pattern)
        probs = self._logits(sc)
        pred = int(np.argmax(probs))
        oh = np.zeros(self.n_out)
        oh[target] = 1
        err = probs - oh
        self.W_out -= self.lr * np.outer(sc, err)
        self.W_in -= self.lr * np.outer(pattern, (err @ self.W_out.T) * (sc > 0))
        return int(pred == target)

    def predict(self, pattern: np.ndarray, **kwargs) -> int:
        sc = self._run(pattern)
        return int(np.argmax(self._logits(sc)))
