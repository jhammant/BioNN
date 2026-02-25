"""Biological Neural Network model — CL SDK wrapper.

Works identically on simulator and DishBrain hardware. Requires ``neurons``
to be passed as a keyword argument (``requires_neurons = True``).
"""

from __future__ import annotations

import numpy as np

import cl

from bionn.models.base import BaseModel


class BNNModel(BaseModel):
    requires_neurons = True
    name = "bnn"

    def __init__(self, cfg: dict) -> None:
        self.n_in = cfg["general"]["num_channels"]
        self.n_out = cfg["general"]["num_patterns"]
        bcfg = cfg["bnn"]
        self.stim_threshold = bcfg["stim_threshold"]
        self.stim_amp_max = bcfg["stim_amplitude_max"]
        self.stim_amp_min = bcfg["stim_amplitude_min"]
        self.stim_quanta = bcfg["stim_quanta"]
        self.read_ticks = bcfg["read_ticks"]
        self.read_tps = bcfg["read_tps"]
        self.lr = bcfg["learning_rate"]
        self.w_min = bcfg["weight_min"]
        self.w_max = bcfg["weight_max"]
        self.n_readout = bcfg["num_readout_channels"]
        self.reset(cfg["general"]["seeds"][0])

    def reset(self, seed: int) -> None:
        rng = np.random.RandomState(seed)
        self.W = rng.uniform(0.1, 0.5, (self.n_readout, self.n_out))
        self._last_activity = np.zeros(self.n_readout)

    def _stim_and_read(self, neurons: cl.Neurons, pattern: np.ndarray) -> np.ndarray:
        for ch in range(self.n_in):
            if pattern[ch] > self.stim_threshold:
                amp = max(self.stim_amp_min, int(pattern[ch] * self.stim_amp_max) // self.stim_quanta * self.stim_quanta)
                neurons.stim(ch, cl.StimDesign(amp, -1, amp, 1))
        sc = np.zeros(self.n_readout)
        for tick in neurons.loop(ticks_per_second=self.read_tps, stop_after_ticks=self.read_ticks):
            for s in tick.analysis.spikes:
                if s.channel < self.n_readout:
                    sc[s.channel] += 1
        self._last_activity = sc
        return sc

    def train_step(self, pattern: np.ndarray, target: int, **kwargs) -> int:
        neurons = kwargs["neurons"]
        sc = self._stim_and_read(neurons, pattern)
        pred = int(np.argmax(sc @ self.W))
        if pred == target:
            self.W[sc > 0, target] += self.lr
        else:
            self.W[:, pred] -= self.lr * 0.3 * (sc / (sc.max() + 1e-8))
            self.W[sc > 0, target] += self.lr * 0.5
        self.W = np.clip(self.W, self.w_min, self.w_max)
        return int(pred == target)

    def predict(self, pattern: np.ndarray, **kwargs) -> int:
        neurons = kwargs["neurons"]
        sc = self._stim_and_read(neurons, pattern)
        return int(np.argmax(sc @ self.W))

    def get_internal_activity(self) -> np.ndarray:
        return self._last_activity
