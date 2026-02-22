"""Abstract base class for all models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Uniform interface for BNN, MLP, SNN, and online-SGD baselines.

    Models that need a CL ``Neurons`` object should set ``requires_neurons = True``.
    The runner will pass ``neurons=<Neurons>`` through **kwargs on every call.
    """

    requires_neurons: bool = False
    name: str = "base"

    @abstractmethod
    def train_step(
        self,
        pattern: np.ndarray,
        target: int,
        **kwargs,
    ) -> int:
        """One training step. Return 1 if prediction was correct, else 0."""

    @abstractmethod
    def predict(self, pattern: np.ndarray, **kwargs) -> int:
        """Return predicted class index."""

    @abstractmethod
    def reset(self, seed: int) -> None:
        """Re-initialise weights with the given seed."""
