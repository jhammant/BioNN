"""Complexity & Consciousness Indicators benchmark.

Three-phase protocol measuring how learning changes internal neural complexity
and whether biological neurons show transient consciousness-like dynamics.

Phases:
1. Spontaneous — random stimulation, no feedback → baseline complexity
2. Learning — classification training over epochs → complexity trajectory
3. Learned — structured patterns, no feedback → post-learning complexity
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import cosine

from bionn.benchmarks.base import BaseBenchmark, add_noise
from bionn.metrics.task import lempel_ziv_complexity
from bionn.models.base import BaseModel


class ComplexityBenchmark(BaseBenchmark):
    name = "complexity"

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        bc = cfg["benchmarks"]["complexity"]
        self.spontaneous_trials = bc["spontaneous_trials"]
        self.train_epochs = bc["train_epochs"]
        self.trials_per_epoch = bc["trials_per_epoch"]
        self.learned_trials = bc["learned_trials"]
        self.train_noise = bc["noise_during_training"]
        self.binarise_threshold = bc["binarise_threshold"]
        self.flicker_low = bc["flicker_lzc_low"]
        self.flicker_high = bc["flicker_lzc_high"]

    def run(self, model: BaseModel, seed: int, **kwargs) -> dict[str, Any]:
        rng = np.random.RandomState(seed)
        patterns = self._make_patterns(rng)
        model.reset(seed)

        all_trial_pcis: list[float] = []

        # Phase 1: Spontaneous — random stimulation, no learning
        spontaneous_activities: list[np.ndarray] = []
        for _ in range(self.spontaneous_trials):
            noise_pattern = rng.uniform(0, 1, self.n_channels)
            model.predict(noise_pattern, **kwargs)
            spontaneous_activities.append(model.get_internal_activity().copy())
            all_trial_pcis.append(self._trial_pci(model.get_internal_activity()))

        spontaneous_lzc = self._compute_lzc(spontaneous_activities)

        # Phase 2: Learning — classification training with per-epoch LZC
        complexity_trajectory: list[float] = []
        for epoch in range(self.train_epochs):
            epoch_activities: list[np.ndarray] = []
            order = self._trial_order(self.trials_per_epoch, rng)
            for tc in order:
                p = add_noise(patterns[tc], self.train_noise, rng)
                model.train_step(p, tc, **kwargs)
                epoch_activities.append(model.get_internal_activity().copy())
                all_trial_pcis.append(self._trial_pci(model.get_internal_activity()))
            complexity_trajectory.append(self._compute_lzc(epoch_activities))

        # Phase 3: Learned — structured patterns, no feedback
        learned_activities: list[np.ndarray] = []
        per_class_activities: dict[int, list[np.ndarray]] = {c: [] for c in range(self.n_patterns)}
        for _ in range(self.learned_trials):
            tc = rng.randint(0, self.n_patterns)
            model.predict(patterns[tc], **kwargs)
            act = model.get_internal_activity().copy()
            learned_activities.append(act)
            per_class_activities[tc].append(act)
            all_trial_pcis.append(self._trial_pci(act))

        learned_lzc = self._compute_lzc(learned_activities)
        differentiation = self._compute_differentiation(per_class_activities)
        integration = self._compute_integration(learned_activities)
        consciousness_score = self._compute_consciousness_score(
            learned_lzc, spontaneous_lzc, differentiation, integration,
        )

        flicker_count, flicker_rate = self._detect_flickers(all_trial_pcis)

        convergence_window = complexity_trajectory[-3:] if len(complexity_trajectory) >= 3 else complexity_trajectory
        complexity_convergence = float(np.std(convergence_window)) if convergence_window else 0.0

        return {
            "spontaneous_lzc": spontaneous_lzc,
            "learned_lzc": learned_lzc,
            "complexity_delta": learned_lzc - spontaneous_lzc,
            "complexity_trajectory": complexity_trajectory,
            "complexity_convergence": complexity_convergence,
            "response_differentiation": differentiation,
            "consciousness_score": consciousness_score,
            "flicker_rate": flicker_rate,
            "flicker_count": flicker_count,
            "per_trial_pci": all_trial_pcis,
        }

    def _compute_lzc(self, activities: list[np.ndarray]) -> float:
        if not activities:
            return 0.0
        concat = np.concatenate(activities)
        binary = (concat > self.binarise_threshold).astype(np.uint8)
        if len(binary) == 0:
            return 0.0
        return lempel_ziv_complexity(binary)

    def _trial_pci(self, activity: np.ndarray) -> float:
        binary = (activity > self.binarise_threshold).astype(np.uint8)
        if len(binary) == 0:
            return 0.0
        return lempel_ziv_complexity(binary)

    def _compute_differentiation(self, per_class: dict[int, list[np.ndarray]]) -> float:
        means: list[np.ndarray] = []
        for acts in per_class.values():
            if acts:
                means.append(np.mean(acts, axis=0))
        if len(means) < 2:
            return 0.0
        distances: list[float] = []
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                norm_i = np.linalg.norm(means[i])
                norm_j = np.linalg.norm(means[j])
                if norm_i < 1e-10 or norm_j < 1e-10:
                    distances.append(0.0)
                else:
                    distances.append(cosine(means[i], means[j]))
        return float(np.clip(np.mean(distances), 0, 1))

    def _compute_integration(self, activities: list[np.ndarray]) -> float:
        if not activities:
            return 0.0
        correlations: list[float] = []
        for act in activities:
            if len(act) < 2:
                continue
            std = np.std(act)
            if std < 1e-10:
                correlations.append(0.0)
                continue
            corrmat = np.corrcoef(act.reshape(-1, 1).T) if len(act) < 2 else None
            # Mean pairwise correlation within the response vector
            # Treat as channel co-activation: correlation between pairs of elements
            # across the response vector seen as a time-like signal
            act_norm = (act - act.mean()) / (std + 1e-10)
            n = len(act_norm)
            pairs = []
            for k in range(n - 1):
                pairs.append(act_norm[k] * act_norm[k + 1])
            if pairs:
                correlations.append(float(np.clip(np.mean(pairs), -1, 1)))
        if not correlations:
            return 0.0
        return float(np.clip((np.mean(correlations) + 1) / 2, 0, 1))

    def _compute_consciousness_score(
        self,
        learned_lzc: float,
        spontaneous_lzc: float,
        differentiation: float,
        integration: float,
    ) -> float:
        # Normalise LZC to [0, 1] — use learned_lzc directly (already normalised by LZ76)
        normalised_lzc = np.clip(learned_lzc, 0, 1)
        # Intermediate LZC peaks at 0.5: 1.0 - abs(x - 0.5) * 2
        intermediate_lzc = 1.0 - abs(normalised_lzc - 0.5) * 2
        score = (intermediate_lzc * 0.4) + (differentiation * 0.3) + (integration * 0.3)
        return float(np.clip(score, 0, 1))

    def _detect_flickers(self, trial_pcis: list[float]) -> tuple[int, float]:
        if not trial_pcis:
            return 0, 0.0
        flickers = [1 for pci in trial_pcis if self.flicker_low <= pci <= self.flicker_high]
        count = len(flickers)
        rate = count / len(trial_pcis)
        return count, rate
