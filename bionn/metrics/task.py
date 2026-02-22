"""Task-level metrics: accuracy, convergence, AUC, confidence intervals."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def confidence_interval(values: list[float], confidence: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) for a list of values."""
    a = np.array(values)
    mean = float(np.mean(a))
    if len(a) < 2:
        return mean, mean, mean
    se = stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2, len(a) - 1)
    return mean, mean - h, mean + h


def learning_curve_auc(curve: list[float]) -> float:
    """Area under the learning curve (higher = faster learning)."""
    return float(np.trapz(curve))


def aggregate_metric(
    results: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    """Aggregate a scalar metric across seeds, returning mean + CI."""
    values = [r[key] for r in results if r.get(key) is not None]
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    mean, lo, hi = confidence_interval(values)
    return {"mean": mean, "ci_low": lo, "ci_high": hi, "n": len(values)}


def significance_test(
    values_a: list[float],
    values_b: list[float],
) -> dict[str, float | None]:
    """Independent-samples t-test between two groups of metric values."""
    if len(values_a) < 2 or len(values_b) < 2:
        return {"t_stat": None, "p_value": None}
    t, p = stats.ttest_ind(values_a, values_b, equal_var=False)
    return {"t_stat": float(t), "p_value": float(p)}
