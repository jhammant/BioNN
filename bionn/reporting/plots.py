"""Matplotlib visualisations for benchmark results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _model_colors() -> dict[str, str]:
    return {"bnn": "#e74c3c", "mlp": "#3498db", "snn": "#2ecc71", "online": "#9b59b6"}


def plot_learning_curves(
    results: dict[str, list[dict[str, Any]]],
    benchmark: str,
    out_dir: Path,
) -> Path:
    """Plot mean learning curves with CI bands for each model."""
    colors = _model_colors()
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, seed_results in results.items():
        curves = [r["learning_curve"] for r in seed_results if "learning_curve" in r]
        if not curves:
            continue
        max_len = max(len(c) for c in curves)
        padded = np.array([c + [c[-1]] * (max_len - len(c)) for c in curves])
        mean = padded.mean(axis=0)
        std = padded.std(axis=0)
        epochs = np.arange(1, max_len + 1)
        color = colors.get(model_name, "#7f8c8d")
        ax.plot(epochs, mean, label=model_name.upper(), color=color, linewidth=2)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{benchmark.replace('_', ' ').title()} — Learning Curves")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"{benchmark}_learning_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_noise_degradation(
    results: dict[str, list[dict[str, Any]]],
    out_dir: Path,
) -> Path:
    """Plot accuracy vs. noise level for each model."""
    colors = _model_colors()
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, seed_results in results.items():
        accs = [r["accuracies"] for r in seed_results if "accuracies" in r]
        if not accs:
            continue
        levels = seed_results[0]["noise_levels"]
        arr = np.array(accs)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        color = colors.get(model_name, "#7f8c8d")
        ax.plot(levels, mean, "o-", label=model_name.upper(), color=color, linewidth=2)
        ax.fill_between(levels, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Noise (sigma)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Noise Robustness — Accuracy vs. Noise Level")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "noise_degradation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_sample_efficiency(
    results: dict[str, list[dict[str, Any]]],
    out_dir: Path,
) -> Path:
    """Plot accuracy vs. training budget for each model."""
    colors = _model_colors()
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, seed_results in results.items():
        accs = [r["accuracies"] for r in seed_results if "accuracies" in r]
        if not accs:
            continue
        budgets = seed_results[0]["budgets_per_class"]
        arr = np.array(accs)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        color = colors.get(model_name, "#7f8c8d")
        ax.plot(budgets, mean, "o-", label=model_name.upper(), color=color, linewidth=2)
        ax.fill_between(budgets, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Samples per class")
    ax.set_ylabel("Accuracy")
    ax.set_title("Sample Efficiency — Accuracy vs. Training Budget")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "sample_efficiency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_summary_radar(
    summary: dict[str, dict[str, float]],
    out_dir: Path,
) -> Path:
    """Radar chart comparing normalised scores across benchmarks."""
    models = list(summary.keys())
    if not models:
        return out_dir / "summary_radar.png"

    metrics = list(summary[models[0]].keys())
    n = len(metrics)
    if n == 0:
        return out_dir / "summary_radar.png"

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    colors = _model_colors()
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for model_name in models:
        vals = [summary[model_name].get(m, 0) or 0 for m in metrics]
        vals += vals[:1]
        color = colors.get(model_name, "#7f8c8d")
        ax.plot(angles, vals, "o-", label=model_name.upper(), color=color, linewidth=2)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], size=9)
    ax.set_title("Model Comparison — Normalised Scores", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    path = out_dir / "summary_radar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
