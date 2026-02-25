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


def plot_complexity_trajectory(
    results: dict[str, list[dict[str, Any]]],
    out_dir: Path,
) -> Path:
    """Plot LZ complexity over learning epochs for each model."""
    colors = _model_colors()
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, seed_results in results.items():
        trajectories = [r["complexity_trajectory"] for r in seed_results if "complexity_trajectory" in r]
        baselines = [r["spontaneous_lzc"] for r in seed_results if "spontaneous_lzc" in r]
        if not trajectories:
            continue
        max_len = max(len(t) for t in trajectories)
        padded = np.array([t + [t[-1]] * (max_len - len(t)) for t in trajectories])
        mean = padded.mean(axis=0)
        std = padded.std(axis=0)
        epochs = np.arange(1, max_len + 1)
        color = colors.get(model_name, "#7f8c8d")
        ax.plot(epochs, mean, label=model_name.upper(), color=color, linewidth=2)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=color)
        if baselines:
            baseline_mean = float(np.mean(baselines))
            ax.axhline(baseline_mean, color=color, linestyle="--", alpha=0.5, linewidth=1)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("LZ Complexity")
    ax.set_title("Complexity Trajectory — LZC Over Learning")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "complexity_trajectory.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_consciousness_indicators(
    results: dict[str, list[dict[str, Any]]],
    out_dir: Path,
) -> Path:
    """2x2 dashboard: consciousness score, differentiation, LZC comparison, flicker rate."""
    colors = _model_colors()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.6

    def _bar_data(key: str) -> tuple[list[float], list[float]]:
        means, stds = [], []
        for m in models:
            vals = [r[key] for r in results[m] if key in r]
            means.append(float(np.mean(vals)) if vals else 0.0)
            stds.append(float(np.std(vals)) if vals else 0.0)
        return means, stds

    # Top-left: Consciousness score
    ax = axes[0, 0]
    means, stds = _bar_data("consciousness_score")
    bars = ax.bar(x, means, width, yerr=stds, capsize=4,
                  color=[colors.get(m, "#7f8c8d") for m in models], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylabel("Score")
    ax.set_title("Consciousness Score")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    # Top-right: Response differentiation
    ax = axes[0, 1]
    means, stds = _bar_data("response_differentiation")
    ax.bar(x, means, width, yerr=stds, capsize=4,
           color=[colors.get(m, "#7f8c8d") for m in models], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylabel("Differentiation")
    ax.set_title("Response Differentiation")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom-left: Spontaneous vs Learned LZC
    ax = axes[1, 0]
    w = 0.3
    sp_means, sp_stds = _bar_data("spontaneous_lzc")
    lr_means, lr_stds = _bar_data("learned_lzc")
    ax.bar(x - w / 2, sp_means, w, yerr=sp_stds, capsize=3, label="Spontaneous", alpha=0.7, color="#95a5a6")
    ax.bar(x + w / 2, lr_means, w, yerr=lr_stds, capsize=3, label="Learned",
           color=[colors.get(m, "#7f8c8d") for m in models], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylabel("LZ Complexity")
    ax.set_title("Spontaneous vs Learned LZC")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom-right: Flicker rate
    ax = axes[1, 1]
    means, stds = _bar_data("flicker_rate")
    ax.bar(x, means, width, yerr=stds, capsize=4,
           color=[colors.get(m, "#7f8c8d") for m in models], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylabel("Flicker Rate")
    ax.set_title("Consciousness-Like Flicker Rate")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Consciousness Indicators", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "consciousness_indicators.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_flicker_raster(
    results: dict[str, list[dict[str, Any]]],
    out_dir: Path,
    model_name: str = "bnn",
) -> Path:
    """Per-trial PCI heatmap for a single model across seeds."""
    seed_results = results.get(model_name, [])
    if not seed_results:
        # Fallback to first available model
        model_name = next(iter(results), "")
        seed_results = results.get(model_name, [])

    pci_lists = [r["per_trial_pci"] for r in seed_results if "per_trial_pci" in r]
    if not pci_lists:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No per-trial PCI data", ha="center", va="center")
        path = out_dir / "flicker_raster.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    max_trials = max(len(p) for p in pci_lists)
    padded = np.zeros((len(pci_lists), max_trials))
    for i, p in enumerate(pci_lists):
        padded[i, : len(p)] = p

    fig, ax = plt.subplots(figsize=(12, max(3, len(pci_lists))))
    im = ax.imshow(padded, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, label="Per-trial PCI")

    # Mark flickers (PCI in sweet spot)
    for i in range(padded.shape[0]):
        for j in range(padded.shape[1]):
            if 0.3 <= padded[i, j] <= 0.7 and padded[i, j] > 0:
                ax.plot(j, i, "k.", markersize=2, alpha=0.5)

    # Phase boundaries
    if seed_results:
        cfg_sample = seed_results[0]
        spont = cfg_sample.get("spontaneous_lzc", None)
        # Estimate phase boundaries from trial counts
        # spontaneous_trials + train_epochs * trials_per_epoch + learned_trials
        n_spont = 32  # default
        n_learn_total = 15 * 16  # default
        ax.axvline(n_spont - 0.5, color="white", linewidth=2, linestyle="--", alpha=0.8)
        ax.axvline(n_spont + n_learn_total - 0.5, color="white", linewidth=2, linestyle="--", alpha=0.8)
        ax.text(n_spont / 2, -0.7, "Spontaneous", ha="center", fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))
        ax.text(n_spont + n_learn_total / 2, -0.7, "Learning", ha="center", fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))
        ax.text(n_spont + n_learn_total + 16, -0.7, "Learned", ha="center", fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

    ax.set_xlabel("Trial")
    ax.set_ylabel("Seed")
    ax.set_yticks(range(len(pci_lists)))
    ax.set_yticklabels([f"Seed {i + 1}" for i in range(len(pci_lists))])
    ax.set_title(f"Flicker Raster — Per-Trial PCI ({model_name.upper()})")
    fig.tight_layout()
    path = out_dir / "flicker_raster.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_complexity_phase_portrait(
    results: dict[str, list[dict[str, Any]]],
    out_dir: Path,
) -> Path:
    """Scatter: spontaneous_lzc vs learned_lzc for each model×seed."""
    colors = _model_colors()
    fig, ax = plt.subplots(figsize=(7, 7))

    all_vals: list[float] = []
    for model_name, seed_results in results.items():
        sx = [r["spontaneous_lzc"] for r in seed_results if "spontaneous_lzc" in r]
        sy = [r["learned_lzc"] for r in seed_results if "learned_lzc" in r]
        if not sx or not sy:
            continue
        color = colors.get(model_name, "#7f8c8d")
        ax.scatter(sx, sy, label=model_name.upper(), color=color, s=60, alpha=0.8, edgecolors="white", linewidths=0.5)
        all_vals.extend(sx)
        all_vals.extend(sy)

    if all_vals:
        lo = max(0, min(all_vals) - 0.05)
        hi = min(1, max(all_vals) + 0.05)
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=1, label="No change")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    ax.set_xlabel("Spontaneous LZC")
    ax.set_ylabel("Learned LZC")
    ax.set_title("Complexity Phase Portrait — Before vs After Learning")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "complexity_phase_portrait.png"
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
