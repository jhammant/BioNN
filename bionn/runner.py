"""Main orchestrator — runs all benchmarks across all models and seeds."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import cl
import numpy as np

from bionn.benchmarks import BENCHMARKS
from bionn.metrics.neuro import analyse_recording
from bionn.metrics.task import aggregate_metric, learning_curve_auc
from bionn.models import MODELS
from bionn.reporting.plots import (
    plot_complexity_phase_portrait,
    plot_complexity_trajectory,
    plot_consciousness_indicators,
    plot_flicker_raster,
    plot_learning_curves,
    plot_noise_degradation,
    plot_sample_efficiency,
    plot_summary_radar,
)
from bionn.reporting.report import generate_report

logger = logging.getLogger(__name__)


def run_suite(cfg: dict) -> dict[str, Any]:
    """Execute the full benchmark suite. Returns the results dict."""
    seeds = cfg["general"]["seeds"]
    results_dir = Path(cfg["general"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    recordings_dir = results_dir / "recordings"
    recordings_dir.mkdir(exist_ok=True)

    model_names = cfg.get("models", list(MODELS.keys()))
    bench_names = list(cfg["benchmarks"].keys())

    # all_results[benchmark][model] = [seed_result, ...]
    all_results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    # neuro_results[benchmark] = [seed_metrics, ...]
    neuro_results: dict[str, list[dict[str, Any]]] = {}
    # recording paths for post-hoc analysis
    recording_paths: dict[str, list[str]] = {}

    with cl.open() as neurons:
        for bench_name in bench_names:
            logger.info("=== Benchmark: %s ===", bench_name)
            if bench_name not in BENCHMARKS:
                logger.warning("Unknown benchmark: %s, skipping", bench_name)
                continue
            bench = BENCHMARKS[bench_name](cfg)
            all_results[bench_name] = {}
            recording_paths[bench_name] = []

            for model_name in model_names:
                if model_name not in MODELS:
                    logger.warning("Unknown model: %s, skipping", model_name)
                    continue
                model = MODELS[model_name](cfg)
                seed_results: list[dict[str, Any]] = []
                logger.info("  Model: %s", model_name)

                for seed in seeds:
                    logger.info("    Seed: %d", seed)
                    kwargs: dict[str, Any] = {}
                    recording = None

                    if model.requires_neurons:
                        kwargs["neurons"] = neurons
                        if cfg["bnn"].get("record", False):
                            recording = neurons.record(
                                file_suffix=f"{bench_name}_seed{seed}",
                                file_location=str(recordings_dir),
                            )

                    t0 = time.time()
                    result = bench.run(model, seed, **kwargs)
                    result["wall_time_sec"] = time.time() - t0

                    if recording is not None:
                        recording.stop()
                        recording.wait_until_stopped()
                        rec_path = str(recordings_dir / f"*{bench_name}_seed{seed}*.h5")
                        # Find the actual file
                        matches = list(recordings_dir.glob(f"*{bench_name}_seed{seed}*.h5"))
                        if matches:
                            rec_path = str(matches[0])
                            recording_paths[bench_name].append(rec_path)
                            result["recording_path"] = rec_path

                    seed_results.append(result)

                all_results[bench_name][model_name] = seed_results

    # Post-hoc neuro analysis on BNN recordings
    for bench_name, paths in recording_paths.items():
        bench_neuro: list[dict[str, Any]] = []
        for path in paths:
            try:
                metrics = analyse_recording(path, cfg)
                bench_neuro.append(metrics)
            except Exception as e:
                logger.warning("Neuro analysis failed for %s: %s", path, e)
        if bench_neuro:
            neuro_results[bench_name] = bench_neuro

    # Generate plots
    logger.info("Generating plots...")
    for bench_name, model_results in all_results.items():
        # Learning curves for benchmarks that have them
        has_lc = any(
            "learning_curve" in r
            for seeds_r in model_results.values()
            for r in seeds_r
        )
        if has_lc:
            plot_learning_curves(model_results, bench_name, plots_dir)

    if "noise" in all_results:
        plot_noise_degradation(all_results["noise"], plots_dir)
    if "sample_efficiency" in all_results:
        plot_sample_efficiency(all_results["sample_efficiency"], plots_dir)
    if "complexity" in all_results:
        plot_complexity_trajectory(all_results["complexity"], plots_dir)
        plot_consciousness_indicators(all_results["complexity"], plots_dir)
        plot_flicker_raster(all_results["complexity"], plots_dir)
        plot_complexity_phase_portrait(all_results["complexity"], plots_dir)

    # Summary radar
    summary = _build_summary(all_results)
    if summary:
        plot_summary_radar(summary, plots_dir)

    # Generate markdown report
    logger.info("Generating report...")
    generate_report(all_results, neuro_results or None, results_dir)

    # Save raw results JSON
    results_json = _serialisable(all_results)
    results_path = results_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    logger.info("Results saved to %s", results_path)

    return all_results


def _build_summary(
    all_results: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, dict[str, float]]:
    """Build normalised 0-1 scores for the radar chart."""
    metric_map = {
        "classification": ("final_accuracy", False),
        "temporal": ("sequence_accuracy", False),
        "adaptation": ("adaptation_ratio", True),   # lower is better
        "noise": ("noise_tolerance_threshold", False),
        "sample_efficiency": ("samples_to_competence", True),  # lower is better
        "continual": ("forgetting_rate", True),  # lower is better
        "complexity": ("consciousness_score", False),
    }

    raw: dict[str, dict[str, float]] = {}
    for bench_name, (key, invert) in metric_map.items():
        if bench_name not in all_results:
            continue
        for model_name, seed_results in all_results[bench_name].items():
            agg = aggregate_metric(seed_results, key)
            if agg["mean"] is None:
                continue
            raw.setdefault(model_name, {})[bench_name] = agg["mean"]

    if not raw:
        return {}

    # Normalise each benchmark to 0-1 across models
    all_benches = set()
    for m in raw.values():
        all_benches.update(m.keys())

    summary: dict[str, dict[str, float]] = {}
    for bench in all_benches:
        vals = {m: raw[m][bench] for m in raw if bench in raw[m]}
        if not vals:
            continue
        invert = metric_map.get(bench, (None, False))[1]
        vmin, vmax = min(vals.values()), max(vals.values())
        rng = vmax - vmin if vmax != vmin else 1.0
        for m, v in vals.items():
            normed = (v - vmin) / rng
            if invert:
                normed = 1.0 - normed
            summary.setdefault(m, {})[bench] = normed

    return summary


def _serialisable(obj: Any) -> Any:
    """Convert numpy types for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialisable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
