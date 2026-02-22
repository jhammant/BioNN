"""Auto-generated markdown report from benchmark results."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bionn.metrics.task import aggregate_metric


def _fmt(val: Any) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.3f}"
    return str(val)


def _fmt_ci(agg: dict[str, Any]) -> str:
    if agg["mean"] is None:
        return "N/A"
    mean = agg["mean"]
    if agg["ci_low"] is not None and agg["n"] > 1:
        return f"{mean:.3f} [{agg['ci_low']:.3f}, {agg['ci_high']:.3f}]"
    return f"{mean:.3f}"


def generate_report(
    all_results: dict[str, dict[str, list[dict[str, Any]]]],
    neuro_results: dict[str, list[dict[str, Any]]] | None,
    out_dir: Path,
) -> Path:
    """Write results/report.md summarising all benchmarks."""
    lines: list[str] = []
    lines.append("# BioNN Benchmark Report")
    lines.append(f"\nGenerated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")

    # Per-benchmark tables
    for bench_name, model_results in all_results.items():
        lines.append(f"## {bench_name.replace('_', ' ').title()}\n")
        models = list(model_results.keys())
        if not models:
            continue

        # Collect metric keys from first seed result
        sample = model_results[models[0]][0]
        scalar_keys = [k for k, v in sample.items() if isinstance(v, (int, float, type(None)))]

        if scalar_keys:
            header = "| Metric | " + " | ".join(m.upper() for m in models) + " |"
            sep = "|" + "|".join(["---"] * (len(models) + 1)) + "|"
            lines.append(header)
            lines.append(sep)

            for key in scalar_keys:
                row = f"| {key} |"
                for m in models:
                    agg = aggregate_metric(model_results[m], key)
                    row += f" {_fmt_ci(agg)} |"
                lines.append(row)
            lines.append("")

    # Neuro metrics table (BNN only)
    if neuro_results:
        lines.append("## Neuroscience Metrics (BNN)\n")
        for bench_name, seed_metrics in neuro_results.items():
            if not seed_metrics:
                continue
            lines.append(f"### {bench_name.replace('_', ' ').title()}\n")
            keys = [k for k in seed_metrics[0].keys() if isinstance(seed_metrics[0][k], (int, float, type(None)))]
            if keys:
                lines.append("| Metric | Mean |")
                lines.append("|---|---|")
                for k in keys:
                    agg = aggregate_metric(seed_metrics, k)
                    lines.append(f"| {k} | {_fmt_ci(agg)} |")
                lines.append("")

    # Plot references
    lines.append("## Figures\n")
    plot_dir = out_dir / "plots"
    if plot_dir.exists():
        for p in sorted(plot_dir.glob("*.png")):
            lines.append(f"![{p.stem}](plots/{p.name})\n")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    return report_path
