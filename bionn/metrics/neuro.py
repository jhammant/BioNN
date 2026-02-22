"""Neuroscience metrics extracted from CL SDK recordings."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def analyse_recording(file_path: str, cfg: dict) -> dict[str, Any]:
    """Run all CL SDK analyses on a recording file. Returns a flat dict of metrics."""
    import cl

    na = cfg["neuro_analysis"]
    rv = cl.RecordingView(file_path)
    metrics: dict[str, Any] = {}

    try:
        fs = rv.analyse_firing_stats(bin_size_sec=na["firing_stats_bin_sec"])
        metrics["mean_firing_rate"] = fs.culture_mean_firing_rates
        metrics["firing_rate_var"] = fs.culture_var_firing_rates
        metrics["isi_mean"] = fs.culture_ISI_mean
        metrics["isi_var"] = fs.culture_ISI_var
    except Exception as e:
        logger.warning("Firing stats failed: %s", e)

    try:
        nb = rv.analyse_network_bursts(
            bin_size_sec=na["burst_bin_sec"],
            onset_freq_hz=na["burst_onset_hz"],
            offset_freq_hz=na["burst_offset_hz"],
        )
        metrics["burst_count"] = nb.network_burst_count
        metrics["total_burst_duration_sec"] = nb.total_network_burst_duration_sec
        if nb.network_burst_durations_sec:
            metrics["mean_burst_duration_sec"] = float(sum(nb.network_burst_durations_sec) / len(nb.network_burst_durations_sec))
        else:
            metrics["mean_burst_duration_sec"] = 0.0
    except Exception as e:
        logger.warning("Network bursts failed: %s", e)

    try:
        fc = rv.analyse_functional_connectivity(
            bin_size_sec=na["connectivity_bin_sec"],
            correlation_threshold=na["connectivity_threshold"],
        )
        metrics["clustering_coefficient"] = fc.clustering_coefficient
        metrics["modularity"] = fc.modularity_index
        metrics["avg_edge_weight"] = fc.average_edge_weights
    except Exception as e:
        logger.warning("Functional connectivity failed: %s", e)

    try:
        lz = rv.analyse_lempel_ziv_complexity(bin_size_sec=na["lzc_bin_sec"])
        scores = lz.lzc_scores_per_channel
        if hasattr(scores, "__len__") and len(scores) > 0:
            import numpy as np
            metrics["lempel_ziv_complexity"] = float(np.nanmean(scores))
        else:
            metrics["lempel_ziv_complexity"] = None
    except Exception as e:
        logger.warning("Lempel-Ziv failed: %s", e)

    try:
        ie = rv.analyse_information_entropy(bin_size_sec=na["entropy_bin_sec"])
        vals = ie.information_entropy_per_time_bin
        if hasattr(vals, "__len__") and len(vals) > 0:
            import numpy as np
            metrics["information_entropy"] = float(np.nanmean(vals))
        else:
            metrics["information_entropy"] = None
    except Exception as e:
        logger.warning("Information entropy failed: %s", e)

    try:
        cr = rv.analyse_criticality(
            bin_size_sec=na["criticality_bin_sec"],
            percentile_threshold=na["criticality_percentile"],
        )
        metrics["branching_ratio"] = float(cr.branching_ratio) if cr.branching_ratio is not None else None
        metrics["dcc"] = float(cr.deviation_from_criticality_coefficient) if cr.deviation_from_criticality_coefficient is not None else None
    except Exception as e:
        logger.warning("Criticality failed: %s", e)

    rv.close()
    return metrics
