"""Benchmark registry."""

from bionn.benchmarks.adaptation import AdaptationBenchmark
from bionn.benchmarks.classification import ClassificationBenchmark
from bionn.benchmarks.complexity import ComplexityBenchmark
from bionn.benchmarks.continual import ContinualBenchmark
from bionn.benchmarks.noise import NoiseBenchmark
from bionn.benchmarks.sample_efficiency import SampleEfficiencyBenchmark
from bionn.benchmarks.temporal import TemporalBenchmark

BENCHMARKS: dict[str, type] = {
    "classification": ClassificationBenchmark,
    "temporal": TemporalBenchmark,
    "adaptation": AdaptationBenchmark,
    "noise": NoiseBenchmark,
    "sample_efficiency": SampleEfficiencyBenchmark,
    "continual": ContinualBenchmark,
    "complexity": ComplexityBenchmark,
}
