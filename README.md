# BioNN Benchmark Suite

Comprehensive benchmark suite comparing **biological neural networks** (Cortical Labs DishBrain / CL SDK simulator) against conventional AI baselines. Designed so the same code runs on both the simulator and real DishBrain hardware without modification.

## Quick Start

```bash
# Install dependencies (inside the project venv)
pip install -e .

# Run the full suite on the simulator (accelerated time)
CL_SDK_ACCELERATED_TIME=1 python scripts/run_all.py

# Run with custom config overrides
CL_SDK_ACCELERATED_TIME=1 python scripts/run_all.py -c config/my_overrides.yaml

# Verbose logging
CL_SDK_ACCELERATED_TIME=1 python scripts/run_all.py -v
```

On DishBrain hardware, omit `CL_SDK_ACCELERATED_TIME` — the SDK runs at real-time automatically.

## Models

| Model | Description | `requires_neurons` |
|-------|------------|-------------------|
| **BNN** | Biological neural network via CL SDK. Stimulates input channels, reads spike counts, trains a readout layer with Hebbian-style updates. | Yes |
| **MLP** | Two-layer NumPy MLP (8→16→4) with softmax + backprop. | No |
| **SNN** | Leaky integrate-and-fire spiking network (pure NumPy). 32 hidden LIF neurons, surrogate gradient training. | No |
| **Online** | Single-layer online SGD softmax baseline. | No |

## Benchmarks

### 1. Classification
4-class pattern discrimination. Each pattern has 8 channels with 2 "hot" channels.
- **Metric**: Epochs to 75% accuracy, final accuracy, learning curve AUC

### 2. Temporal
Predict the next pattern in a cyclic sequence (A→B→C→D→A→...).
- **Metric**: Sequence prediction accuracy

### 3. Adaptation
Train on original label mapping, then shuffle labels and re-train.
- **Metric**: Re-adaptation speed ratio (re-adapt epochs / initial epochs)

### 4. Noise Robustness
Train on clean data, evaluate at increasing Gaussian noise levels [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0].
- **Metric**: Noise tolerance threshold (highest sigma where accuracy >= 50%)

### 5. Sample Efficiency
Vary training budget [1, 2, 4, 8, 16, 32, 64] samples per class.
- **Metric**: Samples to competence (>= 60% accuracy)

### 6. Continual Learning
Train Task A, then Task B (different patterns), re-evaluate Task A.
- **Metric**: Forgetting rate = (A_before - A_after) / A_before

## Neuroscience Metrics (BNN only)

When `bnn.record: true` (default), HDF5 recordings are saved and analysed post-hoc:

- **Firing statistics** — mean firing rate, ISI
- **Network bursts** — count, duration
- **Functional connectivity** — clustering coefficient, modularity
- **Lempel-Ziv complexity**
- **Information entropy**
- **Criticality** — branching ratio, deviation from criticality coefficient (DCC)

## Configuration

All experiment parameters live in `config/default.yaml`. Override any value by creating a custom YAML file and passing it with `-c`:

```yaml
# my_overrides.yaml — run fewer seeds, skip SNN
general:
  seeds: [42, 123]
models:
  - mlp
  - bnn
```

## Output

After a run, `results/` contains:

```text
results/
├── results.json          # Raw metrics for all model x benchmark x seed
├── report.md             # Auto-generated markdown report with tables
├── plots/
│   ├── classification_learning_curves.png
│   ├── temporal_learning_curves.png
│   ├── adaptation_learning_curves.png
│   ├── noise_degradation.png
│   ├── sample_efficiency.png
│   └── summary_radar.png
└── recordings/
    └── *.h5              # CL SDK recordings (BNN runs)
```

## Statistical Rigour

- **5 seeds** by default (configurable)
- **95% confidence intervals** on all aggregated metrics (via scipy t-distribution)
- **Significance tests** between model pairs (Welch's t-test)

## Project Structure

```text
BioNN/
├── config/default.yaml       # Full experiment configuration
├── bionn/
│   ├── config.py             # YAML config loader
│   ├── runner.py             # Main orchestrator
│   ├── benchmarks/           # 6 benchmark implementations
│   │   ├── base.py           # ABC + shared utilities
│   │   ├── classification.py
│   │   ├── temporal.py
│   │   ├── adaptation.py
│   │   ├── noise.py
│   │   ├── sample_efficiency.py
│   │   └── continual.py
│   ├── models/               # 4 model implementations
│   │   ├── base.py           # ABC
│   │   ├── bnn.py            # CL SDK wrapper
│   │   ├── mlp.py            # NumPy MLP
│   │   ├── snn.py            # LIF SNN
│   │   └── online.py         # Online SGD
│   ├── metrics/
│   │   ├── task.py           # Accuracy, CIs, significance tests
│   │   └── neuro.py          # CL SDK recording analysis
│   └── reporting/
│       ├── plots.py          # Matplotlib visualisations
│       └── report.py         # Markdown report generator
├── scripts/
│   └── run_all.py            # CLI entry point
└── results/                  # Output directory (gitignored)
```

## Hardware Notes

- The BNN model uses a **single `cl.open()` context** shared across all benchmarks
- HDF5 recordings are written per benchmark+seed combination
- On hardware, remove `CL_SDK_ACCELERATED_TIME` — the system runs at biological real-time
- All models receive the same randomised pattern sequences per seed for fair comparison

## Dependencies

- `cl-sdk >= 0.29.0`
- `numpy >= 2.0`
- `scipy >= 1.14`
- `matplotlib >= 3.9`
- `pyyaml >= 6.0`
- `networkx >= 3.0` (for connectivity analysis)
- `python-louvain >= 0.16` (for modularity)
