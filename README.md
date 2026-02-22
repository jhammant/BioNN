# BioNN Benchmark Suite

Can living neurons learn? And if so, how do they compare to the algorithms we already have?

This project is a head-to-head benchmark suite that puts **biological neural networks** — real human neurons grown on silicon via [Cortical Labs](https://corticallabs.com/) DishBrain — up against three conventional AI baselines on the same set of tasks. The exact same code runs on the CL SDK simulator (for development) and on actual DishBrain hardware (for the real experiment). No code changes needed.

The goal is not to prove one is "better" — it's to find out **what biological neurons are uniquely good at** compared to gradient-based methods, and to surface that with rigorous, reproducible measurements that the neuroscience community can build on.

## The Big Questions This Suite Is Designed To Answer

1. **Can biological neurons learn to classify patterns at all?** And if so, how many training examples do they need compared to backprop?
2. **Do biological neurons adapt faster when the rules change?** Brains are supposedly great at re-learning — does that show up in a controlled benchmark?
3. **Are biological neurons more robust to noisy inputs?** Biological systems evolved in noisy environments. Do they degrade more gracefully than trained weight matrices?
4. **Can biological neurons learn from very few examples?** Few-shot learning is hard for conventional networks. Is it any easier for neurons?
5. **Do biological neurons resist catastrophic forgetting?** When you train on a new task, do they forget the old one less than artificial networks?
6. **Can biological neurons learn temporal sequences?** Not just static classification, but predicting what comes next.

## What Is DishBrain / CL SDK?

[Cortical Labs](https://corticallabs.com/) grows human neurons on multi-electrode arrays (MEAs). The CL SDK lets you stimulate these neurons with electrical patterns and read back their spiking activity in real-time. The neurons self-organise, form connections, and — the hypothesis — learn from experience through biological plasticity mechanisms (synaptic strengthening, pruning, structural changes) that have evolved over millions of years.

The SDK provides:
- **Stimulation** — send voltage patterns to specific electrodes
- **Spike readout** — detect when neurons fire and on which channels
- **Recording** — save full sessions as HDF5 files for post-hoc neuroscience analysis
- **A simulator** — mock API that generates synthetic spike data, so you can develop and test without hardware

This suite wraps all of that into a clean benchmark framework.

## How It Works

### The Stimulation-Response Loop

Every model in the suite follows the same trial structure. On each trial:

1. A **pattern** is selected (one of 4 classes)
2. The pattern is an 8-dimensional vector where values 0-1 represent activation levels
3. Each class has a distinct "signature" — 2 out of 8 channels are "hot" (0.7-1.0), the rest are background noise (0.05-0.2)
4. The model sees the pattern and must predict which of the 4 classes it belongs to
5. The model gets feedback (correct/incorrect) and updates its internal state

For the **BNN (biological neural network)**, step 2-4 looks like this:
- Channels with activation above 0.3 get **electrical stimulation** via `neurons.stim()`. The amplitude is proportional to the pattern value (20-200 mV, quantised to 20 mV steps)
- The system waits 30 milliseconds (30 ticks at 1000 Hz) for the neurons to respond
- **Spike counts** are collected across 64 readout channels — this is the biological network's "output"
- The spike count vector is multiplied by a learned **readout weight matrix** (64 x 4) to produce a class prediction
- Weights are updated with a **Hebbian-style rule**: correct predictions strengthen active-channel-to-target weights; incorrect predictions weaken active-channel-to-wrong-class weights and mildly strengthen the correct class

The key insight: the biological network itself is a black box. We don't control the synaptic weights between neurons — we only control the readout layer that interprets the network's spiking response. The learning that happens inside the biological network (if any) is entirely driven by the neurons' own plasticity mechanisms.

### The Baselines

The BNN is compared against three conventional models, all receiving the exact same pattern sequences in the exact same order (controlled by seed):

| Model | Architecture | Training | Why It's Included |
|-------|-------------|----------|-------------------|
| **MLP** | 2-layer network (8→16→4), ReLU hidden, softmax output | Full backpropagation, lr=0.1 | Gold standard — the "best case" for this task size |
| **SNN** | 32 leaky integrate-and-fire neurons, 30 timestep simulation | Surrogate gradient descent, lr=0.05 | Closer to biological dynamics than MLP, but still uses gradient-based learning |
| **Online SGD** | Single-layer softmax (8→4), no hidden layer | Online stochastic gradient descent, lr=0.15 | Minimal baseline — if this beats the BNN, the BNN isn't learning |

The SNN is particularly interesting as a comparison because it mimics biological neuron dynamics (membrane potential, spike threshold, leak) but trains with backprop. If the SNN outperforms the BNN on everything, it suggests the biological dynamics help but the learning algorithm matters more. If the BNN beats the SNN on specific tasks (e.g., adaptation, noise robustness), that's evidence for biological plasticity doing something that artificial training can't.

## The Six Benchmarks In Detail

### 1. Classification — "Can you learn to tell patterns apart?"

The simplest test. 4 classes, each with a distinct activation pattern. Train for up to 15 epochs, 16 trials per epoch (4 per class, shuffled). Light Gaussian noise (sigma=0.05) is added to every trial to prevent trivial memorisation.

**What we measure:**
- `epochs_to_target` — How many epochs until the model first hits 75% accuracy in an epoch. Lower is faster learning.
- `final_accuracy` — Accuracy on the last epoch. Are you still improving or have you plateaued?
- `learning_curve` — Full epoch-by-epoch accuracy trace, plotted with confidence bands across seeds.

**What it tells us:** Basic learning capability. If a model can't do this, it can't do anything. The 75% target (well above 25% chance) is set low enough that even a weak learner should get there eventually.

**Default config:** 15 epochs, 16 trials/epoch, target 75%, training noise sigma=0.05.

### 2. Temporal Sequence — "Can you predict what comes next?"

A fixed cyclic sequence: pattern 0 → pattern 1 → pattern 2 → pattern 3 → pattern 0 → ... The model sees the current pattern and must predict the index of the next pattern in the sequence. This tests whether the network can learn input-output mappings that require temporal context — the response to pattern 0 should be "1", not "0".

**What we measure:**
- `sequence_accuracy` — After training, sweep through all 4 positions in the sequence and check if the model predicts the correct successor. 100% means it learned the full sequence.
- `learning_curve` — Epoch-by-epoch training accuracy over 20 epochs.
- `final_accuracy` — Last-epoch training accuracy.

**What it tells us:** This is really a remapped classification task (pattern A → class B), not true sequence memory. But it tests whether the model can learn arbitrary associations rather than just "pattern X means class X". Biological neurons might handle arbitrary remapping differently to gradient methods.

**Default config:** Sequence length 4, 20 epochs, 16 trials/epoch, noise sigma=0.05.

### 3. Adaptation — "Can you re-learn when the rules change?"

Two phases:
1. **Phase 1** — Train normally for 8 epochs (same as classification). Record when the model first hits 75%.
2. **Phase 2** — Randomly permute the class labels (e.g., what was class 0 is now class 3) and train for 10 more epochs. Record when the model re-hits 75%.

The model is NOT reset between phases — it must adapt its existing learned weights to the new mapping.

**What we measure:**
- `pre_epochs_to_target` — Epochs to 75% on the original mapping.
- `re_epochs_to_target` — Epochs to 75% on the shuffled mapping.
- `adaptation_ratio` — `re_epochs / pre_epochs`. A ratio of 1.0 means it re-learns just as fast. Above 1.0 means re-learning is harder (the old mapping interferes). Below 1.0 means re-learning is easier (positive transfer or rapid adaptation).

**What it tells us:** This is where biological neural networks might shine. Brains are famously good at re-learning — neural circuits can rapidly rewire through synaptic plasticity. Backprop-trained networks carry "momentum" from the old mapping that can slow re-learning. A BNN adaptation ratio significantly below the baselines would be strong evidence for biological re-adaptation advantage.

**Default config:** 8 pre-train epochs, 10 re-adapt epochs, 16 trials/epoch, target 75%.

### 4. Noise Robustness — "How much noise before you break?"

Train the model on clean(ish) data (sigma=0.05) for 8 epochs. Then evaluate at 7 noise levels: sigma = 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0. The noise is additive Gaussian, clipped to [0, 1]. At sigma=1.0, the noise magnitude equals the full signal range — the input is heavily corrupted.

**What we measure:**
- `accuracies` — Accuracy at each noise level, producing a degradation curve.
- `noise_tolerance_threshold` — The highest noise sigma where the model still achieves >= 50% accuracy. Higher is more robust.

**What it tells us:** Biological sensory systems evolved to handle noisy, incomplete inputs. If living neurons maintain classification accuracy at noise levels where trained weight matrices break down, that's meaningful evidence for biological noise resilience — potentially from population coding, redundant pathways, or attractor dynamics in the biological network.

**Default config:** 8 training epochs, 16 eval trials per noise level, noise levels [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0].

### 5. Sample Efficiency — "How many examples do you need?"

For each budget in [1, 2, 4, 8, 16, 32, 64] samples per class: reset the model, train on exactly that many examples per class, then evaluate on 16 test trials. The model is fully reset between each budget level — this measures learning-from-scratch at each data point, not cumulative learning.

**What we measure:**
- `accuracies` — Accuracy at each budget level, producing a data efficiency curve.
- `samples_to_competence` — The total number of training samples (budget x 4 classes) needed to first reach 60% accuracy. Lower is more data-efficient.

**What it tells us:** This is arguably the most interesting benchmark for biological networks. Brains are remarkably data-efficient — humans can learn to recognise a new object from one or two examples. If biological neurons show significantly higher accuracy at low training budgets (1-4 samples per class) than the baselines, that's evidence for biological priors or inductive biases that help with few-shot learning. The conventional baselines need enough data to shape their weight gradients — biology might not.

**Default config:** Budgets [1, 2, 4, 8, 16, 32, 64] per class, 16 eval trials, competence threshold 60%.

### 6. Continual Learning — "Do you forget what you learned?"

Three phases:
1. **Phase 1** — Train on Task A (a set of 4 patterns) for 8 epochs. Evaluate accuracy on Task A. Call this `A_before`.
2. **Phase 2** — Train on Task B (a completely different set of 4 patterns, same 4 class labels) for 8 epochs.
3. **Phase 3** — Evaluate accuracy on Task A again (without any re-training). Call this `A_after`. Also evaluate Task B.

Task A and Task B use independently generated pattern sets — they share the same class structure (4 classes, 8 channels) but different activation patterns.

**What we measure:**
- `task_a_accuracy_before` — How well the model learned Task A.
- `task_a_accuracy_after` — How well it remembers Task A after being trained on Task B.
- `task_b_accuracy` — How well it learned Task B.
- `forgetting_rate` — `(A_before - A_after) / A_before`. 0% means perfect retention. 100% means total amnesia. Negative means Task B training somehow improved Task A performance (positive backward transfer).

**What it tells us:** Catastrophic forgetting is one of the biggest unsolved problems in deep learning. When you train a neural network on a new task, it tends to overwrite the weights it learned for the old task. Biological brains don't do this (as much) — you can learn to ride a bicycle without forgetting how to walk. If the BNN shows a significantly lower forgetting rate than the baselines, that's evidence that biological neural dynamics naturally protect old memories during new learning — potentially through mechanisms like synaptic consolidation, sparse representations, or complementary learning systems.

**Default config:** 8 epochs Task A, 8 epochs Task B, 16 trials/epoch, noise sigma=0.05.

## Neuroscience Metrics

For every BNN run, the suite records the full neural activity to an HDF5 file and runs six post-hoc analyses using the CL SDK's `RecordingView`:

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Mean firing rate** | Average spikes per second across all channels | Basic measure of neural activity level. Too low = neurons aren't responding. Too high = possible seizure-like activity. |
| **Inter-spike interval (ISI)** | Time between consecutive spikes on the same channel | Reveals firing regularity. Low ISI variance = clock-like firing. High variance = bursty or irregular. |
| **Network burst count & duration** | Synchronised high-frequency firing across many channels simultaneously | Bursts indicate coordinated network activity. Changes in burst patterns during learning may reflect network reorganisation. |
| **Functional connectivity** | Cross-correlation between channel pairs → graph metrics (clustering coefficient, modularity) | Measures how strongly channels are functionally linked. Higher clustering = more structured network. Higher modularity = specialised sub-circuits. |
| **Lempel-Ziv complexity** | Algorithmic complexity of the binary spike train | Measures the "richness" of neural activity patterns. Low = repetitive. High = complex/random. Intermediate is often associated with healthy, information-processing neural activity. |
| **Information entropy** | Shannon entropy of binned spike counts over time | How unpredictable is the neural activity? Low entropy = stereotyped responses. High entropy = variable/random. |
| **Branching ratio** | Ratio of spikes in successive time bins during avalanches | A branching ratio near 1.0 indicates the network operates at **criticality** — the boundary between ordered and chaotic dynamics, which is theorised to optimise information processing. |
| **Deviation from criticality coefficient (DCC)** | Statistical deviation from power-law scaling in avalanche size/duration | Complementary to branching ratio. DCC near 0 = critical. Large positive/negative = sub/super-critical. |

On the simulator, connectivity and criticality analyses fail gracefully (the mock API doesn't model MEA geometry). On real DishBrain hardware, all six analyses produce meaningful data.

## Running

```bash
# Clone and set up
git clone https://github.com/jhammant/BioNN.git
cd BioNN
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Run on simulator (accelerated — takes ~2 minutes)
CL_SDK_ACCELERATED_TIME=1 python scripts/run_all.py

# Run on DishBrain hardware (real-time — takes longer)
python scripts/run_all.py

# Run with verbose logging
CL_SDK_ACCELERATED_TIME=1 python scripts/run_all.py -v

# Run with custom config overrides
CL_SDK_ACCELERATED_TIME=1 python scripts/run_all.py -c config/my_overrides.yaml
```

## Configuration

All parameters live in `config/default.yaml`. Create a separate YAML file with just the values you want to change and pass it with `-c` — it gets deep-merged on top of the defaults.

```yaml
# quick_test.yaml — fast iteration with fewer seeds and only 2 models
general:
  seeds: [42]
models:
  - mlp
  - bnn
benchmarks:
  classification:
    max_epochs: 5
```

Key parameters you might want to tune:

| Parameter | Default | What It Controls |
|-----------|---------|-----------------|
| `general.seeds` | `[42, 123, 456, 789, 1024]` | Random seeds for reproducibility. More seeds = tighter confidence intervals, longer runtime. |
| `general.num_channels` | `8` | Input dimensionality. Must match your MEA configuration on hardware. |
| `general.num_patterns` | `4` | Number of classes. More classes = harder task. |
| `bnn.stim_threshold` | `0.3` | Minimum activation to trigger stimulation. Lower = more channels get stimulated. |
| `bnn.stim_amplitude_max` | `200` | Maximum stimulation voltage in mV. |
| `bnn.read_ticks` | `30` | How long to wait for neural response (ms at 1000 Hz). Longer = more spikes but slower. |
| `bnn.record` | `true` | Whether to save HDF5 recordings for neuro analysis. Set to `false` to save disk space. |

## Output

After a run, `results/` contains:

```text
results/
├── results.json          # Raw metrics: all model x benchmark x seed combinations
├── report.md             # Auto-generated markdown report with tables and CIs
├── plots/
│   ├── classification_learning_curves.png
│   ├── temporal_learning_curves.png
│   ├── noise_degradation.png
│   ├── sample_efficiency.png
│   └── summary_radar.png
└── recordings/
    └── *.h5              # CL SDK HDF5 recordings (BNN runs only, 1 per benchmark x seed)
```

The `results.json` file contains every metric for every model/benchmark/seed combination. The `report.md` file has summary tables with means and 95% confidence intervals. The plots show learning curves with shaded CI bands, noise degradation curves, sample efficiency curves, and a radar chart comparing normalised scores.

## Statistical Rigour

- **5 seeds** by default — each model x benchmark combination runs 5 times with different random initialisations and pattern orderings
- **95% confidence intervals** on all aggregated metrics via scipy's t-distribution (appropriate for small sample sizes)
- **Welch's t-test** available for pairwise significance testing between models
- **Identical trial sequences** per seed — every model sees the exact same pattern presentations in the exact same order, ensuring fair comparison
- **Seeded RNG** for full reproducibility — same seed produces identical patterns, trial orders, weight initialisations, and noise realisations

## Architecture

```text
BioNN/
├── config/default.yaml       # All experiment parameters in one place
├── bionn/
│   ├── config.py             # YAML loader with deep merge for overrides
│   ├── runner.py             # Orchestrator: opens CL session, runs benchmarks, generates outputs
│   ├── benchmarks/
│   │   ├── base.py           # ABC + pattern generation + noise utilities
│   │   ├── classification.py # Benchmark 1
│   │   ├── temporal.py       # Benchmark 2
│   │   ├── adaptation.py     # Benchmark 3
│   │   ├── noise.py          # Benchmark 4
│   │   ├── sample_efficiency.py  # Benchmark 5
│   │   └── continual.py      # Benchmark 6
│   ├── models/
│   │   ├── base.py           # ABC: train_step(), predict(), reset(), requires_neurons flag
│   │   ├── bnn.py            # CL SDK wrapper (stimulate → read spikes → readout layer)
│   │   ├── mlp.py            # NumPy 2-layer MLP with backprop
│   │   ├── snn.py            # NumPy LIF spiking network
│   │   └── online.py         # Single-layer online SGD
│   ├── metrics/
│   │   ├── task.py           # CIs, AUC, significance tests
│   │   └── neuro.py          # CL SDK RecordingView analysis wrapper
│   └── reporting/
│       ├── plots.py          # Matplotlib: learning curves, degradation, radar
│       └── report.py         # Markdown table generator
├── scripts/
│   └── run_all.py            # CLI entry point
└── results/                  # Gitignored output directory
```

Key design decisions:
- **Single `cl.open()` context** — the CL SDK session is opened once and shared across all benchmarks. The `neurons` object is passed to the BNN model via `**kwargs`.
- **Uniform model interface** — every model implements `train_step(pattern, target) → correct`, `predict(pattern) → class`, `reset(seed)`. The runner doesn't need to know what's inside.
- **No external SNN library** — the LIF model is ~60 lines of pure NumPy. Zero extra dependencies for Cortical Labs to install.
- **HDF5 recording per benchmark+seed** — one recording file per BNN run, enabling fine-grained post-hoc analysis.
- **Registry pattern** — models and benchmarks are dicts in their `__init__.py`. Adding a new model or benchmark is: write the class, add one line to the registry.

## Dependencies

- `cl-sdk >= 0.29.0` — Cortical Labs SDK (simulator + hardware interface)
- `numpy >= 2.0` — array operations, all models except BNN are pure NumPy
- `scipy >= 1.14` — confidence intervals, significance tests
- `matplotlib >= 3.9` — plotting
- `pyyaml >= 6.0` — config loading
- `networkx >= 3.0` — functional connectivity graph analysis
- `python-louvain >= 0.16` — modularity computation for connectivity
