"""Consciousness benchmark — tests for consciousness-adjacent signatures.

This benchmark implements five tests inspired by neuroscience theories of consciousness:

1. Perturbational Complexity Index (PCI-like): Measures the complexity of responses
   to perturbation, inspired by Integrated Information Theory
2. Spontaneous Complexity: Measures intrinsic complexity without stimulation
3. Integrated Information (Φ-lite): Simplified measure of information integration
4. Criticality Check: Tests for critical dynamics (branching ratio ≈ 1.0)
5. Causal Density: Measures the density of causal connections in the network

These tests aim to differentiate biological neural networks from artificial ones
based on signatures associated with consciousness and complex information processing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bionn.benchmarks.base import BaseBenchmark
from bionn.models.base import BaseModel


def lempel_ziv_complexity(sequence: np.ndarray) -> float:
    """
    Compute normalized Lempel-Ziv complexity of a binary sequence.
    
    Uses the LZ76 algorithm to count distinct substrings, then normalizes
    by the theoretical maximum complexity for a sequence of this length.
    
    Args:
        sequence: Binary array or array that can be binarized
        
    Returns:
        Normalized complexity score between 0 and 1
    """
    # Convert to binary string
    binary = ''.join('1' if x > 0 else '0' for x in sequence.flatten())
    n = len(binary)
    
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
        
    # LZ76 complexity calculation
    complexity = 1
    i = 0
    
    while i < n:
        # Find longest match in the already processed prefix
        max_len = 0
        for j in range(i):
            match_len = 0
            while (i + match_len < n and 
                   j + match_len < i and 
                   binary[i + match_len] == binary[j + match_len]):
                match_len += 1
            max_len = max(max_len, match_len)
        
        # Move forward by at least one character
        i += max(1, max_len)
        complexity += 1
    
    # Normalize by theoretical maximum
    if n <= 1:
        return 1.0
    max_complexity = n / np.log2(n)
    return min(1.0, complexity / max_complexity)


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """
    Compute mutual information between two continuous variables.
    
    Args:
        x, y: Input arrays
        bins: Number of bins for discretization
        
    Returns:
        Mutual information in bits
    """
    if len(x) != len(y):
        return 0.0
    
    # Create joint histogram
    joint_hist, _, _ = np.histogram2d(x, y, bins=bins)
    joint_hist = joint_hist + 1e-8  # Avoid log(0)
    joint_prob = joint_hist / joint_hist.sum()
    
    # Marginal probabilities
    px = joint_prob.sum(axis=1)
    py = joint_prob.sum(axis=0)
    
    # Compute MI
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if joint_prob[i, j] > 0:
                mi += joint_prob[i, j] * np.log2(joint_prob[i, j] / (px[i] * py[j]))
                
    return mi


class ConsciousnessBenchmark(BaseBenchmark):
    """
    Consciousness benchmark testing for consciousness-adjacent signatures.
    
    This benchmark implements five complementary tests:
    - PCI: Perturbational complexity index
    - Spontaneous: Intrinsic activity complexity  
    - Phi-lite: Simplified integrated information
    - Criticality: Critical dynamics (branching ratio)
    - Causal density: Network causal connectivity
    """
    
    name = "consciousness"

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        cc = cfg["benchmarks"]["consciousness"]
        
        # PCI parameters
        self.pci_stim_channels = cc["pci_stim_channels"]
        self.pci_stim_amplitude = cc["pci_stim_amplitude"]
        self.pci_response_ms = cc["pci_response_ms"]
        self.pci_bin_ms = cc["pci_bin_ms"]
        self.pci_repeats = cc["pci_repeats"]
        
        # Spontaneous complexity parameters
        self.spontaneous_duration_ms = cc["spontaneous_duration_ms"]
        
        # Integrated information parameters
        self.phi_partitions = cc["phi_partitions"]
        self.phi_stim_trials = cc["phi_stim_trials"]
        
        # Criticality parameters
        self.criticality_duration_ms = cc["criticality_duration_ms"]
        self.criticality_bin_ms = cc["criticality_bin_ms"]
        
        # Causal density parameters
        self.causal_stim_channels = cc["causal_stim_channels"]
        self.causal_response_ms = cc["causal_response_ms"]
        self.causal_significance_threshold = cc["causal_significance_threshold"]

    def run(self, model: BaseModel, seed: int, **kwargs) -> dict[str, Any]:
        """Execute all consciousness tests for the given model."""
        results = {}
        
        # Test 1: Perturbational Complexity Index
        pci_result = self._test_pci(model, seed, **kwargs)
        results.update(pci_result)
        
        # Test 2: Spontaneous Complexity
        spont_result = self._test_spontaneous_complexity(model, seed, **kwargs)
        results.update(spont_result)
        
        # Test 3: Integrated Information (Phi-lite)
        phi_result = self._test_phi_lite(model, seed, **kwargs)
        results.update(phi_result)
        
        # Test 4: Criticality Check
        crit_result = self._test_criticality(model, seed, **kwargs)
        results.update(crit_result)
        
        # Test 5: Causal Density
        causal_result = self._test_causal_density(model, seed, **kwargs)
        results.update(causal_result)
        
        return results

    def _test_pci(self, model: BaseModel, seed: int, **kwargs) -> dict[str, Any]:
        """Test 1: Perturbational Complexity Index."""
        rng = np.random.RandomState(seed)
        
        if model.requires_neurons:
            neurons = kwargs["neurons"]
            pci_scores = []
            
            for _ in range(self.pci_repeats):
                # Create strong stimulation pattern
                pattern = np.zeros(self.n_channels)
                for ch in self.pci_stim_channels:
                    if ch < self.n_channels:
                        pattern[ch] = 1.0
                
                # Apply stimulation and record response
                spike_data = self._record_bnn_response(
                    neurons, pattern, self.pci_response_ms, self.pci_stim_amplitude
                )
                
                # Binarize spike train
                bins_per_ms = 1000 // self.pci_bin_ms
                n_bins = (self.pci_response_ms * bins_per_ms) // 1000
                binary_response = np.zeros(64 * n_bins)  # 64 readout channels
                
                for i, spikes in enumerate(spike_data):
                    if i < 64:  # Only use readout channels
                        for t in spikes:
                            bin_idx = min(int(t * bins_per_ms / 1000), n_bins - 1)
                            binary_response[i * n_bins + bin_idx] = 1
                
                # Compute LZC
                pci_scores.append(lempel_ziv_complexity(binary_response))
            
            pci_mean = np.mean(pci_scores)
            
        else:
            # For non-BNN models, use hidden layer activations
            pattern = np.zeros(self.n_channels)
            for ch in self.pci_stim_channels:
                if ch < self.n_channels:
                    pattern[ch] = 1.0
            
            pci_scores = []
            for _ in range(self.pci_repeats):
                # Get model response (need to access internals)
                if hasattr(model, '_forward'):
                    activations = model._forward(pattern)
                    # Binarize by threshold (mean activation)
                    threshold = np.mean(activations)
                    binary_act = (activations > threshold).astype(int)
                    pci_scores.append(lempel_ziv_complexity(binary_act))
                else:
                    # Fallback: just use prediction as simple binary response
                    pred = model.predict(pattern, **kwargs)
                    binary_pred = np.zeros(self.n_patterns)
                    binary_pred[pred] = 1
                    pci_scores.append(lempel_ziv_complexity(binary_pred))
            
            pci_mean = np.mean(pci_scores)
        
        return {f"{model.name}_pci": pci_mean}

    def _test_spontaneous_complexity(self, model: BaseModel, seed: int, **kwargs) -> dict[str, Any]:
        """Test 2: Spontaneous Complexity (no stimulation)."""
        if model.requires_neurons:
            neurons = kwargs["neurons"]
            
            # Record spontaneous activity
            spike_data = self._record_bnn_spontaneous(neurons, self.spontaneous_duration_ms)
            
            # Binarize full recording
            bins_per_ms = 10  # 10ms bins
            n_bins = self.spontaneous_duration_ms // bins_per_ms
            binary_spont = np.zeros(64 * n_bins)
            
            for i, spikes in enumerate(spike_data):
                if i < 64:
                    for t in spikes:
                        bin_idx = min(int(t / bins_per_ms), n_bins - 1)
                        binary_spont[i * n_bins + bin_idx] = 1
            
            spont_lzc = lempel_ziv_complexity(binary_spont)
            
        else:
            # Non-BNN models have no spontaneous activity without input
            spont_lzc = 0.0
        
        return {f"{model.name}_spontaneous_lzc": spont_lzc}

    def _test_phi_lite(self, model: BaseModel, seed: int, **kwargs) -> dict[str, Any]:
        """Test 3: Integrated Information (simplified Phi)."""
        rng = np.random.RandomState(seed)
        patterns = self._make_patterns(rng)
        
        if model.requires_neurons:
            neurons = kwargs["neurons"]
            phi_scores = []
            
            for _ in range(self.phi_partitions):
                left_counts = []
                right_counts = []
                
                for trial in range(self.phi_stim_trials):
                    pattern = patterns[trial % len(patterns)]
                    spike_data = self._record_bnn_response(neurons, pattern, 200)
                    
                    # Count spikes in left and right halves
                    left_spikes = sum(len(spike_data[i]) for i in range(32) if i < len(spike_data))
                    right_spikes = sum(len(spike_data[i]) for i in range(32, 64) if i < len(spike_data))
                    
                    left_counts.append(left_spikes)
                    right_counts.append(right_spikes)
                
                # Compute correlation between halves
                if len(left_counts) > 1 and np.std(left_counts) > 0 and np.std(right_counts) > 0:
                    phi_scores.append(np.corrcoef(left_counts, right_counts)[0, 1])
                else:
                    phi_scores.append(0.0)
            
            phi_lite = np.mean([abs(s) for s in phi_scores if not np.isnan(s)])
            
        else:
            # For non-BNN models, use hidden layer correlations
            phi_scores = []
            
            for trial in range(self.phi_stim_trials):
                pattern = patterns[trial % len(patterns)]
                
                if hasattr(model, '_forward'):
                    activations = model._forward(pattern)
                    if len(activations) >= 2:
                        mid = len(activations) // 2
                        left_act = activations[:mid]
                        right_act = activations[mid:]
                        
                        if len(left_act) > 0 and len(right_act) > 0:
                            # Use mean activations as proxy
                            left_mean = np.mean(left_act)
                            right_mean = np.mean(right_act)
                            phi_scores.extend([left_mean, right_mean])
            
            if len(phi_scores) >= 4:
                left_means = phi_scores[::2]
                right_means = phi_scores[1::2]
                if np.std(left_means) > 0 and np.std(right_means) > 0:
                    phi_lite = abs(np.corrcoef(left_means, right_means)[0, 1])
                else:
                    phi_lite = 0.0
            else:
                phi_lite = 0.0
        
        return {f"{model.name}_phi_lite": phi_lite}

    def _test_criticality(self, model: BaseModel, seed: int, **kwargs) -> dict[str, Any]:
        """Test 4: Criticality (branching ratio)."""
        if model.requires_neurons:
            neurons = kwargs["neurons"]
            
            # Record longer spontaneous activity for criticality analysis
            spike_data = self._record_bnn_spontaneous(neurons, self.criticality_duration_ms)
            
            # Count spikes in time bins
            bin_size = self.criticality_bin_ms
            n_bins = self.criticality_duration_ms // bin_size
            spike_counts = np.zeros(n_bins)
            
            for spikes in spike_data:
                for t in spikes:
                    bin_idx = min(int(t / bin_size), n_bins - 1)
                    spike_counts[bin_idx] += 1
            
            # Compute branching ratio
            ratios = []
            for i in range(len(spike_counts) - 1):
                if spike_counts[i] > 0:
                    ratios.append(spike_counts[i + 1] / spike_counts[i])
            
            branching_ratio = np.mean(ratios) if ratios else 0.0
            
        else:
            # For non-BNN models, estimate from activation propagation
            rng = np.random.RandomState(seed)
            patterns = self._make_patterns(rng)
            
            activation_ratios = []
            for pattern in patterns[:10]:  # Sample subset
                if hasattr(model, '_forward'):
                    act1 = model._forward(pattern)
                    # Add small perturbation and measure response
                    perturbed = pattern + rng.normal(0, 0.01, pattern.shape)
                    act2 = model._forward(perturbed)
                    
                    # Measure propagation ratio
                    if np.sum(act1) > 0:
                        ratio = np.sum(act2) / np.sum(act1)
                        activation_ratios.append(ratio)
            
            branching_ratio = np.mean(activation_ratios) if activation_ratios else 1.0
        
        return {f"{model.name}_branching_ratio": branching_ratio}

    def _test_causal_density(self, model: BaseModel, seed: int, **kwargs) -> dict[str, Any]:
        """Test 5: Causal Density."""
        if model.requires_neurons:
            neurons = kwargs["neurons"]
            
            causal_matrix = np.zeros((self.causal_stim_channels, 64))
            
            for stim_ch in range(min(self.causal_stim_channels, self.n_channels)):
                # Create single-channel stimulation
                pattern = np.zeros(self.n_channels)
                pattern[stim_ch] = 1.0
                
                # Record response
                spike_data = self._record_bnn_response(
                    neurons, pattern, self.causal_response_ms
                )
                
                # Count responses in each readout channel
                for read_ch, spikes in enumerate(spike_data):
                    if read_ch < 64:
                        causal_matrix[stim_ch, read_ch] = len(spikes)
            
            # Compute causal density (fraction of significant connections)
            threshold = np.mean(causal_matrix) + 2 * np.std(causal_matrix)
            significant_connections = np.sum(causal_matrix > threshold)
            total_possible = self.causal_stim_channels * 64
            causal_density = significant_connections / total_possible if total_possible > 0 else 0.0
            
        else:
            # For non-BNN models, compute Jacobian-like sensitivity
            if hasattr(model, '_forward'):
                rng = np.random.RandomState(seed)
                causal_influences = []
                
                for stim_ch in range(min(self.causal_stim_channels, self.n_channels)):
                    base_pattern = rng.uniform(0, 0.1, self.n_channels)
                    base_response = model._forward(base_pattern)
                    
                    # Perturb single channel
                    perturb_pattern = base_pattern.copy()
                    perturb_pattern[stim_ch] += 0.5
                    perturb_response = model._forward(perturb_pattern)
                    
                    # Measure influence
                    influence = np.sum(np.abs(perturb_response - base_response))
                    causal_influences.append(influence)
                
                # Density as fraction of above-threshold influences
                threshold = np.mean(causal_influences) if causal_influences else 0
                causal_density = np.mean([inf > threshold for inf in causal_influences])
            else:
                causal_density = 0.0
        
        return {f"{model.name}_causal_density": causal_density}

    def _record_bnn_response(self, neurons, pattern: np.ndarray, duration_ms: int, 
                           stim_amplitude: int = None) -> list[list[float]]:
        """Record BNN response to stimulation pattern."""
        if stim_amplitude is None:
            stim_amplitude = self.pci_stim_amplitude
            
        # Apply stimulation
        for ch in range(len(pattern)):
            if pattern[ch] > 0.3:  # Same threshold as BNN model
                import cl
                neurons.stim(ch, cl.StimDesign(stim_amplitude, -1, stim_amplitude, 1))
        
        # Record response
        spike_data = [[] for _ in range(64)]  # 64 readout channels
        ticks_per_ms = 1  # 1000 Hz = 1 tick per ms
        total_ticks = duration_ms * ticks_per_ms
        
        for tick in neurons.loop(ticks_per_second=1000, stop_after_ticks=total_ticks):
            for spike in tick.analysis.spikes:
                if spike.channel < 64:
                    spike_data[spike.channel].append(tick.elapsed_ms)
        
        return spike_data

    def _record_bnn_spontaneous(self, neurons, duration_ms: int) -> list[list[float]]:
        """Record spontaneous BNN activity (no stimulation)."""
        spike_data = [[] for _ in range(64)]
        ticks_per_ms = 1
        total_ticks = duration_ms * ticks_per_ms
        
        for tick in neurons.loop(ticks_per_second=1000, stop_after_ticks=total_ticks):
            for spike in tick.analysis.spikes:
                if spike.channel < 64:
                    spike_data[spike.channel].append(tick.elapsed_ms)
        
        return spike_data