# Copyright 2025 - Mathematical Verification Tests for Dragon Hatching
# Rigorous tests to verify biological plausibility and mathematical correctness

import math
import pytest
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple

# Add parent to path for imports
import sys
sys.path.insert(0, '..')

from hatching import (
    HatchingConfig,
    LeakyIntegrateFire,
    STDPAttention,
    spike_function,
    power_law_sample,
    generate_scale_free_mask,
)
from hatching_advanced import (
    make_hippo_legs,
    make_hippo_legt,
    discretize_zoh,
    HiPPOCell,
    BCMHomeostasis,
    CriticalityRegulator,
)


# =============================================================================
# LEAKY INTEGRATE-AND-FIRE TESTS
# =============================================================================

class TestLIFDynamics:
    """
    Verify LIF neuron dynamics match theoretical predictions.

    Theory: τ_m · dV/dt = -(V - V_rest) + I

    Key properties to verify:
    1. Exponential decay to resting potential without input
    2. Correct time constant (63.2% decay in τ_m steps)
    3. Spike generation at threshold
    4. Reset mechanism
    """

    def test_membrane_decay_without_input(self):
        """
        Without input, membrane should decay exponentially to V_rest.

        V(t) = V_rest + (V_0 - V_rest) * exp(-t/τ_m)
        """
        config = HatchingConfig(tau_mem=10.0, v_rest=0.0, v_threshold=1.0)
        lif = LeakyIntegrateFire(config, n_neurons=100)

        # Start with elevated membrane potential
        V_0 = 0.5
        membrane = torch.full((1, 100), V_0)
        trace = torch.zeros(1, 100)

        # Simulate 50 timesteps with zero input
        zero_input = torch.zeros(1, 100)
        voltages = [V_0]

        for t in range(50):
            _, membrane, trace = lif(zero_input, membrane, trace)
            voltages.append(membrane.mean().item())

        # Theoretical prediction
        alpha = math.exp(-1.0 / config.tau_mem)
        theoretical = [V_0 * (alpha ** t) for t in range(51)]

        # Verify exponential decay
        for t in range(51):
            assert abs(voltages[t] - theoretical[t]) < 1e-5, \
                f"Decay mismatch at t={t}: got {voltages[t]:.6f}, expected {theoretical[t]:.6f}"

    def test_time_constant_definition(self):
        """
        Verify τ_m is the time for 63.2% decay (1 - 1/e).

        After τ_m timesteps: V(τ_m) = V_0 * exp(-1) ≈ 0.368 * V_0
        """
        config = HatchingConfig(tau_mem=10.0, v_rest=0.0)
        lif = LeakyIntegrateFire(config, n_neurons=1)

        V_0 = 1.0
        membrane = torch.full((1, 1), V_0)
        trace = torch.zeros(1, 1)
        zero_input = torch.zeros(1, 1)

        # Simulate exactly τ_m timesteps
        for _ in range(int(config.tau_mem)):
            _, membrane, trace = lif(zero_input, membrane, trace)

        # Should be approximately 1/e of initial
        expected = V_0 * math.exp(-1)
        actual = membrane.item()

        assert abs(actual - expected) < 0.01, \
            f"Time constant test failed: got {actual:.4f}, expected {expected:.4f}"

    def test_spike_threshold(self):
        """
        Verify spikes occur when membrane exceeds threshold.
        """
        config = HatchingConfig(tau_mem=10.0, v_threshold=1.0, v_rest=0.0)
        lif = LeakyIntegrateFire(config, n_neurons=10)

        # Strong input to push over threshold
        membrane = torch.full((1, 10), 0.0)
        trace = torch.zeros(1, 10)
        strong_input = torch.full((1, 10), 20.0)  # Very strong current

        spikes, membrane, trace = lif(strong_input, membrane, trace)

        # Should have generated spikes
        assert spikes.sum() > 0, "No spikes generated with strong input"

        # After spike, membrane should be reset
        spiked_neurons = spikes > 0
        assert (membrane[spiked_neurons] < config.v_threshold).all(), \
            "Membrane not reset after spike"

    def test_refractory_reset(self):
        """
        Verify reset mechanism: V → V_reset after spike.
        """
        config = HatchingConfig(v_threshold=1.0, v_reset=0.0, tau_mem=10.0)
        lif = LeakyIntegrateFire(config, n_neurons=1)

        # Set membrane just above threshold
        membrane = torch.tensor([[1.5]])
        trace = torch.zeros(1, 1)
        zero_input = torch.zeros(1, 1)

        spikes, membrane_after, _ = lif(zero_input, membrane, trace)

        # Should spike and reset
        assert spikes.item() == 1.0, "Should have spiked"
        # Soft reset: membrane - (threshold - reset) when spike
        # membrane_after ≈ membrane - threshold + reset
        expected_after = membrane.item() - (config.v_threshold - config.v_reset)
        assert abs(membrane_after.item() - expected_after) < 0.5, \
            f"Reset failed: got {membrane_after.item():.4f}"


class TestSurrogateGradient:
    """
    Verify surrogate gradient enables learning through spikes.
    """

    def test_gradient_flow(self):
        """
        Verify gradients flow through spike function.
        """
        x = torch.randn(10, requires_grad=True)
        threshold = 0.0

        spikes = spike_function(x, threshold)
        loss = spikes.sum()
        loss.backward()

        # Gradients should exist and be non-zero for values near threshold
        assert x.grad is not None, "No gradient computed"
        assert (x.grad != 0).any(), "All gradients are zero"

    def test_surrogate_shape(self):
        """
        Verify surrogate gradient has expected shape:
        - Maximum at threshold
        - Decays away from threshold
        """
        x = torch.linspace(-2, 2, 100, requires_grad=True)
        threshold = 0.0

        spikes = spike_function(x, threshold)

        # Compute gradient for each point
        grads = []
        for i in range(len(x)):
            if x.grad is not None:
                x.grad.zero_()
            spikes[i].backward(retain_graph=True)
            grads.append(x.grad[i].item())

        grads = np.array(grads)

        # Gradient should peak near threshold
        peak_idx = np.argmax(grads)
        assert 40 < peak_idx < 60, f"Peak not near threshold: idx={peak_idx}"


# =============================================================================
# STDP TESTS
# =============================================================================

class TestSTDPKernel:
    """
    Verify STDP kernel has correct temporal structure.

    Theory:
    - Pre-before-post (Δt > 0): LTP (positive weight change)
    - Post-before-pre (Δt < 0): LTD (negative weight change)
    - Exponential decay with time constants τ_+ and τ_-
    """

    def test_causal_ltp(self):
        """
        Verify positive weights for causal (pre-before-post) pairs.
        """
        config = HatchingConfig(tau_plus=20.0, A_plus=0.01)
        attn = STDPAttention(config)

        T = 32
        kernel = attn.compute_stdp_kernel(T, device=torch.device('cpu'))

        # Lower triangle (i > j, meaning post after pre) should be positive
        lower = torch.tril(kernel, diagonal=-1)
        assert (lower >= 0).all(), "Causal pairs should have non-negative weights"

    def test_exponential_decay(self):
        """
        Verify exponential decay with distance.

        K[i,j] ∝ exp(-|i-j|/τ) for i > j
        """
        config = HatchingConfig(tau_plus=20.0, A_plus=1.0)  # A_plus=1 for easy checking
        attn = STDPAttention(config)

        T = 64
        kernel = attn.compute_stdp_kernel(T, device=torch.device('cpu'))

        # Check decay along a row
        row = 50  # Check row 50
        for j in range(1, 10):
            delta_t = row - j
            expected_ratio = math.exp(-1.0 / config.tau_plus)  # Ratio for Δt+1
            if kernel[row, j].item() > 0 and kernel[row, j+1].item() > 0:
                actual_ratio = kernel[row, j+1].item() / kernel[row, j].item()
                assert abs(actual_ratio - expected_ratio) < 0.1, \
                    f"Decay ratio mismatch at j={j}: got {actual_ratio:.4f}, expected {expected_ratio:.4f}"

    def test_causality_mask(self):
        """
        Verify strictly causal attention (no future information).
        """
        config = HatchingConfig()
        attn = STDPAttention(config)

        T = 16
        kernel = attn.compute_stdp_kernel(T, device=torch.device('cpu'))

        # Upper triangle (including diagonal) should be zero
        upper = torch.triu(kernel, diagonal=0)
        assert (upper == 0).all(), "STDP kernel should be strictly causal"


# =============================================================================
# HIPPO TESTS
# =============================================================================

class TestHiPPOMatrices:
    """
    Verify HiPPO matrices have correct mathematical properties.

    Key properties:
    1. A is lower triangular (for LegS)
    2. Eigenvalues determine stability
    3. Discretization preserves dynamics
    """

    def test_legs_structure(self):
        """
        Verify LegS matrix structure.

        A_nk = -√(2n+1)√(2k+1) for n > k
        A_nn = -(n+1)
        A_nk = 0 for n < k
        """
        N = 16
        A, B = make_hippo_legs(N)

        # Check lower triangular structure
        for n in range(N):
            for k in range(N):
                if n < k:
                    assert A[n, k].item() == 0, f"A[{n},{k}] should be 0"
                elif n == k:
                    expected = -(n + 1)
                    assert abs(A[n, k].item() - expected) < 1e-6, \
                        f"A[{n},{n}] should be {expected}"
                else:  # n > k
                    expected = -math.sqrt(2*n+1) * math.sqrt(2*k+1)
                    assert abs(A[n, k].item() - expected) < 1e-6, \
                        f"A[{n},{k}] should be {expected}"

    def test_b_vector(self):
        """
        Verify B vector: B_n = √(2n+1)
        """
        N = 16
        A, B = make_hippo_legs(N)

        for n in range(N):
            expected = math.sqrt(2*n + 1)
            assert abs(B[n, 0].item() - expected) < 1e-6, \
                f"B[{n}] should be {expected}"

    def test_discretization_stability(self):
        """
        Verify discretized system is stable (eigenvalues inside unit circle).
        """
        N = 16
        A, B = make_hippo_legs(N)
        A_bar, B_bar = discretize_zoh(A, B, dt=1.0)

        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvals(A_bar)
        magnitudes = eigenvalues.abs()

        # All eigenvalues should have magnitude ≤ 1 for stability
        assert (magnitudes <= 1.0 + 1e-6).all(), \
            f"Unstable eigenvalue found: max magnitude = {magnitudes.max().item()}"

    def test_hippo_memory_reconstruction(self):
        """
        Verify HiPPO can reconstruct input history.

        The state c[t] should encode coefficients such that
        we can approximately reconstruct recent inputs.
        """
        input_dim = 1
        state_dim = 32
        hippo = HiPPOCell(input_dim, state_dim)

        # Create a simple signal
        T = 100
        t = torch.linspace(0, 2*math.pi, T)
        signal = torch.sin(t).view(1, T, 1)  # [B=1, T, D=1]

        # Process through HiPPO
        output, final_state = hippo(signal)

        # The output should track the input with some delay/smoothing
        # Compute correlation as a basic check
        input_flat = signal.squeeze()
        output_flat = output.squeeze()

        correlation = torch.corrcoef(torch.stack([input_flat, output_flat]))[0, 1]

        # Should have positive correlation (tracking the signal)
        assert correlation > 0.5, f"Poor signal tracking: correlation = {correlation:.4f}"


# =============================================================================
# BCM HOMEOSTASIS TESTS
# =============================================================================

class TestBCMHomeostasis:
    """
    Verify BCM synaptic homeostasis properties.

    Key property: Sliding threshold tracks activity to maintain stability.
    θ_M adapts such that low activity → lower threshold → easier LTP
                        high activity → higher threshold → easier LTD
    """

    def test_threshold_adaptation_high_activity(self):
        """
        High activity should increase threshold.
        """
        bcm = BCMHomeostasis(n_neurons=100, tau_theta=10.0)
        initial_theta = bcm.theta.mean().item()

        # Simulate high activity
        for _ in range(50):
            high_activity = torch.ones(1, 100) * 2.0  # High activity
            _, _ = bcm(high_activity, update_threshold=True)

        final_theta = bcm.theta.mean().item()

        assert final_theta > initial_theta, \
            f"Threshold should increase: {initial_theta:.4f} → {final_theta:.4f}"

    def test_threshold_adaptation_low_activity(self):
        """
        Low activity should decrease threshold.
        """
        bcm = BCMHomeostasis(n_neurons=100, tau_theta=10.0, target_rate=0.5)

        # Initialize with high threshold
        bcm.theta = torch.full((100,), 1.0)
        initial_theta = bcm.theta.mean().item()

        # Simulate low activity
        for _ in range(50):
            low_activity = torch.ones(1, 100) * 0.01  # Low activity
            _, _ = bcm(low_activity, update_threshold=True)

        final_theta = bcm.theta.mean().item()

        assert final_theta < initial_theta, \
            f"Threshold should decrease: {initial_theta:.4f} → {final_theta:.4f}"

    def test_homeostatic_equilibrium(self):
        """
        At equilibrium, threshold should equal activity².

        Steady state of τ · dθ/dt = v² - θ  is  θ = v²
        """
        bcm = BCMHomeostasis(n_neurons=1, tau_theta=10.0)

        # Fixed activity level
        activity_level = 0.3
        fixed_activity = torch.full((1, 1), activity_level)

        # Run until equilibrium
        for _ in range(200):
            _, _ = bcm(fixed_activity, update_threshold=True)

        expected_theta = activity_level ** 2
        actual_theta = bcm.theta.item()

        assert abs(actual_theta - expected_theta) < 0.01, \
            f"Equilibrium mismatch: got {actual_theta:.4f}, expected {expected_theta:.4f}"


# =============================================================================
# SCALE-FREE NETWORK TESTS
# =============================================================================

class TestScaleFreeNetwork:
    """
    Verify scale-free connectivity has power-law degree distribution.
    """

    def test_power_law_sampling(self):
        """
        Verify power_law_sample produces heavy-tailed distribution.
        """
        n_samples = 10000
        gamma = 2.5

        samples = power_law_sample(n_samples, gamma, device=torch.device('cpu'))

        # Should have some very high values (heavy tail)
        max_val = samples.max().item()
        mean_val = samples.mean().item()

        # Heavy-tailed: max >> mean
        assert max_val > 5 * mean_val, \
            f"Not heavy-tailed enough: max={max_val:.1f}, mean={mean_val:.1f}"

    def test_degree_distribution(self):
        """
        Verify generated mask has approximately power-law degree distribution.
        """
        n_neurons = 500
        avg_degree = 20
        gamma = 2.5

        mask = generate_scale_free_mask(n_neurons, avg_degree, gamma, torch.device('cpu'))

        # Compute degree sequence
        degrees = mask.sum(dim=1)

        # Should have high variance (heterogeneous degrees)
        cv = degrees.std() / degrees.mean()  # Coefficient of variation

        # Scale-free networks have high CV (typically > 1)
        # Random networks have CV ≈ 1/√(avg_degree)
        random_cv = 1.0 / math.sqrt(avg_degree)

        assert cv > 2 * random_cv, \
            f"Degree distribution not heterogeneous enough: CV={cv:.2f}, random would be {random_cv:.2f}"

    def test_hub_existence(self):
        """
        Verify existence of hub nodes (high-degree nodes).
        """
        n_neurons = 500
        avg_degree = 20
        gamma = 2.5

        mask = generate_scale_free_mask(n_neurons, avg_degree, gamma, torch.device('cpu'))
        degrees = mask.sum(dim=1)

        # Should have some hubs with degree >> average
        max_degree = degrees.max().item()

        assert max_degree > 3 * avg_degree, \
            f"No clear hubs: max_degree={max_degree:.1f}, avg={avg_degree}"


# =============================================================================
# CRITICALITY TESTS
# =============================================================================

class TestCriticality:
    """
    Verify critical dynamics regulation.
    """

    def test_variance_regulation(self):
        """
        Verify criticality regulator maintains target variance.
        """
        crit = CriticalityRegulator(n_features=64, target_radius=1.0)

        # High variance input
        x_high = torch.randn(100, 64) * 5.0

        # Process multiple times to let adaptation occur
        for _ in range(50):
            y = crit(x_high)

        # Output variance should be closer to target
        output_std = y.std().item()
        input_std = x_high.std().item()

        assert output_std < input_std, \
            f"Should reduce variance: input_std={input_std:.2f}, output_std={output_std:.2f}"

    def test_gain_adaptation(self):
        """
        Verify gain adapts to maintain stability.
        """
        crit = CriticalityRegulator(n_features=32, target_radius=1.0, tau_adapt=10.0)

        # Track variance over time with constant high input
        variances = []
        x = torch.randn(10, 32) * 3.0

        for _ in range(100):
            y = crit(x)
            variances.append(y.var().item())

        # Variance should decrease over time (converging to target)
        assert variances[-1] < variances[0], \
            f"Variance should decrease: start={variances[0]:.2f}, end={variances[-1]:.2f}"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestFullModel:
    """
    Integration tests for the full Hatching model.
    """

    def test_forward_backward(self):
        """
        Verify gradients flow through entire model.
        """
        from hatching import Hatching, HatchingConfig

        config = HatchingConfig(
            n_layer=2, n_embd=32, n_head=2,
            mlp_internal_dim_multiplier=16, vocab_size=256
        )
        model = Hatching(config)

        x = torch.randint(0, 256, (2, 16))
        y = torch.randint(0, 256, (2, 16))

        logits, loss, _ = model(x, y)
        loss.backward()

        # Check gradients exist for key parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                if 'encoder' in name or 'decoder' in name:
                    assert (param.grad != 0).any(), f"Zero gradient for {name}"

    def test_sparse_activations(self):
        """
        Verify activations are sparse (biologically plausible).
        """
        from hatching import Hatching, HatchingConfig

        config = HatchingConfig(
            n_layer=2, n_embd=64, n_head=2,
            mlp_internal_dim_multiplier=32, vocab_size=256
        )
        model = Hatching(config)

        x = torch.randint(0, 256, (4, 32))
        _, _, states = model(x, return_states=True)

        # Check trace sparsity
        for i, trace in enumerate(states.get('traces', [])):
            if trace is not None:
                sparsity = (trace == 0).float().mean().item()
                # Should have reasonable sparsity (not all active)
                assert sparsity > 0.3, f"Layer {i} not sparse enough: {1-sparsity:.1%} active"

    def test_generation_coherence(self):
        """
        Verify generation produces valid token sequences.
        """
        from hatching import Hatching, HatchingConfig

        config = HatchingConfig(
            n_layer=2, n_embd=64, n_head=2,
            mlp_internal_dim_multiplier=32, vocab_size=256
        )
        model = Hatching(config)
        model.eval()

        prompt = torch.randint(0, 256, (1, 5))
        generated = model.generate(prompt, max_new_tokens=20, temperature=1.0)

        # Should have correct shape
        assert generated.shape == (1, 25), f"Wrong shape: {generated.shape}"

        # Should be valid tokens
        assert (generated >= 0).all() and (generated < 256).all(), "Invalid tokens"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
