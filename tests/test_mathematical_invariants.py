# Copyright 2025 - Mathematical Invariant Tests for Dragon Hatching Fixes
#
# These tests verify the mathematical properties proven in MATHEMATICAL_PROOFS.md
# Each test corresponds to a theorem and checks that the implementation
# satisfies the proven invariants.

import math
import pytest
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple

import sys
sys.path.insert(0, '..')

from hatching_fixes import (
    DaleLawLinearFixed,
    BCMHomeostasisFixed,
    spike_function_fixed,
    OptimalSurrogateSpike,
    STDPKernelCache,
    STDPAttentionFixed,
    compute_per_sample_loss,
    compute_task_losses,
    verify_loss_decomposition,
    RoPEConfig,
    StatefulLIF,
    LossAccumulator,
)


# =============================================================================
# THEOREM 1.1: DALE'S LAW SIGN CONSTRAINT
# =============================================================================

class TestDalesLaw:
    """
    Verify Dale's Law implementation satisfies Theorem 1.1:
    The sign constraint applies to columns (presynaptic neurons).
    """

    def test_excitatory_columns_positive(self):
        """Excitatory neuron columns should have all non-negative weights."""
        layer = DaleLawLinearFixed(100, 200, excitatory_fraction=0.8)
        W = layer.weight.detach()

        # First 80 columns are excitatory
        excitatory_cols = W[:, :80]
        assert (excitatory_cols >= -1e-6).all(), "Excitatory columns should be non-negative"

    def test_inhibitory_columns_negative(self):
        """Inhibitory neuron columns should have all non-positive weights."""
        layer = DaleLawLinearFixed(100, 200, excitatory_fraction=0.8)
        W = layer.weight.detach()

        # Last 20 columns are inhibitory
        inhibitory_cols = W[:, 80:]
        assert (inhibitory_cols <= 1e-6).all(), "Inhibitory columns should be non-positive"

    def test_column_sign_consistency(self):
        """Each column should have consistent sign (all + or all -)."""
        layer = DaleLawLinearFixed(50, 100, excitatory_fraction=0.6)
        W = layer.weight.detach()

        for j in range(50):
            col = W[:, j]
            signs = torch.sign(col[col.abs() > 1e-8])  # Ignore near-zero
            if len(signs) > 0:
                # All signs should be the same
                assert (signs == signs[0]).all(), f"Column {j} has inconsistent signs"

    def test_verify_method(self):
        """The verify_dales_law method should return True for valid weights."""
        layer = DaleLawLinearFixed(64, 128)
        assert layer.verify_dales_law(), "verify_dales_law should return True"

    def test_forward_preserves_constraint(self):
        """Forward pass should not violate Dale's Law."""
        layer = DaleLawLinearFixed(32, 64)
        x = torch.randn(4, 32)

        # Multiple forward passes
        for _ in range(10):
            _ = layer(x)

        assert layer.verify_dales_law(), "Constraint should hold after forward passes"


# =============================================================================
# THEOREM 2.3: BCM POSITIVITY GUARANTEE
# =============================================================================

class TestBCMPositivity:
    """
    Verify BCM implementation satisfies Theorem 2.3:
    Theta is guaranteed positive via log parameterization.
    """

    def test_initial_theta_positive(self):
        """Theta should be positive at initialization."""
        bcm = BCMHomeostasisFixed(100)
        assert (bcm.theta > 0).all(), "Initial theta should be positive"

    def test_theta_stays_positive_zero_activity(self):
        """Theta should remain positive with zero activity."""
        bcm = BCMHomeostasisFixed(100, tau_theta=10.0)

        # Many updates with zero activity
        zero_activity = torch.zeros(4, 100)
        for _ in range(1000):
            _, theta = bcm(zero_activity)
            assert (theta > 0).all(), "Theta should stay positive with zero activity"

    def test_theta_stays_positive_high_activity(self):
        """Theta should remain positive with very high activity."""
        bcm = BCMHomeostasisFixed(100, tau_theta=10.0)

        high_activity = torch.ones(4, 100) * 1000
        for _ in range(100):
            _, theta = bcm(high_activity)
            assert (theta > 0).all(), "Theta should stay positive with high activity"
            assert not torch.isnan(theta).any(), "Theta should not be NaN"
            assert not torch.isinf(theta).any(), "Theta should not be infinite"

    def test_theta_stays_positive_random_activity(self):
        """Theta should remain positive with random activity."""
        bcm = BCMHomeostasisFixed(100, tau_theta=10.0)

        for _ in range(500):
            activity = torch.randn(8, 100) * 10  # Random, potentially negative
            _, theta = bcm(activity)
            assert (theta > 0).all(), "Theta should stay positive"
            assert bcm.verify_positivity(), "verify_positivity should return True"

    def test_sqrt_never_fails(self):
        """sqrt(theta) should never fail (no negative values)."""
        bcm = BCMHomeostasisFixed(64)

        for _ in range(100):
            activity = torch.randn(4, 64) * 5
            modulation, theta = bcm(activity)

            # This should not raise
            theta_sqrt = theta.sqrt()
            assert not torch.isnan(theta_sqrt).any(), "sqrt(theta) should not be NaN"


# =============================================================================
# THEOREM 3.1: SURROGATE GRADIENT PROPERTIES
# =============================================================================

class TestSurrogateGradient:
    """
    Verify surrogate gradient satisfies Theorem 3.1:
    - Maximum at threshold
    - Integrates to approximately 1
    - Enables gradient flow
    """

    def test_gradient_exists(self):
        """Gradients should flow through spike function."""
        x = torch.randn(100, requires_grad=True)
        spikes = spike_function_fixed(x, threshold=0.0)
        loss = spikes.sum()
        loss.backward()

        assert x.grad is not None, "Gradient should exist"
        assert (x.grad != 0).any(), "Some gradients should be non-zero"

    def test_gradient_maximum_at_threshold(self):
        """Surrogate gradient should be maximum near threshold."""
        x = torch.linspace(-5, 5, 1000, requires_grad=True)
        threshold = 0.0

        spikes = spike_function_fixed(x, threshold)

        # Compute gradient for each point
        grads = []
        for i in range(len(x)):
            if x.grad is not None:
                x.grad.zero_()
            spikes[i].backward(retain_graph=True)
            grads.append(x.grad[i].item())

        grads = np.array(grads)
        max_idx = np.argmax(grads)
        max_x = x[max_idx].item()

        # Maximum should be near threshold
        assert abs(max_x - threshold) < 0.5, f"Max gradient at {max_x}, expected near {threshold}"

    def test_gradient_integral_approximately_one(self):
        """Surrogate gradient should integrate to approximately 1."""
        x = torch.linspace(-20, 20, 2000, requires_grad=True)
        beta = 10.0
        dx = 40 / 2000

        spikes = spike_function_fixed(x, threshold=0.0, beta=beta)
        grad = torch.autograd.grad(spikes.sum(), x)[0]

        integral = grad.sum().item() * dx

        # Normalized surrogate should integrate to 1
        assert abs(integral - 1.0) < 0.15, f"Integral = {integral}, expected ~1.0"

    def test_gradient_decays_from_threshold(self):
        """Gradient should decay away from threshold."""
        x = torch.tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        spikes = spike_function_fixed(x, threshold=0.0)

        grads = []
        for i in range(len(x)):
            if x.grad is not None:
                x.grad.zero_()
            spikes[i].backward(retain_graph=True)
            grads.append(x.grad[i].item())

        # Gradients should decrease with distance from threshold
        for i in range(1, len(grads)):
            assert grads[i] < grads[i-1], "Gradient should decay with distance"


# =============================================================================
# THEOREM 4.2: STDP KERNEL PROPERTIES
# =============================================================================

class TestSTDPKernel:
    """
    Verify STDP kernel satisfies Theorem 4.2:
    - Toeplitz structure (K[i,j] = f(i-j))
    - Causal (K[i,j] = 0 for i <= j)
    - Exponential decay
    """

    def test_kernel_is_causal(self):
        """Kernel should be strictly causal (lower triangular, zero diagonal)."""
        cache = STDPKernelCache(tau_plus=20.0, A_plus=0.01)
        K = cache.get_kernel(64, torch.device('cpu'))

        # Upper triangle including diagonal should be zero
        upper = torch.triu(K, diagonal=0)
        assert (upper == 0).all(), "STDP kernel should be causal"

    def test_kernel_exponential_decay(self):
        """Kernel values should decay exponentially with distance."""
        tau_plus = 20.0
        cache = STDPKernelCache(tau_plus=tau_plus, A_plus=1.0)
        K = cache.get_kernel(100, torch.device('cpu'))

        # Check decay ratio along a row
        row = 50
        for delta in range(1, 10):
            if K[row, row-delta].item() > 0 and K[row, row-delta-1].item() > 0:
                expected_ratio = math.exp(-1.0 / tau_plus)
                actual_ratio = K[row, row-delta-1].item() / K[row, row-delta].item()
                assert abs(actual_ratio - expected_ratio) < 0.01, \
                    f"Decay ratio mismatch: {actual_ratio:.4f} vs {expected_ratio:.4f}"

    def test_kernel_toeplitz_structure(self):
        """K[i,j] should depend only on i-j."""
        cache = STDPKernelCache()
        K = cache.get_kernel(32, torch.device('cpu'))

        # All diagonals should have constant values
        for diag in range(-31, 0):  # Only lower diagonals
            diagonal = torch.diagonal(K, offset=diag)
            if len(diagonal) > 1:
                # All values on this diagonal should be equal
                diff = (diagonal - diagonal[0]).abs().max()
                assert diff < 1e-6, f"Diagonal {diag} not constant"

    def test_caching_works(self):
        """Same kernel should be returned from cache."""
        cache = STDPKernelCache()

        K1 = cache.get_kernel(32, torch.device('cpu'))
        K2 = cache.get_kernel(32, torch.device('cpu'))

        assert K1 is K2, "Cached kernel should be same object"

    def test_different_sizes_cached_separately(self):
        """Different sizes should have different cached kernels."""
        cache = STDPKernelCache()

        K32 = cache.get_kernel(32, torch.device('cpu'))
        K64 = cache.get_kernel(64, torch.device('cpu'))

        assert K32.shape != K64.shape, "Different sizes should have different shapes"


# =============================================================================
# THEOREM 5.1: MULTI-TASK LOSS DECOMPOSITION
# =============================================================================

class TestMultiTaskLoss:
    """
    Verify multi-task loss satisfies Theorem 5.1:
    L_total = sum_task (|B_task|/|B|) * L_task
    """

    def test_per_sample_loss_shape(self):
        """Per-sample loss should have shape [B]."""
        logits = torch.randn(8, 16, 100)
        targets = torch.randint(0, 100, (8, 16))

        losses = compute_per_sample_loss(logits, targets)
        assert losses.shape == (8,), f"Expected shape (8,), got {losses.shape}"

    def test_per_sample_loss_values(self):
        """Per-sample losses should be positive."""
        logits = torch.randn(8, 16, 100)
        targets = torch.randint(0, 100, (8, 16))

        losses = compute_per_sample_loss(logits, targets)
        assert (losses > 0).all(), "Losses should be positive"

    def test_loss_decomposition_theorem(self):
        """Weighted task losses should equal batch loss (Theorem 5.1)."""
        B, T, V = 16, 32, 50
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        task_ids = torch.randint(0, 3, (B,))
        task_names = ["task_a", "task_b", "task_c"]

        # Compute task losses
        task_losses = compute_task_losses(logits, targets, task_ids, task_names)

        # Compute batch loss
        batch_loss = F.cross_entropy(logits.view(-1, V), targets.view(-1)).item()

        # Verify decomposition
        assert verify_loss_decomposition(task_losses, batch_loss, tolerance=0.01), \
            "Loss decomposition theorem should hold"

    def test_single_task_matches_batch(self):
        """With single task, task loss should equal batch loss."""
        logits = torch.randn(8, 16, 50)
        targets = torch.randint(0, 50, (8, 16))
        task_ids = torch.zeros(8, dtype=torch.long)  # All same task
        task_names = ["only_task"]

        task_losses = compute_task_losses(logits, targets, task_ids, task_names)
        batch_loss = F.cross_entropy(logits.view(-1, 50), targets.view(-1)).item()

        task_mean = np.mean(task_losses["only_task"])
        assert abs(task_mean - batch_loss) < 0.01, \
            f"Single task loss {task_mean:.4f} should equal batch loss {batch_loss:.4f}"

    def test_handles_padding(self):
        """Should handle padding tokens correctly."""
        logits = torch.randn(4, 10, 50)
        targets = torch.randint(0, 50, (4, 10))
        targets[:, -3:] = -100  # Padding

        losses = compute_per_sample_loss(logits, targets, ignore_index=-100)
        assert not torch.isnan(losses).any(), "Should not produce NaN with padding"
        assert not torch.isinf(losses).any(), "Should not produce inf with padding"


# =============================================================================
# THEOREM 6.1: ROPE CONFIGURATION
# =============================================================================

class TestRoPEConfig:
    """
    Verify RoPE configuration satisfies Theorem 6.1:
    Optimal theta derived from information theory.
    """

    def test_theta_scales_with_sequence_length(self):
        """Larger sequence length should give larger theta."""
        cfg1 = RoPEConfig(max_seq_len=1024, n_embd=256)
        cfg2 = RoPEConfig(max_seq_len=4096, n_embd=256)

        assert cfg2.theta >= cfg1.theta, "Theta should scale with sequence length"

    def test_theta_scales_with_extrapolation(self):
        """Larger extrapolation factor should give larger theta."""
        cfg1 = RoPEConfig(max_seq_len=2048, n_embd=256, extrapolation_factor=2.0)
        cfg2 = RoPEConfig(max_seq_len=2048, n_embd=256, extrapolation_factor=8.0)

        assert cfg2.theta >= cfg1.theta, "Theta should scale with extrapolation"

    def test_theta_is_power_of_two(self):
        """Computed theta should be power of 2 for efficiency."""
        cfg = RoPEConfig(max_seq_len=2048, n_embd=256)
        theta = cfg.theta

        log2_theta = math.log2(theta)
        assert log2_theta == int(log2_theta), f"Theta {theta} should be power of 2"

    def test_theta_covers_sequence_length(self):
        """Theta should be large enough to cover sequence length."""
        for L in [512, 1024, 2048, 4096]:
            cfg = RoPEConfig(max_seq_len=L, n_embd=256)
            # Wavelength at highest frequency should cover sequence
            min_wavelength = 2 * math.pi
            max_wavelength = 2 * math.pi * cfg.theta
            assert max_wavelength >= L, f"Theta {cfg.theta} too small for L={L}"

    def test_override_works(self):
        """Override should bypass calculation."""
        cfg = RoPEConfig(override_theta=12345.0)
        assert cfg.theta == 12345.0, "Override should be used"

    def test_validation_catches_issues(self):
        """Validation should catch problematic configs."""
        # Very small embedding dim
        cfg = RoPEConfig(n_embd=16)
        warnings = cfg.validate()
        assert len(warnings) > 0, "Should warn about small embedding dim"


# =============================================================================
# THEOREM 7.2: STATEFUL GENERATION
# =============================================================================

class TestStatefulLIF:
    """
    Verify stateful LIF satisfies Theorem 7.2:
    State (M_t, Sigma_t) correctly propagates through time.
    """

    def test_state_initialization(self):
        """Initial state should be zeros."""
        lif = StatefulLIF(n_neurons=64)
        x = torch.randn(4, 64)

        spikes, membrane, trace = lif(x)

        assert membrane.shape == (4, 64), "Membrane shape incorrect"
        assert trace.shape == (4, 64), "Trace shape incorrect"

    def test_state_propagation(self):
        """State should propagate correctly through time."""
        lif = StatefulLIF(n_neurons=32, tau_mem=10.0)

        membrane = torch.zeros(2, 32)
        trace = torch.zeros(2, 32)

        # Step 1
        x1 = torch.ones(2, 32) * 0.5
        _, membrane, trace = lif(x1, membrane, trace)
        mem1 = membrane.clone()

        # Step 2
        x2 = torch.zeros(2, 32)
        _, membrane, trace = lif(x2, membrane, trace)

        # Membrane should decay
        expected_decay = lif.alpha_mem
        actual_ratio = membrane.mean() / mem1.mean()
        assert abs(actual_ratio - expected_decay) < 0.1, \
            f"Membrane decay incorrect: {actual_ratio:.3f} vs {expected_decay:.3f}"

    def test_sequence_matches_incremental(self):
        """Full sequence processing should match incremental."""
        lif = StatefulLIF(n_neurons=16)

        # Full sequence
        x_seq = torch.randn(2, 10, 16)
        spikes_seq, mem_seq, trace_seq = lif.forward_sequence(x_seq)

        # Incremental
        membrane = torch.zeros(2, 16)
        trace = torch.zeros(2, 16)
        spikes_inc = []

        for t in range(10):
            s, membrane, trace = lif(x_seq[:, t, :], membrane, trace)
            spikes_inc.append(s)

        spikes_inc = torch.stack(spikes_inc, dim=1)

        # Should match
        assert torch.allclose(spikes_seq, spikes_inc), "Sequence and incremental should match"
        assert torch.allclose(mem_seq, membrane), "Final membrane should match"
        assert torch.allclose(trace_seq, trace), "Final trace should match"


# =============================================================================
# LOSS ACCUMULATOR (ENGINEERING FIX)
# =============================================================================

class TestLossAccumulator:
    """
    Verify LossAccumulator prevents memory leaks.
    """

    def test_accumulates_correctly(self):
        """Should track sum and count correctly."""
        acc = LossAccumulator()

        losses = [0.5, 0.3, 0.2, 0.4]
        for l in losses:
            acc.add(torch.tensor(l))

        assert acc.count() == 4
        assert abs(acc.sum() - 1.4) < 1e-6
        assert abs(acc.mean() - 0.35) < 1e-6

    def test_reset_works(self):
        """Reset should clear state."""
        acc = LossAccumulator()
        acc.add(torch.tensor(1.0))
        acc.add(torch.tensor(2.0))
        acc.reset()

        assert acc.count() == 0
        assert acc.sum() == 0.0

    def test_handles_cuda_tensors(self):
        """Should work with CUDA tensors (if available)."""
        acc = LossAccumulator()

        if torch.cuda.is_available():
            loss = torch.tensor(0.5, device='cuda')
            acc.add(loss)
            assert acc.count() == 1
            # Value should be extracted (not kept on GPU)
            assert isinstance(acc._sum, float)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
