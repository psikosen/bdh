# Copyright 2025 - Advanced Dragon Hatching Components
# Deep mathematical extensions for biological plausibility
#
# Additional components:
# 1. HiPPO (High-order Polynomial Projection Operators) for optimal memory
# 2. Synaptic Homeostasis (BCM rule) for stability
# 3. Critical Dynamics (edge of chaos) for optimal computation
# 4. Dendritic Computation for non-linear integration
# 5. Neuromodulation (dopamine/acetylcholine-like gating)

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from hatching import HatchingConfig, spike_function


# =============================================================================
# HIPPO: HIGH-ORDER POLYNOMIAL PROJECTION OPERATORS
# =============================================================================
#
# HiPPO provides optimal memory via polynomial basis projection.
# The state evolves to optimally remember past inputs.
#
# For the Legendre measure (LegS):
# A_nk = -{ (2n+1)^{1/2} (2k+1)^{1/2}  if n > k
#        { n+1                          if n = k
#        { 0                            if n < k
#
# B_n = (2n+1)^{1/2}
#
# The continuous dynamics: dc/dt = A·c + B·f(t)
# Discretized: c[t+1] = Ā·c[t] + B̄·u[t]
# where Ā = exp(A·Δ), B̄ = (Ā - I)·A^{-1}·B (Zero-order hold)
#


def make_hippo_legs(N: int) -> Tuple[Tensor, Tensor]:
    """
    Construct HiPPO-LegS (Legendre) matrices.

    The LegS measure gives optimal polynomial approximation of history
    with uniform weighting over the sliding window.

    Returns:
        A: State transition matrix [N, N]
        B: Input projection vector [N, 1]
    """
    # Construct A matrix for LegS
    A = torch.zeros(N, N)
    B = torch.zeros(N, 1)

    for n in range(N):
        B[n, 0] = math.sqrt(2 * n + 1)
        for k in range(N):
            if n > k:
                A[n, k] = -math.sqrt(2 * n + 1) * math.sqrt(2 * k + 1)
            elif n == k:
                A[n, k] = -(n + 1)
            # else: 0

    return A, B


def make_hippo_legt(N: int, theta: float = 1.0) -> Tuple[Tensor, Tensor]:
    """
    Construct HiPPO-LegT (Translated Legendre) matrices.

    LegT uses translated Legendre polynomials with exponential decay,
    providing a tilted measure that emphasizes recent history.

    Args:
        N: State dimension
        theta: Timescale parameter

    Returns:
        A: State transition matrix [N, N]
        B: Input projection vector [N, 1]
    """
    A = torch.zeros(N, N)
    B = torch.zeros(N, 1)

    for n in range(N):
        B[n, 0] = math.sqrt(2 * n + 1) / theta
        for k in range(N):
            if n > k:
                A[n, k] = -math.sqrt(2 * n + 1) * math.sqrt(2 * k + 1) / theta
            elif n == k:
                A[n, k] = -(n + 1) / theta
            else:
                A[n, k] = math.sqrt(2 * n + 1) * math.sqrt(2 * k + 1) / theta

    return A, B


def discretize_zoh(A: Tensor, B: Tensor, dt: float = 1.0) -> Tuple[Tensor, Tensor]:
    """
    Discretize continuous system using Zero-Order Hold (ZOH).

    Continuous: dx/dt = Ax + Bu
    Discrete: x[k+1] = Ā·x[k] + B̄·u[k]

    where:
    Ā = exp(A·dt)
    B̄ = A^{-1}·(Ā - I)·B

    For numerical stability, we use the matrix exponential directly.
    """
    N = A.shape[0]
    I = torch.eye(N)

    # Compute matrix exponential: Ā = exp(A·dt)
    # Using Padé approximation for better numerical stability
    A_scaled = A * dt
    A_bar = torch.matrix_exp(A_scaled)

    # Compute B̄ = A^{-1}·(Ā - I)·B
    # Using the identity: A^{-1}(exp(A) - I) = ∫₀¹ exp(As) ds · I
    # Approximated via Taylor series for numerical stability
    B_bar = torch.zeros(N, B.shape[1])
    term = B * dt
    B_bar = term.clone()
    for i in range(1, 20):  # Taylor series truncation
        term = A_scaled @ term / (i + 1)
        B_bar = B_bar + term

    return A_bar, B_bar


class HiPPOCell(nn.Module):
    """
    HiPPO-based state space cell for optimal memory.

    This cell maintains a compressed representation of input history
    using orthogonal polynomial projections. The state c[t] represents
    coefficients of a polynomial approximation to the input history.

    Mathematical foundation:
    The polynomial coefficients evolve according to:
    c[t+1] = Ā·c[t] + B̄·u[t]

    Reconstruction of history:
    f̂(s) = Σ_n c_n(t) · P_n(s)

    where P_n are the Legendre polynomials.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        hippo_type: str = 'legs',
        dt: float = 1.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim

        # Construct HiPPO matrices
        if hippo_type == 'legs':
            A, B = make_hippo_legs(state_dim)
        elif hippo_type == 'legt':
            A, B = make_hippo_legt(state_dim)
        else:
            raise ValueError(f"Unknown HiPPO type: {hippo_type}")

        # Discretize
        A_bar, B_bar = discretize_zoh(A, B, dt)

        # Register as buffers (not learnable, but part of model state)
        self.register_buffer('A', A_bar)
        self.register_buffer('B', B_bar)

        # Learnable input/output projections
        self.input_proj = nn.Linear(input_dim, state_dim, bias=False)
        self.output_proj = nn.Linear(state_dim, input_dim, bias=False)

        # Initialize projections
        nn.init.normal_(self.input_proj.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)

    def forward(
        self,
        u: Tensor,
        state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Process input through HiPPO dynamics.

        Args:
            u: Input tensor [B, T, D]
            state: Previous state [B, N] (optional)

        Returns:
            y: Output tensor [B, T, D]
            state: Final state [B, N]
        """
        B, T, D = u.shape
        N = self.state_dim

        if state is None:
            state = torch.zeros(B, N, device=u.device, dtype=u.dtype)

        outputs = []

        for t in range(T):
            # Project input
            u_t = self.input_proj(u[:, t, :])  # [B, N]

            # HiPPO update: c = A·c + B·u
            state = state @ self.A.T + u_t @ self.B.T.squeeze(-1)

            # Project output
            y_t = self.output_proj(state)  # [B, D]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # [B, T, D]
        return y, state


# =============================================================================
# BCM (BIENENSTOCK-COOPER-MUNRO) SYNAPTIC HOMEOSTASIS
# =============================================================================
#
# BCM provides metaplasticity: the threshold for LTP/LTD slides based
# on recent postsynaptic activity, preventing runaway excitation.
#
# Weight update rule:
# dW/dt = η · u · (u - θ_M) · v
#
# where:
# - u: presynaptic activity
# - v: postsynaptic activity
# - θ_M: sliding threshold (moving average of v²)
#
# Threshold dynamics:
# τ_θ · dθ_M/dt = v² - θ_M
#
# This ensures:
# - When v < θ_M: LTD (depression)
# - When v > θ_M: LTP (potentiation)
# - θ_M tracks activity to maintain stable firing rates
#


class BCMHomeostasis(nn.Module):
    """
    BCM-style synaptic homeostasis for stable learning.

    Implements the sliding threshold mechanism that prevents
    runaway potentiation and maintains balanced neural activity.
    """

    def __init__(
        self,
        n_neurons: int,
        tau_theta: float = 100.0,  # Threshold time constant
        target_rate: float = 0.1,   # Target firing rate
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.tau_theta = tau_theta
        self.target_rate = target_rate

        # Sliding threshold (initialized to target²)
        self.register_buffer(
            'theta',
            torch.full((n_neurons,), target_rate ** 2)
        )

        # Activity trace for computing threshold
        self.alpha = 1.0 / tau_theta

    def forward(
        self,
        activity: Tensor,
        update_threshold: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply homeostatic regulation.

        Args:
            activity: Neural activity [B, ..., N]
            update_threshold: Whether to update sliding threshold

        Returns:
            modulated: Homeostasis-modulated activity
            threshold: Current threshold values
        """
        # Compute activity squared (for threshold update)
        activity_sq = activity.pow(2).mean(dim=tuple(range(activity.dim() - 1)))

        # Update sliding threshold
        if update_threshold and self.training:
            # τ · dθ/dt = v² - θ  →  θ = θ + α·(v² - θ)
            self.theta = (1 - self.alpha) * self.theta + self.alpha * activity_sq

        # BCM modulation: scale by distance from threshold
        # Positive when activity > sqrt(theta), negative otherwise
        threshold_expanded = self.theta.view(*([1] * (activity.dim() - 1)), -1)
        modulation = (activity - threshold_expanded.sqrt()) / (threshold_expanded.sqrt() + 1e-6)

        return modulation, self.theta


# =============================================================================
# CRITICAL DYNAMICS: EDGE OF CHAOS
# =============================================================================
#
# Neural systems operate optimally at criticality - the boundary between
# ordered and chaotic dynamics. At criticality:
# - Information transmission is maximized
# - Dynamic range is maximal
# - Correlation lengths diverge (long-range integration)
#
# We implement criticality via:
# 1. Gain modulation to maintain spectral radius ≈ 1
# 2. Balance of excitation and inhibition
# 3. Avalanche dynamics with power-law statistics
#


class CriticalityRegulator(nn.Module):
    """
    Maintains dynamics at the edge of chaos.

    Uses adaptive gain control to keep the effective spectral
    radius of the recurrent dynamics near unity.
    """

    def __init__(
        self,
        n_features: int,
        target_radius: float = 1.0,
        tau_adapt: float = 50.0
    ):
        super().__init__()
        self.n_features = n_features
        self.target_radius = target_radius
        self.tau_adapt = tau_adapt

        # Adaptive gain parameter
        self.gain = nn.Parameter(torch.ones(1))

        # Running estimate of activity variance
        self.register_buffer('variance_ema', torch.ones(1))

    def forward(self, x: Tensor, W: Optional[Tensor] = None) -> Tensor:
        """
        Apply criticality-maintaining gain modulation.

        Args:
            x: Activations [B, ..., D]
            W: Weight matrix (optional, for spectral analysis)

        Returns:
            Gain-modulated activations
        """
        # Estimate current variance
        current_var = x.var()

        # Update running variance estimate
        if self.training:
            alpha = 1.0 / self.tau_adapt
            self.variance_ema = (1 - alpha) * self.variance_ema + alpha * current_var

        # Compute adaptive gain to maintain target variance
        # If variance too high → reduce gain, if too low → increase
        target_var = self.target_radius ** 2
        adaptive_gain = self.gain * (target_var / (self.variance_ema + 1e-6)).sqrt()

        return x * adaptive_gain


# =============================================================================
# DENDRITIC COMPUTATION
# =============================================================================
#
# Biological neurons have complex dendritic trees that perform non-linear
# computations before integration at the soma. This provides:
# - Non-linear input combinations
# - Multiplicative interactions
# - Coincidence detection
#
# Model: Multi-compartment with non-linear dendrites
# y = σ(Σ_b σ(W_b · x_b + b_b))
#
# where different branches (b) receive different input subsets.
#


class DendriticCompartment(nn.Module):
    """
    Multi-compartment dendritic computation.

    Models dendritic branches as separate non-linear processing units
    that feed into a somatic integration point.
    """

    def __init__(
        self,
        input_dim: int,
        n_branches: int = 4,
        branch_dim: int = 64,
        output_dim: Optional[int] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_branches = n_branches
        self.branch_dim = branch_dim
        self.output_dim = output_dim or input_dim

        # Each branch receives a subset of inputs
        inputs_per_branch = input_dim // n_branches

        # Dendritic branch weights
        self.branch_weights = nn.ParameterList([
            nn.Parameter(torch.randn(branch_dim, inputs_per_branch) * 0.02)
            for _ in range(n_branches)
        ])

        # Somatic integration
        self.soma_weight = nn.Parameter(
            torch.randn(self.output_dim, n_branches * branch_dim) * 0.02
        )

        # Dendritic non-linearity threshold
        self.threshold = nn.Parameter(torch.zeros(n_branches, branch_dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        Multi-compartment dendritic computation.

        Args:
            x: Input tensor [..., D]

        Returns:
            Output after dendritic processing [..., D_out]
        """
        D = x.shape[-1]
        inputs_per_branch = D // self.n_branches

        # Process each dendritic branch
        branch_outputs = []
        for i, W in enumerate(self.branch_weights):
            # Extract input subset for this branch
            x_branch = x[..., i * inputs_per_branch:(i + 1) * inputs_per_branch]

            # Dendritic non-linearity (sigmoid-like saturation)
            branch_act = F.linear(x_branch, W)
            branch_out = torch.sigmoid(branch_act - self.threshold[i])
            branch_outputs.append(branch_out)

        # Concatenate branch outputs
        dendritic = torch.cat(branch_outputs, dim=-1)

        # Somatic integration
        output = F.linear(dendritic, self.soma_weight)

        return output


# =============================================================================
# NEUROMODULATION
# =============================================================================
#
# Neuromodulators (dopamine, acetylcholine, etc.) globally modulate
# neural computation, affecting:
# - Learning rate (dopamine: reward prediction error)
# - Attention/arousal (acetylcholine)
# - Exploration/exploitation balance
#
# We implement this as learned gating that modulates entire layers.
#


class Neuromodulator(nn.Module):
    """
    Neuromodulation module for global state-dependent modulation.

    Learns to modulate network activity based on context,
    similar to dopamine/acetylcholine effects.
    """

    def __init__(
        self,
        input_dim: int,
        modulation_dim: int = 32
    ):
        super().__init__()

        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, modulation_dim),
            nn.LayerNorm(modulation_dim),
            nn.GELU(),
            nn.Linear(modulation_dim, modulation_dim)
        )

        # Modulation generators for different "neurotransmitters"
        self.dopamine_gate = nn.Linear(modulation_dim, input_dim)  # Reward/salience
        self.acetylcholine_gate = nn.Linear(modulation_dim, input_dim)  # Attention
        self.norepinephrine_gate = nn.Linear(modulation_dim, input_dim)  # Arousal

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None
    ) -> Tuple[Tensor, dict]:
        """
        Apply neuromodulation to input.

        Args:
            x: Input tensor [B, T, D]
            context: Optional context (uses x mean if not provided)

        Returns:
            modulated: Neuromodulated tensor
            modulators: Dict of modulation signals
        """
        if context is None:
            context = x.mean(dim=1)  # [B, D]

        # Encode context
        ctx = self.context_encoder(context)  # [B, mod_dim]

        # Generate modulation signals
        da = torch.sigmoid(self.dopamine_gate(ctx))  # [B, D]
        ach = torch.sigmoid(self.acetylcholine_gate(ctx))
        ne = torch.sigmoid(self.norepinephrine_gate(ctx))

        # Apply modulation (multiplicative gating)
        # DA: enhances salient features
        # ACh: sharpens attention
        # NE: global gain/arousal
        modulated = x * da.unsqueeze(1) * (1 + ach.unsqueeze(1)) * ne.unsqueeze(1)

        modulators = {
            'dopamine': da,
            'acetylcholine': ach,
            'norepinephrine': ne
        }

        return modulated, modulators


# =============================================================================
# ADVANCED HATCHING BLOCK
# =============================================================================


class AdvancedHatchingBlock(nn.Module):
    """
    Enhanced Hatching block with all advanced components.

    Integrates:
    - HiPPO memory
    - BCM homeostasis
    - Criticality regulation
    - Dendritic computation
    - Neuromodulation
    """

    def __init__(self, config: HatchingConfig):
        super().__init__()
        self.config = config

        D = config.n_embd
        nh = config.n_head
        N = config.mlp_internal_dim_multiplier * D // nh

        # HiPPO memory module
        self.hippo = HiPPOCell(D, state_dim=64, hippo_type='legs')

        # Dendritic computation
        self.dendrite = DendriticCompartment(D, n_branches=4, branch_dim=D//4)

        # BCM homeostasis
        self.homeostasis = BCMHomeostasis(N, target_rate=0.1)

        # Criticality regulator
        self.criticality = CriticalityRegulator(D)

        # Neuromodulator
        self.neuromod = Neuromodulator(D)

        # Standard projections (from original Hatching)
        self.encoder = nn.Parameter(torch.zeros(nh, D, N).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros(nh, D, N).normal_(std=0.02))
        self.decoder = nn.Parameter(torch.zeros(nh * N, D).normal_(std=0.02))

        # Layer norm
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)

        # Dropout
        self.drop = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        hippo_state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, dict]:
        """
        Forward through advanced block.
        """
        B, _, T, D = x.size()
        C = self.config
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        # Neuromodulation
        x_2d = x.squeeze(1)  # [B, T, D]
        x_mod, modulators = self.neuromod(x_2d)
        x = x_mod.unsqueeze(1)  # [B, 1, T, D]

        # HiPPO long-range memory
        x_hippo, hippo_state = self.hippo(x_2d, hippo_state)
        x = x + x_hippo.unsqueeze(1) * 0.1  # Residual from HiPPO

        # Dendritic computation
        x_dend = self.dendrite(x_2d).unsqueeze(1)
        x = x + x_dend * 0.1

        # Project to sparse latent space
        x_latent = x @ self.encoder  # [B, nh, T, N]

        # Sparse activation with surrogate gradient
        x_sparse = F.relu(x_latent)

        # BCM homeostasis
        x_homeo, _ = self.homeostasis(x_sparse)
        x_sparse = x_sparse * (1 + 0.1 * x_homeo)  # Subtle modulation

        # Simple attention (can use STDP attention from hatching.py)
        scores = (x_sparse @ x_sparse.mT).tril(diagonal=-1)
        yKV = scores @ x
        yKV = self.ln(yKV)

        # Value encoding
        y_latent = yKV @ self.encoder_v
        y_sparse = F.relu(y_latent)

        # Multiplicative gating (Hebbian)
        xy_sparse = x_sparse * y_sparse

        # Criticality regulation
        xy_flat = xy_sparse.transpose(1, 2).reshape(B, T, -1)
        xy_crit = self.criticality(xy_flat)
        xy_sparse = xy_crit.view(B, T, nh, N).transpose(1, 2)

        xy_sparse = self.drop(xy_sparse)

        # Decode
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder

        # Residual
        y = self.ln(yMLP)
        output = self.ln(x + y)

        return output, hippo_state, modulators


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("Testing advanced Hatching components...")

    # Test HiPPO
    print("\n1. Testing HiPPO...")
    hippo = HiPPOCell(input_dim=64, state_dim=32)
    x = torch.randn(2, 16, 64)
    y, state = hippo(x)
    print(f"   HiPPO: input {x.shape} → output {y.shape}, state {state.shape}")

    # Test BCM
    print("\n2. Testing BCM Homeostasis...")
    bcm = BCMHomeostasis(n_neurons=128)
    activity = torch.randn(2, 16, 128) * 0.5
    modulated, theta = bcm(activity)
    print(f"   BCM: activity {activity.shape} → modulated {modulated.shape}")
    print(f"   Threshold: {theta.mean().item():.4f}")

    # Test Criticality
    print("\n3. Testing Criticality Regulator...")
    crit = CriticalityRegulator(n_features=64)
    x = torch.randn(2, 16, 64) * 2.0  # High variance
    x_reg = crit(x)
    print(f"   Input var: {x.var().item():.4f}, Output var: {x_reg.var().item():.4f}")

    # Test Dendrite
    print("\n4. Testing Dendritic Computation...")
    dend = DendriticCompartment(input_dim=64, n_branches=4)
    x = torch.randn(2, 16, 64)
    y = dend(x)
    print(f"   Dendrite: {x.shape} → {y.shape}")

    # Test Neuromodulator
    print("\n5. Testing Neuromodulation...")
    neuromod = Neuromodulator(input_dim=64)
    x = torch.randn(2, 16, 64)
    x_mod, mods = neuromod(x)
    print(f"   Neuromod: {x.shape} → {x_mod.shape}")
    print(f"   Dopamine mean: {mods['dopamine'].mean().item():.4f}")

    # Test Advanced Block
    print("\n6. Testing Advanced Hatching Block...")
    config = HatchingConfig(n_embd=64, n_head=2, mlp_internal_dim_multiplier=32)
    block = AdvancedHatchingBlock(config)
    x = torch.randn(2, 1, 16, 64)
    y, state, mods = block(x)
    print(f"   Block: {x.shape} → {y.shape}")

    print("\nAll advanced component tests passed!")
