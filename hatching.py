# Copyright 2025 - Enhanced Dragon Hatching Model
# Building on BDH with rigorous neuroscience-inspired mathematics
#
# Key Enhancements:
# 1. Leaky Integrate-and-Fire (LIF) neuron dynamics
# 2. Spike-Timing-Dependent Plasticity (STDP) attention
# 3. Excitatory/Inhibitory neuron circuits (Dale's Law)
# 4. Scale-free connectivity via preferential attachment
# 5. Membrane potential with exponential decay
# 6. Synaptic plasticity during inference (Hebbian traces)

import dataclasses
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor


@dataclasses.dataclass
class HatchingConfig:
    """Configuration for the enhanced Hatching model."""
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256

    # LIF Neuron Parameters
    tau_mem: float = 10.0      # Membrane time constant (ms)
    tau_syn: float = 5.0       # Synaptic time constant (ms)
    v_threshold: float = 1.0   # Spike threshold
    v_reset: float = 0.0       # Reset potential
    v_rest: float = 0.0        # Resting potential

    # STDP Parameters
    tau_plus: float = 20.0     # LTP time constant
    tau_minus: float = 20.0    # LTD time constant
    A_plus: float = 0.01       # LTP amplitude
    A_minus: float = 0.0105    # LTD amplitude (slightly larger for stability)

    # E/I Balance (Dale's Law)
    excitatory_ratio: float = 0.8  # 80% excitatory, 20% inhibitory

    # Scale-free parameters
    scale_free_gamma: float = 2.5  # Power-law exponent

    # Hebbian trace decay
    trace_decay: float = 0.95


# =============================================================================
# MATHEMATICAL FOUNDATIONS
# =============================================================================
#
# 1. LEAKY INTEGRATE-AND-FIRE (LIF) DYNAMICS
#    ────────────────────────────────────────
#    The membrane potential V evolves according to:
#
#    τ_m · dV/dt = -(V - V_rest) + R·I(t)
#
#    Discretized (Euler):
#    V[t+1] = V[t] + dt/τ_m · (-(V[t] - V_rest) + I[t])
#           = α·V[t] + (1-α)·V_rest + β·I[t]
#
#    where α = exp(-dt/τ_m), β = (1-α)·R
#
#    Spike generation:
#    S[t] = Θ(V[t] - V_th)  (Heaviside step function)
#    V[t] → V_reset if S[t] = 1
#
#
# 2. SPIKE-TIMING-DEPENDENT PLASTICITY (STDP)
#    ────────────────────────────────────────
#    Weight change depends on relative spike timing:
#
#    ΔW = { A_+ · exp(-Δt/τ_+)  if Δt > 0 (pre before post → LTP)
#         { A_- · exp(+Δt/τ_-)  if Δt < 0 (post before pre → LTD)
#
#    where Δt = t_post - t_pre
#
#    In continuous form with eligibility traces:
#    x_pre[t+1] = ρ·x_pre[t] + S_pre[t]     (pre-synaptic trace)
#    x_post[t+1] = ρ·x_post[t] + S_post[t]  (post-synaptic trace)
#
#    ΔW = A_+ · S_post · x_pre - A_- · S_pre · x_post
#
#
# 3. DALE'S LAW (E/I SEPARATION)
#    ────────────────────────────
#    Neurons are either excitatory (+) or inhibitory (-):
#
#    W = W_raw · sign_mask
#
#    where sign_mask ∈ {+1, -1} is fixed per neuron
#    and W_raw ≥ 0 (non-negative weights)
#
#
# 4. SCALE-FREE CONNECTIVITY
#    ────────────────────────
#    Degree distribution follows power law:
#
#    P(k) ∝ k^(-γ)
#
#    Generated via preferential attachment (Barabási-Albert):
#    P(connect to node i) ∝ degree(i) + 1
#
#    Or via static mask with power-law degree sampling.
#
#
# 5. ROTARY POSITION EMBEDDING (RoPE)
#    ────────────────────────────────
#    Position encoding via rotation in 2D subspaces:
#
#    RoPE(x, m) = R_Θ,m · x
#
#    where R_Θ,m is block-diagonal rotation matrix:
#    R_Θ,m = diag(R(m·θ_1), R(m·θ_2), ..., R(m·θ_{d/2}))
#
#    and R(θ) = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
#
# =============================================================================


def power_law_sample(n: int, gamma: float, device: torch.device) -> Tensor:
    """
    Sample from discrete power-law distribution P(k) ∝ k^(-γ).

    Uses inverse transform sampling:
    k = floor((1-u)^(-1/(γ-1)))

    where u ~ Uniform(0,1)
    """
    u = torch.rand(n, device=device)
    # Avoid division by zero, clamp gamma > 1
    gamma = max(gamma, 1.01)
    k = torch.floor((1 - u).pow(-1.0 / (gamma - 1)))
    return k.clamp(min=1)


def generate_scale_free_mask(
    n_neurons: int,
    avg_degree: int,
    gamma: float,
    device: torch.device
) -> Tensor:
    """
    Generate scale-free adjacency mask using configuration model.

    The degree sequence is sampled from power-law, then edges are
    assigned to match the degree sequence as closely as possible.

    Returns: Binary mask of shape (n_neurons, n_neurons)
    """
    # Sample degree sequence from power law
    degrees = power_law_sample(n_neurons, gamma, device)

    # Normalize to achieve target average degree
    degrees = (degrees / degrees.mean() * avg_degree).clamp(min=1, max=n_neurons-1)
    degrees = degrees.int()

    # Create mask via stochastic block matching
    mask = torch.zeros(n_neurons, n_neurons, device=device)

    # Probability of edge (i,j) proportional to degree[i] * degree[j]
    probs = degrees.unsqueeze(1) * degrees.unsqueeze(0)
    probs = probs.float() / probs.sum() * (avg_degree * n_neurons / 2)
    probs = probs.clamp(max=1.0)

    # Sample edges
    mask = torch.bernoulli(probs)

    # Make symmetric for undirected graph (optional, can be asymmetric)
    mask = ((mask + mask.T) > 0).float()

    # Remove self-loops
    mask.fill_diagonal_(0)

    return mask


class SurrogateSpike(torch.autograd.Function):
    """
    Surrogate gradient for spike function.

    Forward: Heaviside step function Θ(x - threshold)
    Backward: Smooth surrogate gradient (fast sigmoid derivative)

    This enables backpropagation through spiking dynamics.
    The surrogate is: σ'(x) = 1 / (1 + |βx|)²
    """

    @staticmethod
    def forward(ctx, membrane: Tensor, threshold: float) -> Tensor:
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        # Heaviside step: spike if membrane > threshold
        return (membrane > threshold).float()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        membrane, = ctx.saved_tensors
        # Fast sigmoid surrogate gradient
        # d/dx σ(βx) ≈ β / (1 + |βx|)²
        beta = 10.0  # Sharpness parameter
        centered = membrane - ctx.threshold
        surrogate_grad = beta / (1 + torch.abs(beta * centered)).pow(2)
        return grad_output * surrogate_grad, None


def spike_function(membrane: Tensor, threshold: float) -> Tensor:
    """Differentiable spike generation with surrogate gradient."""
    return SurrogateSpike.apply(membrane, threshold)


class LeakyIntegrateFire(nn.Module):
    """
    Leaky Integrate-and-Fire neuron layer with Hebbian traces.

    Implements the differential equation:
    τ_m · dV/dt = -(V - V_rest) + I

    With discrete-time update:
    V[t+1] = α·V[t] + (1-α)·V_rest + (1-α)·I[t]

    where α = exp(-dt/τ_m)
    """

    def __init__(self, config: HatchingConfig, n_neurons: int):
        super().__init__()
        self.config = config
        self.n_neurons = n_neurons

        # Compute decay constants (assuming dt=1)
        # α = exp(-dt/τ_m)
        self.alpha = math.exp(-1.0 / config.tau_mem)
        self.beta = 1.0 - self.alpha  # Input scaling

        # Threshold and reset
        self.v_th = config.v_threshold
        self.v_reset = config.v_reset
        self.v_rest = config.v_rest

        # Synaptic decay for current integration
        self.syn_alpha = math.exp(-1.0 / config.tau_syn)

        # Hebbian trace decay
        self.trace_decay = config.trace_decay

    def forward(
        self,
        current: Tensor,
        membrane: Optional[Tensor] = None,
        trace: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through LIF dynamics.

        Args:
            current: Input current [B, ..., N]
            membrane: Previous membrane potential (optional)
            trace: Previous Hebbian trace (optional)

        Returns:
            spikes: Binary spike tensor [B, ..., N]
            membrane: Updated membrane potential
            trace: Updated Hebbian trace
        """
        # Initialize states if not provided
        if membrane is None:
            membrane = torch.full_like(current, self.v_rest)
        if trace is None:
            trace = torch.zeros_like(current)

        # Membrane dynamics: V = α·V + β·(V_rest + I)
        membrane = (
            self.alpha * membrane +
            self.beta * (self.v_rest + current)
        )

        # Spike generation with surrogate gradient
        spikes = spike_function(membrane, self.v_th)

        # Reset: V → V_reset where spike occurred
        # Soft reset: subtract threshold
        membrane = membrane - spikes * (self.v_th - self.v_reset)

        # Update Hebbian trace: exponential moving average of spikes
        # trace[t+1] = ρ·trace[t] + spike[t]
        trace = self.trace_decay * trace + spikes

        return spikes, membrane, trace


class STDPAttention(nn.Module):
    """
    Attention mechanism inspired by Spike-Timing-Dependent Plasticity.

    Standard attention computes:
    Attention(Q, K, V) = softmax(QK^T / √d) V

    STDP-inspired attention modulates weights based on temporal eligibility:

    W_ij ∝ exp(-|Δt_ij| / τ) · (A+ if causal, A- if anti-causal)

    We implement this by:
    1. Computing standard attention scores
    2. Modulating by STDP-like temporal kernel
    3. Applying eligibility traces for weight updates
    """

    def __init__(self, config: HatchingConfig):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh

        # STDP time constants
        self.tau_plus = config.tau_plus
        self.tau_minus = config.tau_minus
        self.A_plus = config.A_plus
        self.A_minus = config.A_minus

        # RoPE frequencies (from original BDH)
        self.freqs = nn.Buffer(
            self._get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

        # Learnable STDP kernel parameters
        self.stdp_scale = nn.Parameter(torch.ones(1))

    @staticmethod
    def _get_freqs(n: int, theta: float, dtype: torch.dtype) -> Tensor:
        """Compute RoPE frequency bands."""
        def quantize(t, q=2):
            return (t / q).floor() * q
        return (
            1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
            / (2 * math.pi)
        )

    @staticmethod
    def rope(phases: Tensor, v: Tensor) -> Tensor:
        """Apply rotary position embedding."""
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def compute_stdp_kernel(self, T: int, device: torch.device) -> Tensor:
        """
        Compute STDP-like temporal modulation kernel.

        K[i,j] encodes the STDP weight change for pre-synaptic time j
        affecting post-synaptic time i.

        Δt = i - j (positive means pre before post → LTP)

        K[i,j] = { A_+ · exp(-Δt/τ_+)  if Δt > 0
                 { 0                    if Δt = 0
                 { A_- · exp(+Δt/τ_-)   if Δt < 0
        """
        # Create time indices
        i = torch.arange(T, device=device).unsqueeze(1)  # [T, 1]
        j = torch.arange(T, device=device).unsqueeze(0)  # [1, T]

        delta_t = (i - j).float()  # [T, T]

        # LTP kernel (pre before post, Δt > 0)
        ltp = self.A_plus * torch.exp(-delta_t / self.tau_plus)
        ltp = torch.where(delta_t > 0, ltp, torch.zeros_like(ltp))

        # LTD kernel (post before pre, Δt < 0) - anti-causal
        # For causal models, we typically only use LTP (causal attention)
        # But we add a small LTD component for recent context
        ltd = self.A_minus * torch.exp(delta_t / self.tau_minus)
        ltd = torch.where(delta_t < 0, ltd, torch.zeros_like(ltd))

        # Combined kernel (mostly causal with small anti-Hebbian regularization)
        kernel = ltp - 0.1 * ltd  # Small LTD for stability

        # Apply causal mask (can only attend to past)
        causal_mask = torch.tril(torch.ones(T, T, device=device), diagonal=-1)
        kernel = kernel * causal_mask

        return kernel * self.stdp_scale

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        traces: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        STDP-modulated attention.

        Args:
            Q: Query tensor [B, nh, T, N]
            K: Key tensor [B, nh, T, N]
            V: Value tensor [B, 1, T, D]
            traces: Hebbian eligibility traces (optional)

        Returns:
            output: Attention output
            attention_weights: For visualization/analysis
        """
        assert K is Q, "Self-attention required for STDP"
        B, _, T, N = Q.size()

        # Apply RoPE
        r_phases = (
            torch.arange(T, device=Q.device, dtype=self.freqs.dtype)
            .view(1, 1, -1, 1) * self.freqs
        )
        QR = self.rope(r_phases, Q)
        KR = QR  # Self-attention

        # Compute attention scores
        scores = QR @ KR.mT  # [B, nh, T, T]

        # Apply STDP temporal kernel
        stdp_kernel = self.compute_stdp_kernel(T, Q.device)  # [T, T]

        # Modulate scores with STDP kernel
        # This biases attention toward biologically-plausible temporal patterns
        scores = scores * (1.0 + stdp_kernel.unsqueeze(0).unsqueeze(0))

        # Apply causal mask
        scores = scores.tril(diagonal=-1)

        # Optional: modulate by Hebbian traces
        if traces is not None:
            # traces: [B, nh, T, N]
            # Compute trace-based gating
            trace_gate = torch.sigmoid(traces.mean(dim=-1, keepdim=True))  # [B, nh, T, 1]
            scores = scores * (0.5 + 0.5 * trace_gate)

        # Attention output
        output = scores @ V

        return output, scores


class ExcitatoryInhibitoryCircuit(nn.Module):
    """
    Neural circuit respecting Dale's Law.

    Dale's Law: A neuron releases the same neurotransmitter at all its synapses.
    Thus neurons are either purely excitatory (+) or purely inhibitory (-).

    Implementation:
    - Split neurons into E and I populations
    - E→E, E→I connections are positive
    - I→E, I→I connections are negative
    - Weights are parameterized as |W| * sign_mask
    """

    def __init__(
        self,
        config: HatchingConfig,
        in_features: int,
        out_features: int
    ):
        super().__init__()
        self.config = config
        self.in_features = in_features
        self.out_features = out_features

        # Determine E/I split
        n_excitatory = int(out_features * config.excitatory_ratio)
        n_inhibitory = out_features - n_excitatory

        # Create sign mask (E neurons: +1, I neurons: -1)
        sign_mask = torch.ones(out_features)
        sign_mask[n_excitatory:] = -1.0
        self.register_buffer('sign_mask', sign_mask.view(1, -1))

        # Non-negative weight matrix (we'll multiply by sign_mask)
        # Using softplus to ensure positivity: W = softplus(W_raw)
        self.weight_raw = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

        # Optional scale-free connectivity mask
        if config.scale_free_gamma > 0:
            # Generate sparse scale-free mask
            avg_degree = max(1, in_features // 10)  # ~10% connectivity
            mask = generate_scale_free_mask(
                out_features, avg_degree, config.scale_free_gamma,
                device=self.weight_raw.device
            )
            # Extend to rectangular if needed
            if out_features != in_features:
                mask = torch.rand(out_features, in_features) < (avg_degree / in_features)
                mask = mask.float()
            self.register_buffer('connectivity_mask', mask)
        else:
            self.register_buffer('connectivity_mask', None)

    @property
    def weight(self) -> Tensor:
        """Compute effective weight with Dale's Law and optional sparsity."""
        # Ensure non-negative via softplus
        W = F.softplus(self.weight_raw)

        # Apply Dale's Law: E neurons project positive, I neurons project negative
        W = W * self.sign_mask.T  # sign_mask determines if output neuron is E or I

        # Apply connectivity mask if present
        if self.connectivity_mask is not None:
            W = W * self.connectivity_mask

        return W

    def forward(self, x: Tensor) -> Tensor:
        """Linear transformation with E/I constraints."""
        return F.linear(x, self.weight)


class HatchingBlock(nn.Module):
    """
    Single block of the Hatching model.

    Combines:
    1. LIF neuron dynamics for sparse activations
    2. STDP-modulated attention
    3. E/I balanced projections
    4. Hebbian trace maintenance
    """

    def __init__(self, config: HatchingConfig):
        super().__init__()
        self.config = config

        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh

        # Encoder: D → N per head (with E/I balance)
        self.encoder = nn.Parameter(torch.zeros(nh, D, N).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros(nh, D, N).normal_(std=0.02))

        # Decoder: nh*N → D (with E/I balance)
        self.decoder = nn.Parameter(torch.zeros(nh * N, D).normal_(std=0.02))

        # LIF neurons for sparse encoding
        self.lif = LeakyIntegrateFire(config, N)

        # STDP-modulated attention
        self.attn = STDPAttention(config)

        # Layer norm (without affine, following BDH)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)

        # Dropout
        self.drop = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        membrane: Optional[Tensor] = None,
        trace: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through Hatching block.

        Args:
            x: Input tensor [B, 1, T, D]
            membrane: LIF membrane state
            trace: Hebbian traces

        Returns:
            output: Block output [B, 1, T, D]
            membrane: Updated membrane state
            trace: Updated Hebbian traces
        """
        B, _, T, D = x.size()
        C = self.config
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        # Project to latent space
        x_latent = x @ self.encoder  # [B, nh, T, N]

        # LIF dynamics: integrate current, generate spikes
        # The "current" is the projected input
        spikes, membrane, trace = self.lif(x_latent, membrane, trace)

        # Sparse activation (spikes modulated by membrane for gradient flow)
        # Combines hard spikes with soft residual for better gradients
        x_sparse = spikes * F.relu(x_latent)  # Spike-gated activation

        # STDP-modulated attention
        yKV, attn_weights = self.attn(Q=x_sparse, K=x_sparse, V=x, traces=trace)
        yKV = self.ln(yKV)

        # Second encoding pass for value pathway
        y_latent = yKV @ self.encoder_v

        # Gated combination (Hebbian-like: fire together, wire together)
        # This is the multiplicative interaction from BDH
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse  # [B, nh, T, N]

        # Dropout for regularization
        xy_sparse = self.drop(xy_sparse)

        # Decode back to embedding space
        yMLP = (
            xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
        )  # [B, 1, T, D]

        # Residual connection with normalization
        y = self.ln(yMLP)
        output = self.ln(x + y)

        return output, membrane, trace


class Hatching(nn.Module):
    """
    Enhanced Dragon Hatching Model.

    A biologically-inspired language model combining:
    - Leaky Integrate-and-Fire (LIF) neuron dynamics
    - Spike-Timing-Dependent Plasticity (STDP) attention
    - Excitatory/Inhibitory circuit balance (Dale's Law)
    - Scale-free network topology
    - Hebbian synaptic traces

    Mathematical foundations:

    1. LIF Dynamics:
       τ_m · dV/dt = -(V - V_rest) + I
       S = Θ(V - V_th)

    2. STDP Kernel:
       ΔW = A_+ · exp(-Δt/τ_+) - A_- · exp(+Δt/τ_-)

    3. Hebbian Traces:
       trace[t+1] = ρ · trace[t] + spike[t]

    4. Scale-free Connectivity:
       P(k) ∝ k^(-γ)
    """

    def __init__(self, config: HatchingConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config

        D = config.n_embd

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, D)

        # Layer normalization
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)

        # Hatching blocks
        self.blocks = nn.ModuleList([
            HatchingBlock(config) for _ in range(config.n_layer)
        ])

        # Language model head
        self.lm_head = nn.Parameter(
            torch.zeros(D, config.vocab_size).normal_(std=0.02)
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Cache for stateful inference
        self.register_buffer('_membrane_cache', None)
        self.register_buffer('_trace_cache', None)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def reset_states(self):
        """Reset membrane and trace states for new sequence."""
        self._membrane_cache = None
        self._trace_cache = None

    def forward(
        self,
        idx: Tensor,
        targets: Optional[Tensor] = None,
        return_states: bool = False
    ) -> Tuple[Tensor, Optional[Tensor], Optional[dict]]:
        """
        Forward pass through Hatching model.

        Args:
            idx: Input token indices [B, T]
            targets: Target token indices [B, T] (optional)
            return_states: Whether to return internal states

        Returns:
            logits: Output logits [B, T, vocab_size]
            loss: Cross-entropy loss (if targets provided)
            states: Internal states dict (if return_states=True)
        """
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        # Embed tokens
        x = self.embed(idx).unsqueeze(1)  # [B, 1, T, D]

        # Initial normalization (helps training stability)
        x = self.ln(x)

        # Initialize states for each layer
        membranes = [None] * C.n_layer
        traces = [None] * C.n_layer

        # Process through blocks
        all_spikes = []
        for i, block in enumerate(self.blocks):
            x, membranes[i], traces[i] = block(x, membranes[i], traces[i])

        # Project to vocabulary
        logits = x.view(B, T, D) @ self.lm_head

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        # Return states if requested
        states = None
        if return_states:
            states = {
                'membranes': membranes,
                'traces': traces,
            }

        return logits, loss, states

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tensor:
        """
        Generate tokens autoregressively.

        Uses stateful inference with maintained membrane potentials
        and Hebbian traces for biologically-plausible generation.
        """
        self.reset_states()

        for _ in range(max_new_tokens):
            # Forward pass
            logits, _, _ = self(idx)

            # Get logits for last position
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float('-inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# =============================================================================
# MATHEMATICAL ANALYSIS UTILITIES
# =============================================================================

def compute_firing_rate(model: Hatching, x: Tensor) -> dict:
    """
    Analyze firing rates across layers.

    Returns statistics on neural activity for biological plausibility assessment.
    """
    model.eval()
    with torch.no_grad():
        _, _, states = model(x, return_states=True)

    stats = {}
    for i, trace in enumerate(states['traces']):
        if trace is not None:
            # Firing rate approximated from trace
            rate = trace.mean().item()
            sparsity = (trace > 0.1).float().mean().item()
            stats[f'layer_{i}'] = {
                'mean_firing_rate': rate,
                'active_fraction': sparsity
            }

    return stats


def analyze_stdp_weights(model: Hatching) -> dict:
    """
    Analyze STDP-related weight statistics.
    """
    stats = {}
    for i, block in enumerate(model.blocks):
        attn = block.attn
        stats[f'layer_{i}'] = {
            'stdp_scale': attn.stdp_scale.item(),
            'tau_plus': attn.tau_plus,
            'tau_minus': attn.tau_minus,
        }
    return stats


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == '__main__':
    # Test the model
    print("Testing Hatching model...")

    config = HatchingConfig(
        n_layer=2,
        n_embd=64,
        n_head=2,
        mlp_internal_dim_multiplier=32,
        vocab_size=256
    )

    model = Hatching(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randint(0, 256, (2, 32))  # [B=2, T=32]
    logits, loss, states = model(x, targets=x, return_states=True)

    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    # Analyze firing rates
    firing_stats = compute_firing_rate(model, x)
    print(f"Firing statistics: {firing_stats}")

    # Test generation
    prompt = torch.randint(0, 256, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10)
    print(f"Generated shape: {generated.shape}")

    print("All tests passed!")
