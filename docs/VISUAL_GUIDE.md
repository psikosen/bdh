# BDH Visual Guide - Diagrams and Illustrations

A visual companion to the Architecture document.

---

## 1. High-Level Architecture

```
                              BABY DRAGON HATCHLING
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║   Input: "The dragon breathes fire"                                   ║
    ║                                                                       ║
    ║       ┌─────┐ ┌────────┐ ┌──────────┐ ┌──────┐                       ║
    ║       │ The │ │ dragon │ │ breathes │ │ fire │                       ║
    ║       └──┬──┘ └───┬────┘ └────┬─────┘ └──┬───┘                       ║
    ║          │        │           │          │                            ║
    ║          ▼        ▼           ▼          ▼                            ║
    ║   ┌──────────────────────────────────────────┐                       ║
    ║   │           EMBEDDING LAYER                │                       ║
    ║   │  Token IDs → Dense Vectors (256-dim)     │                       ║
    ║   └──────────────────┬───────────────────────┘                       ║
    ║                      │                                                ║
    ║                      ▼                                                ║
    ║   ┌──────────────────────────────────────────┐                       ║
    ║   │              LAYER 1                      │                       ║
    ║   │  ┌─────────────┐  ┌──────────────────┐   │                       ║
    ║   │  │   SPARSE    │  │  STDP ATTENTION  │   │                       ║
    ║   │  │    MLP      │→→│   (Temporal)     │   │                       ║
    ║   │  │  (LIF)      │  │                  │   │                       ║
    ║   │  └─────────────┘  └──────────────────┘   │                       ║
    ║   └──────────────────┬───────────────────────┘                       ║
    ║                      │                                                ║
    ║                      ⋮  (Repeat N layers)                            ║
    ║                      │                                                ║
    ║                      ▼                                                ║
    ║   ┌──────────────────────────────────────────┐                       ║
    ║   │            OUTPUT HEAD                    │                       ║
    ║   │  Dense Vectors → Vocabulary Logits       │                       ║
    ║   └──────────────────┬───────────────────────┘                       ║
    ║                      │                                                ║
    ║                      ▼                                                ║
    ║                                                                       ║
    ║   Output: Probability distribution over next token                    ║
    ║           P("and") = 0.23, P("the") = 0.18, P(".") = 0.15...        ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
```

---

## 2. Single Layer Detail

```
                           BDH/HATCHING LAYER
    ═══════════════════════════════════════════════════════════════════

    Input x [B, T, D]
        │
        │
        ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                    SPARSE ENCODER                             │
    │                                                               │
    │   x ────────► [Encoder Matrix] ────────► x_latent             │
    │   [B,T,D]      [D × N×nh]               [B,nh,T,N]            │
    │                                                               │
    │               N = D × 128 / nh  (e.g., 8192)                  │
    │               nh = number of heads (e.g., 4)                  │
    └───────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                    SPARSE ACTIVATION                          │
    │                                                               │
    │   x_latent ────────► ReLU ────────► x_sparse                  │
    │                        │                                      │
    │                        │  ┌─────────────────────────────┐     │
    │                        │  │ Values:                     │     │
    │                        └─►│  Before: [-0.3, 0.7, -0.1, 0.9]│   │
    │                           │  After:  [0.0, 0.7, 0.0, 0.9] │   │
    │                           │  (~50-90% become zero!)      │     │
    │                           └─────────────────────────────┘     │
    │                                                               │
    │   [Hatching Only] LIF Neuron Dynamics:                        │
    │   ┌──────────────────────────────────────┐                    │
    │   │  membrane = α·membrane + β·input     │                    │
    │   │  spike = (membrane > threshold)      │                    │
    │   │  membrane -= spike × (thresh-reset)  │                    │
    │   └──────────────────────────────────────┘                    │
    └───────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                    STDP ATTENTION                             │
    │                                                               │
    │   ┌─────────┐         ┌─────────────┐         ┌─────────┐    │
    │   │    Q    │         │   Scores    │         │    V    │    │
    │   │(sparse) │ ──────► │  Q × K^T    │ ◄────── │  (x)    │    │
    │   └────┬────┘         │  × STDP     │         └────┬────┘    │
    │        │              │   kernel    │              │         │
    │        │              └──────┬──────┘              │         │
    │        │                     │                     │         │
    │   Apply RoPE            Causal mask           Original x     │
    │   (position)            (lower tri)                          │
    │                              │                               │
    │                              ▼                               │
    │                         Attended                             │
    └───────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                    HEBBIAN GATING                             │
    │                                                               │
    │   x_sparse ──────┐                                            │
    │                  │                                            │
    │                  ▼                                            │
    │              ┌───────┐                                        │
    │              │   ×   │  ◄── Element-wise multiply             │
    │              └───┬───┘      "Fire together, wire together"    │
    │                  │                                            │
    │   y_sparse ──────┘                                            │
    │   (attended)                                                  │
    │                  │                                            │
    │                  ▼                                            │
    │              xy_sparse                                        │
    └───────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                    DECODER + RESIDUAL                         │
    │                                                               │
    │   xy_sparse ────► [Decoder Matrix] ────► y                    │
    │   [B,nh,T,N]         [N×nh × D]        [B,T,D]               │
    │                                                               │
    │                          │                                    │
    │                          ▼                                    │
    │                    ┌───────────┐                              │
    │             x ────►│    +      │────► output                  │
    │                    └───────────┘                              │
    │                     Residual                                  │
    │                     Connection                                │
    └───────────────────────────────────────────────────────────────┘
```

---

## 3. LIF Neuron Dynamics

```
                    LEAKY INTEGRATE-AND-FIRE NEURON
    ════════════════════════════════════════════════════════════════

    TIME ──────────────────────────────────────────────────────────►

    Input
    Current:    ▂▃▅▂▁▂▇▅▃▂▁▂▃▅▇▃▂▁▂▃▅▃▂▁▁▂▃▅▆▅▃▂▁

    Membrane    ──────────────────────────────────────── Threshold
    Potential:              ╱╲        ╱╲
                          ╱  ╲      ╱  ╲    ╱╲
                        ╱    ╲    ╱    ╲  ╱  ╲
                      ╱      ╲  ╱      ╲╱    ╲
               _____╱        ╲╱              ╲_____

    Output              │          │              │
    Spikes:     ────────█──────────█──────────────█───────
                       t₁         t₂             t₃


    EQUATIONS:
    ┌────────────────────────────────────────────────────────────┐
    │                                                            │
    │  1. Integration:    V[t+1] = α·V[t] + (1-α)·I[t]          │
    │                     where α = exp(-dt/τ_membrane)          │
    │                                                            │
    │  2. Spike:          S[t] = 1 if V[t] > V_threshold        │
    │                            0 otherwise                     │
    │                                                            │
    │  3. Reset:          V[t] = V[t] - S[t]·(V_th - V_reset)   │
    │                                                            │
    │  4. Trace:          trace[t+1] = ρ·trace[t] + S[t]        │
    │                     (memory of recent spikes)              │
    │                                                            │
    └────────────────────────────────────────────────────────────┘
```

---

## 4. STDP Kernel

```
                    SPIKE-TIMING-DEPENDENT PLASTICITY
    ════════════════════════════════════════════════════════════════

              Weight
              Change
              (ΔW)
                │
            LTP │     ╲
       (+)      │      ╲
                │       ╲
                │        ╲
    ────────────┼─────────╲───────────────────────► Δt = t_post - t_pre
       (-)      │          ╲
            LTD │           ╲
                │
                │

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  If pre-synaptic neuron fires BEFORE post-synaptic:         │
    │     Δt > 0  →  POTENTIATE (strengthen connection)           │
    │     ΔW = A₊ · exp(-Δt/τ₊)                                   │
    │                                                             │
    │  If post-synaptic neuron fires BEFORE pre-synaptic:         │
    │     Δt < 0  →  DEPRESS (weaken connection)                  │
    │     ΔW = -A₋ · exp(Δt/τ₋)                                   │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘


    STDP KERNEL MATRIX (for attention):

         t_pre:    1    2    3    4    5
                ┌────┬────┬────┬────┬────┐
    t_post: 1   │ 0  │    │    │    │    │   Zero diagonal
            2   │0.95│ 0  │    │    │    │   (no self-attention)
            3   │0.90│0.95│ 0  │    │    │
            4   │0.86│0.90│0.95│ 0  │    │   Exponential decay
            5   │0.81│0.86│0.90│0.95│ 0  │   with distance
                └────┴────┴────┴────┴────┘

    Values = A₊ · exp(-(t_post - t_pre) / τ₊)

    This biases attention toward recent, causally-related tokens.
```

---

## 5. Dale's Law (E/I Circuits)

```
                    EXCITATORY / INHIBITORY CIRCUITS
    ════════════════════════════════════════════════════════════════

    STANDARD NEURAL NETWORK:          DALE'S LAW (BDH):

       Any neuron can have              Neurons have FIXED type
       +/- connections                  (can't change sign)

         ┌───┐                            ┌───┐
         │ A ├──(+)──►                    │ E ├──(+)──► Always +
         │   ├──(-)──►                    │   ├──(+)──►
         └───┘                            └───┘

         ┌───┐                            ┌───┐
         │ B ├──(+)──►                    │ I ├──(-)──► Always -
         │   ├──(-)──►                    │   ├──(-)──►
         └───┘                            └───┘


    WEIGHT MATRIX STRUCTURE:

    Standard:                     Dale's Law:

    ┌─────────────────┐          ┌─────────────────┐
    │ +0.3  -0.2  +0.1│          │ +0.3  +0.2  -0.1│
    │ -0.4  +0.5  -0.3│          │ +0.4  +0.5  -0.3│
    │ +0.2  -0.1  +0.4│          │ +0.2  +0.1  -0.4│
    └─────────────────┘          └─────────────────┘
       Mixed signs                     │      │
       in each column              Excitatory  Inhibitory
                                   columns     columns
                                   (all +)     (all -)

    IMPLEMENTATION:

    W = softplus(W_raw) × sign_mask

    sign_mask = [+1, +1, +1, ..., -1, -1]
                ├──────────────┤  ├──────┤
                  80% Excitatory  20% Inhibitory
```

---

## 6. Scale-Free Connectivity

```
                    SCALE-FREE NETWORK TOPOLOGY
    ════════════════════════════════════════════════════════════════

    RANDOM NETWORK:                  SCALE-FREE NETWORK:
    (Erdős-Rényi)                    (Barabási-Albert)

         ●───●───●                        ●
        ╱│╲ ╱│╲ ╱│                       ╱│╲
       ● │ ● │ ● │                      ● │ ●
        ╲│╱ ╲│╱ ╲│                     ╱ │ ╲
         ●───●───●                    ●──●──●───●
                                        ╲│╱ │
    All nodes similar                    ●  ●──●
    degree (connections)                    ╲│╱
                                            HUB
                                            (many connections)

    DEGREE DISTRIBUTION:

    Random:                          Scale-Free:
    P(k)                             P(k)
      │                                │
      │    ▓▓▓                         │▓
      │   ▓▓▓▓▓                        │▓▓
      │  ▓▓▓▓▓▓▓                       │▓▓▓▓
      │ ▓▓▓▓▓▓▓▓▓                      │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
      └────────────► k                 └────────────────────► k
       Peaked (Poisson)                Power law: P(k) ∝ k^(-γ)


    WHY SCALE-FREE?

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  1. Biological brains have scale-free structure             │
    │                                                             │
    │  2. Hubs enable efficient information routing               │
    │                                                             │
    │  3. Robust to random failures (but vulnerable to hub loss)  │
    │                                                             │
    │  4. Enables "small world" property (short path lengths)     │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

---

## 7. Attention Pattern Comparison

```
                    ATTENTION PATTERNS
    ════════════════════════════════════════════════════════════════

    STANDARD TRANSFORMER:              BDH/HATCHING (STDP):

    Query: "The dragon breathes fire"  Query: "The dragon breathes fire"

           The dragon breathes fire           The dragon breathes fire
         ┌────┬────┬────┬────┬────┐        ┌────┬────┬────┬────┬────┐
    The  │    │    │    │    │    │   The  │    │    │    │    │    │
    drag │████│    │    │    │    │   drag │▓▓▓▓│    │    │    │    │
    brea │████│████│    │    │    │   brea │░░░░│▓▓▓▓│    │    │    │
    fire │████│████│████│    │    │   fire │    │░░░░│▓▓▓▓│    │    │
         └────┴────┴────┴────┴────┘        └────┴────┴────┴────┴────┘

    Softmax-normalized                 STDP-modulated
    (values sum to 1)                  (exponential decay)

    █ = Strong attention               ▓ = Strong (recent)
                                       ░ = Weak (older)


    DIFFERENCE:

    Standard: All past tokens equally "available"
              (softmax just normalizes)

    STDP:     Recent tokens get BONUS from STDP kernel
              Older tokens naturally decay
              Mimics biological temporal attention
```

---

## 8. Training Flow

```
                    TRAINING PIPELINE
    ════════════════════════════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────────┐
    │                                                              │
    │   DATA                                                       │
    │   ════                                                       │
    │                                                              │
    │   "The dragon breathes fire and smoke"                       │
    │      │                                                       │
    │      ▼                                                       │
    │   ┌────────────────────────────────────────────────────┐    │
    │   │ Input:  [The] [dragon] [breathes] [fire] [and]     │    │
    │   │ Target: [dragon] [breathes] [fire] [and] [smoke]   │    │
    │   └────────────────────────────────────────────────────┘    │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                                                              │
    │   FORWARD PASS                                               │
    │   ════════════                                               │
    │                                                              │
    │   Input ──► Embed ──► [Layer 1] ──► ... ──► [Layer N] ──►   │
    │                         │                      │             │
    │                    Update LIF             Update LIF         │
    │                    membrane               membrane           │
    │                         │                      │             │
    │                         ▼                      ▼             │
    │   ◄─────────────── Logits ◄────────────────────             │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                                                              │
    │   LOSS COMPUTATION                                           │
    │   ════════════════                                           │
    │                                                              │
    │   L_total = L_CE + λ₁·L_sparsity + λ₂·L_firing_rate         │
    │                                                              │
    │   L_CE = CrossEntropy(logits, targets)                       │
    │   L_sparsity = ||activations||₁  (encourage zeros)           │
    │   L_firing_rate = (rate - target_rate)²                      │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                                                              │
    │   BACKWARD PASS                                              │
    │   ═════════════                                              │
    │                                                              │
    │   ∂L/∂W ◄── Surrogate gradient through spikes                │
    │                                                              │
    │   SPIKE FUNCTION:                                            │
    │   ┌─────────────────────────────────────────────┐           │
    │   │  Forward:  s = Θ(V - V_th)  (step function) │           │
    │   │  Backward: ∂s/∂V ≈ β/(1+|β(V-V_th)|)²       │           │
    │   └─────────────────────────────────────────────┘           │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                                                              │
    │   OPTIMIZER STEP                                             │
    │   ══════════════                                             │
    │                                                              │
    │   W ← W - lr · ∂L/∂W  (with AdamW, gradient clipping)       │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
```

---

## 9. Generation Process

```
                    TEXT GENERATION
    ════════════════════════════════════════════════════════════════

    Prompt: "The dragon"

    STEP 1:
    ┌─────────────────────────────────────────────────────────────┐
    │  Input: [The] [dragon]                                      │
    │                    │                                        │
    │                    ▼                                        │
    │              ┌─────────┐                                    │
    │              │  MODEL  │                                    │
    │              └────┬────┘                                    │
    │                   │                                         │
    │                   ▼                                         │
    │  Logits for position 2: [0.1, 0.3, 0.05, ..., 0.02]        │
    │                              │                              │
    │                   ┌──────────┴──────────┐                   │
    │                   ▼                     ▼                   │
    │              Temperature            Top-k filter            │
    │              scaling                                        │
    │                   │                     │                   │
    │                   └──────────┬──────────┘                   │
    │                              ▼                              │
    │                          Softmax                            │
    │                              │                              │
    │                              ▼                              │
    │  Probabilities: P("breathes")=0.23, P("flew")=0.18, ...    │
    │                              │                              │
    │                              ▼                              │
    │                      Sample: "breathes"                     │
    └─────────────────────────────────────────────────────────────┘

    STEP 2:
    ┌─────────────────────────────────────────────────────────────┐
    │  Input: [The] [dragon] [breathes]                           │
    │                              │                              │
    │                              ▼                              │
    │                        ┌─────────┐                          │
    │                        │  MODEL  │                          │
    │                        └────┬────┘                          │
    │                             │                               │
    │                             ▼                               │
    │                      Sample: "fire"                         │
    └─────────────────────────────────────────────────────────────┘

    STEP 3:
    ┌─────────────────────────────────────────────────────────────┐
    │  Input: [The] [dragon] [breathes] [fire]                    │
    │                                     │                       │
    │                                     ▼                       │
    │                               ┌─────────┐                   │
    │                               │  MODEL  │                   │
    │                               └────┬────┘                   │
    │                                    │                        │
    │                                    ▼                        │
    │                             Sample: "and"                   │
    └─────────────────────────────────────────────────────────────┘

    ... continue until max_tokens or <EOS> ...

    FINAL OUTPUT: "The dragon breathes fire and smoke fills the sky"
```

---

## 10. Model Size Scaling

```
                    MODEL CONFIGURATIONS
    ════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  SMALL (Default)                                            │
    │  ────────────────                                           │
    │  Layers: 6                                                  │
    │  Embedding: 256                                             │
    │  Heads: 4                                                   │
    │  Parameters: ~10M                                           │
    │                                                             │
    │  Good for: Learning, experimentation, CPU training          │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  MEDIUM                                                     │
    │  ──────                                                     │
    │  Layers: 12                                                 │
    │  Embedding: 512                                             │
    │  Heads: 8                                                   │
    │  Parameters: ~100M                                          │
    │                                                             │
    │  Good for: Serious experiments, single GPU                  │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  LARGE                                                      │
    │  ─────                                                      │
    │  Layers: 24                                                 │
    │  Embedding: 1024                                            │
    │  Heads: 16                                                  │
    │  Parameters: ~1B                                            │
    │                                                             │
    │  Good for: Production quality, multi-GPU training           │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘


    PARAMETER COUNT FORMULA:

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  params ≈ 12 × L × D²                                       │
    │                                                             │
    │  where:                                                     │
    │    L = number of layers                                     │
    │    D = embedding dimension                                  │
    │                                                             │
    │  Example: L=12, D=768  →  12 × 12 × 768² ≈ 85M             │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

---

*Visual Guide for Baby Dragon Hatchling - 2026*
