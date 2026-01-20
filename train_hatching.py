# Copyright 2025 - Training script for Dragon Hatching Model
# Enhanced biologically-inspired language model training

import os
from contextlib import nullcontext
from typing import Optional

import numpy as np
import requests
import torch
from torch import nn

import hatching
from hatching import Hatching, HatchingConfig, compute_firing_rate

# =============================================================================
# DEVICE SETUP
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]

ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if "cuda" in device.type
    else nullcontext()
)
scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print(f"Using device: {device} with dtype {dtype}")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model configuration with biologically-inspired parameters
HATCHING_CONFIG = HatchingConfig(
    n_layer=6,
    n_embd=256,
    dropout=0.1,
    n_head=4,
    mlp_internal_dim_multiplier=128,
    vocab_size=256,

    # LIF parameters (tuned for stability)
    tau_mem=10.0,       # Membrane time constant
    tau_syn=5.0,        # Synaptic time constant
    v_threshold=1.0,    # Spike threshold
    v_reset=0.0,        # Reset potential
    v_rest=0.0,         # Resting potential

    # STDP parameters
    tau_plus=20.0,      # LTP time constant
    tau_minus=20.0,     # LTD time constant
    A_plus=0.01,        # LTP amplitude
    A_minus=0.0105,     # LTD amplitude

    # E/I balance
    excitatory_ratio=0.8,

    # Scale-free network
    scale_free_gamma=2.5,

    # Hebbian trace
    trace_decay=0.95
)

# Training configuration
BLOCK_SIZE = 512
BATCH_SIZE = 32
MAX_ITERS = 3000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
LOG_FREQ = 100
GRAD_CLIP = 1.0

# Biological regularization
SPARSITY_TARGET = 0.1  # Target firing rate
SPARSITY_WEIGHT = 0.001  # Regularization strength

input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")


# =============================================================================
# DATA LOADING
# =============================================================================

def fetch_data():
    """Fetch the tiny Shakespeare dataset."""
    if not os.path.exists(input_file_path):
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading data from {data_url}...")
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)
        print("Data downloaded.")


def get_batch(split: str):
    """Get a batch of data for training or validation."""
    data = np.memmap(input_file_path, dtype=np.uint8, mode="r")
    if split == "train":
        data = data[: int(0.9 * len(data))]
    else:
        data = data[int(0.9 * len(data)):]

    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([
        torch.from_numpy((data[i : i + BLOCK_SIZE]).astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy((data[i + 1 : i + 1 + BLOCK_SIZE]).astype(np.int64))
        for i in ix
    ])

    if torch.cuda.is_available():
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


# =============================================================================
# BIOLOGICAL REGULARIZATION
# =============================================================================

def compute_sparsity_loss(model: Hatching, x: torch.Tensor) -> torch.Tensor:
    """
    Compute sparsity regularization loss.

    Encourages biologically plausible firing rates by penalizing
    deviation from target sparsity level.

    L_sparse = Σ (rate_i - target)²
    """
    # Get firing statistics
    with torch.no_grad():
        _, _, states = model(x, return_states=True)

    total_loss = torch.tensor(0.0, device=x.device)
    for trace in states.get('traces', []):
        if trace is not None:
            # Compute mean firing rate
            rate = trace.mean()
            # MSE from target
            total_loss = total_loss + (rate - SPARSITY_TARGET) ** 2

    return total_loss


def compute_balance_loss(model: Hatching) -> torch.Tensor:
    """
    Compute E/I balance regularization.

    Encourages balanced excitation and inhibition for stable dynamics.
    """
    balance_loss = torch.tensor(0.0, device=device)

    for block in model.blocks:
        # Get encoder weights
        W = block.encoder
        # Compute mean absolute weight
        e_strength = W[..., :int(W.shape[-1] * 0.8)].abs().mean()
        i_strength = W[..., int(W.shape[-1] * 0.8):].abs().mean()

        # Target: I slightly stronger than E for stability
        target_ratio = 1.2
        actual_ratio = i_strength / (e_strength + 1e-6)
        balance_loss = balance_loss + (actual_ratio - target_ratio) ** 2

    return balance_loss


# =============================================================================
# TRAINING LOOP
# =============================================================================

@torch.no_grad()
def evaluate(model: Hatching, n_batches: int = 10) -> dict:
    """Evaluate model on validation set."""
    model.eval()
    losses = []
    firing_stats_list = []

    for _ in range(n_batches):
        x, y = get_batch("val")
        with ctx:
            logits, loss, _ = model(x, y)
        losses.append(loss.item())

        # Compute firing statistics
        stats = compute_firing_rate(model, x)
        firing_stats_list.append(stats)

    model.train()

    # Aggregate firing stats
    avg_firing_rate = np.mean([
        s.get('layer_0', {}).get('mean_firing_rate', 0)
        for s in firing_stats_list
    ])

    return {
        'val_loss': np.mean(losses),
        'avg_firing_rate': avg_firing_rate
    }


def train():
    """Main training loop."""
    print("=" * 60)
    print("DRAGON HATCHING - Enhanced Biologically-Inspired LLM")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Layers: {HATCHING_CONFIG.n_layer}")
    print(f"  Embedding: {HATCHING_CONFIG.n_embd}")
    print(f"  Heads: {HATCHING_CONFIG.n_head}")
    print(f"  τ_mem: {HATCHING_CONFIG.tau_mem}")
    print(f"  τ_STDP+: {HATCHING_CONFIG.tau_plus}")
    print(f"  E/I ratio: {HATCHING_CONFIG.excitatory_ratio}")
    print(f"  Scale-free γ: {HATCHING_CONFIG.scale_free_gamma}")
    print()

    # Fetch data
    fetch_data()

    # Initialize model
    model = Hatching(HATCHING_CONFIG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Try to compile (PyTorch 2.0+)
    try:
        model = torch.compile(model)
        print("Model compiled with torch.compile()")
    except Exception as e:
        print(f"torch.compile() not available: {e}")

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95)  # Slightly different betas for stability
    )

    # Learning rate scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_ITERS,
        eta_min=LEARNING_RATE / 10
    )

    # Training state
    loss_acc = 0.0
    loss_steps = 0
    best_val_loss = float('inf')

    print(f"\nStarting training for {MAX_ITERS} iterations...")
    print("-" * 60)

    for step in range(MAX_ITERS):
        # Get batch
        x, y = get_batch("train")

        # Forward pass
        with ctx:
            logits, loss, states = model(x, y, return_states=True)

            # Add biological regularization (every 10 steps for efficiency)
            if step % 10 == 0:
                # Sparsity regularization
                sparsity_loss = torch.tensor(0.0, device=device)
                if states and 'traces' in states:
                    for trace in states['traces']:
                        if trace is not None:
                            rate = trace.mean()
                            sparsity_loss = sparsity_loss + (rate - SPARSITY_TARGET) ** 2
                loss = loss + SPARSITY_WEIGHT * sparsity_loss

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient clipping
        if GRAD_CLIP > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        # Accumulate loss
        loss_acc += loss.item()
        loss_steps += 1

        # Logging
        if step % LOG_FREQ == 0:
            avg_loss = loss_acc / loss_steps

            # Evaluate
            eval_stats = evaluate(model)

            # Print progress
            lr = scheduler.get_last_lr()[0]
            print(
                f"Step {step:5d}/{MAX_ITERS} | "
                f"Loss: {avg_loss:.4f} | "
                f"Val: {eval_stats['val_loss']:.4f} | "
                f"LR: {lr:.2e} | "
                f"FR: {eval_stats['avg_firing_rate']:.3f}"
            )

            # Reset accumulator
            loss_acc = 0.0
            loss_steps = 0

            # Save best model
            if eval_stats['val_loss'] < best_val_loss:
                best_val_loss = eval_stats['val_loss']
                torch.save({
                    'model': model.state_dict(),
                    'config': HATCHING_CONFIG,
                    'step': step,
                    'val_loss': best_val_loss
                }, 'hatching_best.pt')

    print("-" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Generate sample
    print("\nGenerating sample...")
    model.eval()
    prompt = torch.tensor(
        bytearray("To be or ", "utf-8"),
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    generated = model.generate(prompt, max_new_tokens=200, top_k=5, temperature=0.8)
    text = bytes(generated.to(torch.uint8).cpu().squeeze(0)).decode(errors='backslashreplace')

    print("\n" + "=" * 60)
    print("GENERATED TEXT:")
    print("=" * 60)
    print(text)
    print("=" * 60)

    # Final analysis
    print("\nFiring rate analysis:")
    stats = compute_firing_rate(model, x)
    for layer, layer_stats in stats.items():
        print(f"  {layer}: rate={layer_stats['mean_firing_rate']:.4f}, "
              f"active={layer_stats['active_fraction']:.2%}")


if __name__ == "__main__":
    train()
