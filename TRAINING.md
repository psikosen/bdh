# Dragon Hatching Training Guide

Complete guide for training the Dragon Hatching model with monitoring, debugging, and best practices.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training Stages](#training-stages)
3. [Monitoring](#monitoring)
4. [Troubleshooting](#troubleshooting)
5. [Hardware Requirements](#hardware-requirements)
6. [Advanced Configuration](#advanced-configuration)

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/psikosen/bdh.git
cd bdh

# Install dependencies
pip install torch numpy requests datasets tqdm matplotlib

# Optional: Install monitoring tools
pip install tensorboard wandb
```

### Basic Training

```bash
# Train with default settings
python training/run_training.py

# Train with custom config
python training/run_training.py --config configs/small.yaml

# Resume from checkpoint
python training/run_training.py --resume checkpoints/hatching_step_1000.pt
```

### Monitor Training

```bash
# Start monitor in separate terminal
python training/monitor.py --log-dir logs/

# Or use TensorBoard
tensorboard --logdir logs/
```

---

## Training Stages

### Stage 1: Text Coherence (Iterations 0-2000)

**Goal:** Establish base language modeling capability

**Datasets:**
- TinyStories (primary)
- WikiText-103 (secondary)

**Task Mix:**
```yaml
text_completion: 0.70
chain_of_thought: 0.30
```

**Expected Metrics:**
| Metric | Start | End |
|--------|-------|-----|
| Loss | ~5.5 | ~3.0 |
| Perplexity | ~250 | ~20 |
| Coherence Score | 0.3 | 0.6 |

**What to Watch:**
- Loss should decrease smoothly
- No NaN values
- Gradients should be stable (norm < 10)

---

### Stage 2: Function Calling (Iterations 2000-5000)

**Goal:** Learn structured function call output

**Datasets:**
- Glaive Function Calling v2
- BFCL (validation)

**Task Mix:**
```yaml
text_completion: 0.30
function_call: 0.40
chain_of_thought: 0.30
```

**Expected Metrics:**
| Metric | Start | End |
|--------|-------|-----|
| Loss | ~3.0 | ~2.2 |
| Function Format Accuracy | 0.1 | 0.7 |
| Argument Accuracy | 0.0 | 0.5 |

**What to Watch:**
- Function call format should emerge
- JSON validity should increase
- May see loss spike when stage changes (normal)

---

### Stage 3: Bash Commands (Iterations 5000-8000)

**Goal:** Learn bash syntax and command patterns

**Datasets:**
- NL2Bash
- unix-commands

**Task Mix:**
```yaml
text_completion: 0.20
function_call: 0.30
bash_command: 0.30
chain_of_thought: 0.20
```

**Expected Metrics:**
| Metric | Start | End |
|--------|-------|-----|
| Loss | ~2.2 | ~1.8 |
| Bash Format Accuracy | 0.2 | 0.8 |
| Command Validity | 0.1 | 0.6 |

**What to Watch:**
- Code block formatting
- Common command recognition
- Flag syntax accuracy

---

### Stage 4: Full Multi-Task (Iterations 8000-10000)

**Goal:** Integrate all capabilities with tool use

**Datasets:**
- All previous
- ToolBench (if available)

**Task Mix:**
```yaml
text_completion: 0.15
function_call: 0.20
bash_command: 0.20
tool_use: 0.25
chain_of_thought: 0.20
```

**Expected Metrics:**
| Metric | Start | End |
|--------|-------|-----|
| Loss | ~1.8 | ~1.5 |
| Multi-step Accuracy | 0.1 | 0.4 |
| Overall Score | 0.5 | 0.7 |

---

## Monitoring

### Key Metrics to Track

#### 1. Loss Metrics
```
train_loss        - Primary training loss
val_loss          - Validation loss (check overfitting)
task_loss_*       - Per-task losses
sparsity_loss     - Biological regularization
```

#### 2. Gradient Health
```
grad_norm         - Should be < 10, ideally 0.1-2.0
grad_max          - Maximum gradient value
nan_count         - Should always be 0
inf_count         - Should always be 0
```

#### 3. Activation Statistics
```
firing_rate       - Target: ~0.1 (10%)
activation_mean   - Should be stable
activation_std    - Should not explode
```

#### 4. Learning Dynamics
```
learning_rate     - Follows schedule
weight_norm       - Should grow slowly
update_ratio      - |update| / |weight|, target: ~1e-3
```

### Using the Monitor Script

```bash
# Basic monitoring
python training/monitor.py

# With specific log directory
python training/monitor.py --log-dir logs/run_001

# Real-time plotting
python training/monitor.py --plot --refresh 5

# Export metrics to CSV
python training/monitor.py --export metrics.csv
```

### Alert Thresholds

The monitor will alert you when:

| Condition | Severity | Action |
|-----------|----------|--------|
| NaN detected | 🔴 CRITICAL | Stop training, reduce LR |
| Loss > 10 | 🔴 CRITICAL | Check data, reduce LR |
| Grad norm > 100 | 🟡 WARNING | Enable gradient clipping |
| Firing rate > 0.5 | 🟡 WARNING | Increase sparsity weight |
| Loss plateaus 500 iters | 🟡 WARNING | Adjust LR or check data |

---

## Troubleshooting

### NaN Values

**Symptoms:**
- Loss becomes NaN
- Gradients become NaN
- Model outputs garbage

**Causes & Solutions:**

1. **Learning rate too high**
   ```python
   # Reduce learning rate
   learning_rate = 1e-4  # was 1e-3
   ```

2. **Gradient explosion**
   ```python
   # Enable gradient clipping
   grad_clip = 1.0
   ```

3. **Numerical instability in LIF neurons**
   ```python
   # Increase tau_mem for stability
   tau_mem = 20.0  # was 10.0
   ```

4. **Bad data sample**
   ```python
   # Add data validation
   if torch.isnan(loss):
       print(f"Bad sample: {batch}")
       continue
   ```

### Loss Not Decreasing

**Check:**
1. Learning rate (try 3e-4 to 1e-3)
2. Batch size (try 16-64)
3. Data shuffling (ensure enabled)
4. Model initialization

### Out of Memory

**Solutions:**
```python
# Reduce batch size
batch_size = 16  # was 32

# Reduce sequence length
max_seq_len = 256  # was 512

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
dtype = "bfloat16"
```

### Overfitting

**Symptoms:**
- Train loss decreases, val loss increases
- Perfect on training, poor on eval

**Solutions:**
```python
# Increase dropout
dropout = 0.2  # was 0.1

# Add weight decay
weight_decay = 0.1

# Reduce model size
n_layer = 4  # was 6
```

---

## Hardware Requirements

### Minimum (Small Model)
- GPU: 8GB VRAM (RTX 3070, etc.)
- RAM: 16GB
- Storage: 10GB

```python
# Small config
n_layer = 4
n_embd = 128
batch_size = 16
max_seq_len = 256
```

### Recommended (Medium Model)
- GPU: 16GB VRAM (RTX 4080, A4000)
- RAM: 32GB
- Storage: 50GB

```python
# Medium config
n_layer = 6
n_embd = 256
batch_size = 32
max_seq_len = 512
```

### Large Scale
- GPU: 24GB+ VRAM (RTX 4090, A100)
- RAM: 64GB
- Storage: 200GB

```python
# Large config
n_layer = 12
n_embd = 512
batch_size = 64
max_seq_len = 1024
```

---

## Advanced Configuration

### Custom Training Config

Create `configs/custom.yaml`:

```yaml
# Model
model:
  n_layer: 6
  n_embd: 256
  n_head: 4
  dropout: 0.1
  vocab_size: 256

# LIF Neurons
lif:
  tau_mem: 10.0
  tau_syn: 5.0
  v_threshold: 1.0
  v_reset: 0.0

# STDP
stdp:
  tau_plus: 20.0
  tau_minus: 20.0
  A_plus: 0.01
  A_minus: 0.0105

# Training
training:
  batch_size: 32
  max_seq_len: 512
  learning_rate: 1e-3
  weight_decay: 0.1
  max_iters: 10000
  warmup_iters: 500
  grad_clip: 1.0

# Curriculum
curriculum:
  enabled: true
  stages:
    - iters: 2000
      weights: {text_completion: 0.7, chain_of_thought: 0.3}
    - iters: 3000
      weights: {text_completion: 0.3, function_call: 0.4, chain_of_thought: 0.3}
    - iters: 3000
      weights: {text_completion: 0.2, function_call: 0.3, bash_command: 0.3, chain_of_thought: 0.2}
    - iters: 2000
      weights: {text_completion: 0.15, function_call: 0.2, bash_command: 0.2, tool_use: 0.25, chain_of_thought: 0.2}

# Biological Regularization
bio:
  sparsity_target: 0.1
  sparsity_weight: 0.001

# Monitoring
monitor:
  log_freq: 100
  eval_freq: 500
  save_freq: 1000
  nan_check: true
```

### Distributed Training

```bash
# Multi-GPU with torchrun
torchrun --nproc_per_node=4 training/run_training.py --distributed

# With specific GPUs
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 training/run_training.py
```

### Hyperparameter Sweep

```bash
# Using wandb sweeps
wandb sweep configs/sweep.yaml
wandb agent <sweep_id>
```

---

## Checkpoints

### Saving

Checkpoints are saved automatically:
- `hatching_step_N.pt` - Every `save_freq` iterations
- `hatching_best.pt` - Best validation loss
- `hatching_final.pt` - End of training

### Loading

```python
from training.train_multitask import MultiTaskTrainer, TrainingConfig

config = TrainingConfig()
trainer = MultiTaskTrainer(config)

# Resume training
start_step = trainer.load_checkpoint("hatching_step_5000.pt")
trainer.train(start_step=start_step)
```

### Checkpoint Contents

```python
{
    'step': int,
    'model_state': dict,
    'optimizer_state': dict,
    'scheduler_state': dict,
    'config': TrainingConfig,
    'eval_metrics': dict,
    'rng_state': dict,  # For reproducibility
}
```

---

## Best Practices

1. **Always monitor first few hundred iterations closely**
2. **Save checkpoints frequently** (every 500-1000 iters)
3. **Use validation set** to detect overfitting
4. **Start with small model** to debug pipeline
5. **Log everything** - you'll thank yourself later
6. **Version your configs** - track what worked
7. **Test generation periodically** - loss isn't everything

---

## Example Training Run

```bash
# 1. Prepare data
python training/datasets.py  # Downloads NL2Bash

# 2. Start monitoring (in separate terminal)
python training/monitor.py --log-dir logs/run_001 --plot

# 3. Run training
python training/run_training.py \
    --config configs/medium.yaml \
    --log-dir logs/run_001 \
    --name "hatching_v1"

# 4. Evaluate
python training/evaluate.py --checkpoint hatching_best.pt

# 5. Generate samples
python training/generate.py --checkpoint hatching_best.pt --prompt "User: What is the weather?"
```
