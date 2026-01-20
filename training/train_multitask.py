# Copyright 2025 - Multi-Task Training Pipeline for Dragon Hatching
# Unified training for text coherence, function calling, tool use, and bash

import os
import sys
import math
import json
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from contextlib import nullcontext
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hatching import Hatching, HatchingConfig
from training.data_generators import (
    UnifiedDataGenerator, TaskType, TrainingSample,
    FunctionCallGenerator, BashCommandGenerator,
    CoherenceGenerator, ToolUseGenerator, ChainOfThoughtGenerator
)
from training.evaluation import (
    UnifiedEvaluator, CoherenceEvaluator,
    FunctionCallEvaluator, BashEvaluator, ToolUseEvaluator
)


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Multi-task training configuration."""

    # Model
    n_layer: int = 6
    n_embd: int = 256
    n_head: int = 4
    dropout: float = 0.1
    vocab_size: int = 256  # Byte-level

    # Training
    batch_size: int = 32
    max_seq_len: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    max_iters: int = 10000
    warmup_iters: int = 500
    grad_clip: float = 1.0

    # Multi-task weights
    task_weights: Dict[str, float] = field(default_factory=lambda: {
        'text_completion': 0.20,
        'function_call': 0.25,
        'bash_command': 0.20,
        'tool_use': 0.20,
        'chain_of_thought': 0.15,
    })

    # Curriculum learning
    curriculum_enabled: bool = True
    curriculum_stages: List[Dict] = field(default_factory=lambda: [
        # Stage 1: Basic text completion
        {'iters': 2000, 'weights': {'text_completion': 0.7, 'chain_of_thought': 0.3}},
        # Stage 2: Add function calling
        {'iters': 3000, 'weights': {'text_completion': 0.3, 'function_call': 0.4, 'chain_of_thought': 0.3}},
        # Stage 3: Add bash
        {'iters': 3000, 'weights': {'text_completion': 0.2, 'function_call': 0.3, 'bash_command': 0.3, 'chain_of_thought': 0.2}},
        # Stage 4: Full multi-task with tool use
        {'iters': 2000, 'weights': {'text_completion': 0.15, 'function_call': 0.2, 'bash_command': 0.2, 'tool_use': 0.25, 'chain_of_thought': 0.2}},
    ])

    # Biological regularization
    sparsity_target: float = 0.1
    sparsity_weight: float = 0.001

    # Logging
    log_freq: int = 100
    eval_freq: int = 500
    save_freq: int = 1000

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'


# =============================================================================
# DATASET
# =============================================================================

class MultiTaskDataset(Dataset):
    """
    Dynamic multi-task dataset.

    Generates samples on-the-fly with configurable task weights.
    """

    def __init__(
        self,
        config: TrainingConfig,
        task_weights: Dict[str, float] = None,
        max_seq_len: int = 512
    ):
        self.config = config
        self.max_seq_len = max_seq_len
        self.task_weights = task_weights or config.task_weights

        # Initialize generators
        self.generators = {
            'text_completion': CoherenceGenerator(),
            'function_call': FunctionCallGenerator(),
            'bash_command': BashCommandGenerator(),
            'tool_use': ToolUseGenerator(),
            'chain_of_thought': ChainOfThoughtGenerator(),
        }

        # Normalize weights
        total = sum(self.task_weights.values())
        self.task_weights = {k: v/total for k, v in self.task_weights.items()}

    def set_task_weights(self, weights: Dict[str, float]):
        """Update task weights (for curriculum learning)."""
        total = sum(weights.values())
        self.task_weights = {k: v/total for k, v in weights.items()}

    def __len__(self):
        # Return a large number for infinite-style dataset
        return 100000

    def __getitem__(self, idx):
        # Sample task type based on weights
        task_names = list(self.task_weights.keys())
        task_probs = list(self.task_weights.values())
        task_name = random.choices(task_names, weights=task_probs)[0]

        # Generate sample
        generator = self.generators.get(task_name)
        if generator is None:
            # Fallback to text completion
            generator = self.generators['text_completion']

        sample = generator.generate_sample()

        # Convert to tokens (byte-level)
        full_text = sample.input_text + sample.target_text
        tokens = list(full_text.encode('utf-8'))

        # Truncate or pad
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        elif len(tokens) < self.max_seq_len:
            tokens = tokens + [0] * (self.max_seq_len - len(tokens))

        # Create input/target pairs
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)

        # Task type as integer for tracking
        task_id = list(self.generators.keys()).index(task_name)

        return x, y, task_id


def collate_fn(batch):
    """Collate function for DataLoader."""
    xs, ys, task_ids = zip(*batch)
    return (
        torch.stack(xs),
        torch.stack(ys),
        torch.tensor(task_ids)
    )


# =============================================================================
# TRAINER
# =============================================================================

class MultiTaskTrainer:
    """
    Multi-task trainer with curriculum learning.

    Features:
    - Curriculum learning (gradually introduce tasks)
    - Task-specific loss weighting
    - Biological regularization (sparsity)
    - Comprehensive evaluation
    """

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Setup device
        self.device = torch.device(config.device)
        self.ptdtype = {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
        }[config.dtype]

        # Autocast context
        self.ctx = (
            torch.amp.autocast(device_type=self.device.type, dtype=self.ptdtype)
            if 'cuda' in config.device
            else nullcontext()
        )
        self.scaler = torch.amp.GradScaler(
            device=self.device.type,
            enabled=(config.dtype == 'float16')
        )

        # Initialize model
        hatching_config = HatchingConfig(
            n_layer=config.n_layer,
            n_embd=config.n_embd,
            n_head=config.n_head,
            dropout=config.dropout,
            vocab_size=config.vocab_size,
        )
        self.model = Hatching(hatching_config).to(self.device)

        # Try to compile
        try:
            self.model = torch.compile(self.model)
            print("Model compiled with torch.compile()")
        except Exception as e:
            print(f"torch.compile() not available: {e}")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95)
        )

        # Learning rate scheduler (cosine with warmup)
        def lr_lambda(step):
            if step < config.warmup_iters:
                return step / config.warmup_iters
            decay_ratio = (step - config.warmup_iters) / (config.max_iters - config.warmup_iters)
            return 0.1 + 0.9 * (1 + math.cos(math.pi * decay_ratio)) / 2

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Dataset
        self.dataset = MultiTaskDataset(config, config.task_weights, config.max_seq_len)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0  # Keep simple for now
        )

        # Evaluator
        self.evaluator = UnifiedEvaluator()

        # Metrics tracking
        self.metrics_history = defaultdict(list)
        self.current_stage = 0

        print(f"Initialized trainer on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def get_curriculum_weights(self, step: int) -> Dict[str, float]:
        """Get task weights for current curriculum stage."""
        if not self.config.curriculum_enabled:
            return self.config.task_weights

        # Find current stage
        cumulative_iters = 0
        for i, stage in enumerate(self.config.curriculum_stages):
            cumulative_iters += stage['iters']
            if step < cumulative_iters:
                if i != self.current_stage:
                    self.current_stage = i
                    print(f"\n>>> Curriculum Stage {i+1}: {stage['weights']}")
                return stage['weights']

        # Past all stages, use final weights
        return self.config.curriculum_stages[-1]['weights']

    def compute_sparsity_loss(self, states: Dict) -> torch.Tensor:
        """Compute sparsity regularization loss."""
        if states is None or 'traces' not in states:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)
        for trace in states['traces']:
            if trace is not None:
                rate = trace.mean()
                loss = loss + (rate - self.config.sparsity_target) ** 2

        return loss * self.config.sparsity_weight

    def train_step(self, batch, step: int) -> Dict[str, float]:
        """Single training step."""
        x, y, task_ids = batch
        x = x.to(self.device)
        y = y.to(self.device)

        # Forward pass
        with self.ctx:
            logits, loss, states = self.model(x, y, return_states=True)

            # Add sparsity loss
            if step % 10 == 0:  # Every 10 steps for efficiency
                sparsity_loss = self.compute_sparsity_loss(states)
                loss = loss + sparsity_loss
            else:
                sparsity_loss = torch.tensor(0.0)

        # Backward pass
        self.scaler.scale(loss).backward()

        # Gradient clipping
        if self.config.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.scheduler.step()

        # Per-task loss tracking
        task_losses = defaultdict(list)
        task_names = list(self.dataset.generators.keys())
        for i, tid in enumerate(task_ids):
            task_losses[task_names[tid.item()]].append(loss.item())

        return {
            'loss': loss.item(),
            'sparsity_loss': sparsity_loss.item() if isinstance(sparsity_loss, torch.Tensor) else sparsity_loss,
            'lr': self.scheduler.get_last_lr()[0],
            'task_losses': {k: sum(v)/len(v) for k, v in task_losses.items() if v}
        }

    @torch.no_grad()
    def evaluate(self, n_samples: int = 50) -> Dict[str, float]:
        """Evaluate model on generated samples."""
        self.model.eval()

        metrics = {
            'coherence': [],
            'function_call': [],
            'bash': [],
            'tool_use': [],
        }

        for task_name, generator in self.dataset.generators.items():
            for _ in range(n_samples // 5):  # 10 samples per task
                # Generate a sample
                sample = generator.generate_sample()

                # Get model completion
                prompt_tokens = list(sample.input_text.encode('utf-8'))
                prompt = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)

                # Generate
                generated = self.model.generate(
                    prompt,
                    max_new_tokens=100,
                    temperature=0.8,
                    top_k=40
                )

                # Decode
                generated_text = bytes(generated[0].cpu().tolist()).decode(errors='replace')

                # Evaluate
                eval_result = self.evaluator.evaluate(generated_text, task_name)

                metrics['coherence'].append(eval_result.coherence.overall_score)
                if task_name == 'function_call':
                    metrics['function_call'].append(eval_result.function_call.overall_score)
                elif task_name == 'bash_command':
                    metrics['bash'].append(eval_result.bash.overall_score)
                elif task_name == 'tool_use':
                    metrics['tool_use'].append(eval_result.tool_use.overall_score)

        self.model.train()

        return {
            k: sum(v) / len(v) if v else 0.0
            for k, v in metrics.items()
        }

    def train(self):
        """Main training loop."""
        print("=" * 60)
        print("DRAGON HATCHING - Multi-Task Training")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Max iterations: {self.config.max_iters}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Curriculum: {self.config.curriculum_enabled}")
        print()

        # Training loop
        data_iter = iter(self.dataloader)
        loss_acc = 0.0
        loss_count = 0
        best_eval_score = 0.0

        for step in range(self.config.max_iters):
            # Update curriculum
            weights = self.get_curriculum_weights(step)
            self.dataset.set_task_weights(weights)

            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            # Train step
            step_metrics = self.train_step(batch, step)
            loss_acc += step_metrics['loss']
            loss_count += 1

            # Logging
            if step % self.config.log_freq == 0:
                avg_loss = loss_acc / loss_count

                task_loss_str = ", ".join(
                    f"{k[:4]}={v:.3f}"
                    for k, v in step_metrics['task_losses'].items()
                )

                print(
                    f"Step {step:5d}/{self.config.max_iters} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {step_metrics['lr']:.2e} | "
                    f"Stage: {self.current_stage+1} | "
                    f"Tasks: [{task_loss_str}]"
                )

                self.metrics_history['loss'].append(avg_loss)
                loss_acc = 0.0
                loss_count = 0

            # Evaluation
            if step > 0 and step % self.config.eval_freq == 0:
                print("\n--- Evaluation ---")
                eval_metrics = self.evaluate()

                for k, v in eval_metrics.items():
                    print(f"  {k}: {v:.4f}")
                    self.metrics_history[f'eval_{k}'].append(v)

                # Track best
                overall = sum(eval_metrics.values()) / len(eval_metrics)
                if overall > best_eval_score:
                    best_eval_score = overall
                    self.save_checkpoint('hatching_best.pt', step, eval_metrics)
                    print(f"  >>> New best: {overall:.4f}")

                print()

            # Save checkpoint
            if step > 0 and step % self.config.save_freq == 0:
                self.save_checkpoint(f'hatching_step_{step}.pt', step)

        print("=" * 60)
        print("Training complete!")
        print(f"Best evaluation score: {best_eval_score:.4f}")

        # Final save
        self.save_checkpoint('hatching_final.pt', self.config.max_iters)

        return self.metrics_history

    def save_checkpoint(self, filename: str, step: int, eval_metrics: Dict = None):
        """Save training checkpoint."""
        checkpoint = {
            'step': step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'config': self.config,
            'eval_metrics': eval_metrics,
        }
        torch.save(checkpoint, filename)
        print(f"  Saved checkpoint: {filename}")

    def load_checkpoint(self, filename: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        print(f"Loaded checkpoint from step {checkpoint['step']}")
        return checkpoint['step']


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    # Configuration
    config = TrainingConfig(
        n_layer=4,  # Smaller for faster training
        n_embd=128,
        n_head=4,
        batch_size=16,
        max_seq_len=256,
        learning_rate=3e-4,
        max_iters=5000,
        warmup_iters=200,
        curriculum_enabled=True,
        log_freq=50,
        eval_freq=250,
        save_freq=500,
    )

    # Create trainer
    trainer = MultiTaskTrainer(config)

    # Train
    metrics = trainer.train()

    # Sample generation
    print("\n" + "=" * 60)
    print("SAMPLE GENERATIONS")
    print("=" * 60)

    trainer.model.eval()

    prompts = [
        "User: What is the weather in Tokyo?\nAssistant:",
        "User: How do I list all Python files?\nAssistant:",
        "User: Explain machine learning briefly.\nAssistant:",
        "User: Help me analyze this project step by step.\nAssistant:",
    ]

    for prompt in prompts:
        tokens = torch.tensor(
            [list(prompt.encode('utf-8'))],
            dtype=torch.long,
            device=trainer.device
        )
        generated = trainer.model.generate(tokens, max_new_tokens=100, temperature=0.7, top_k=40)
        text = bytes(generated[0].cpu().tolist()).decode(errors='replace')
        print(f"\n{'-'*40}")
        print(text[:300])

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
